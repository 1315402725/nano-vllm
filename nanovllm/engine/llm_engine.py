import atexit
from dataclasses import fields
from time import perf_counter
from tqdm.auto import tqdm
from transformers import AutoTokenizer
import torch.multiprocessing as mp

from nanovllm.config import Config
from nanovllm.sampling_params import SamplingParams
from nanovllm.engine.sequence import Sequence
from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.model_runner import ModelRunner


class LLMEngine:

    def __init__(self, model, **kwargs):
        # fields(Config) 读取配置数据类的字段；集合推导式收集所有合法参数名，例如 tensor_parallel_size。
        config_fields = {field.name for field in fields(Config)}
        # 从调用者传入的 kwargs 中筛选 Config 支持的参数；此处不在字段集合中的参数会被忽略。
        config_kwargs = {k: v for k, v in kwargs.items() if k in config_fields}
        # 创建配置对象：model 是模型目录，**config_kwargs 将字典展开为关键字参数，未传的字段使用默认值。
        # Config 的 __post_init__ 会校验目录和部分配置、读取模型 config.json，并限制 max_model_len 不超过模型位置上限。
        config = Config(model, **config_kwargs)
        # 设置 Sequence 的类属性，让序列按同一个块大小计算 token 与 KV Cache 块的对应关系；此处不分配显存。
        Sequence.block_size = config.kvcache_block_size
        # 保存后续创建的子进程对象，退出时通过它们等待子进程结束；不包含当前主进程。
        self.ps = []
        # 保存每个子进程对应的通知事件；主进程写入共享内存命令后，用事件唤醒对应子进程。
        self.events = []
        # 选择 spawn 启动方式：子进程使用独立的 Python 解释器启动，而不是直接 fork 复制父进程的运行状态。
        ctx = mp.get_context("spawn")
        # rank 0 由当前主进程负责，只为 rank 1 到 N-1 创建子进程；单卡 N=1 时跳过整个循环。
        for i in range(1, config.tensor_parallel_size):
            # 创建跨进程通知事件，初始未触发；这里只负责通知，具体命令和参数另通过共享内存传递。
            event = ctx.Event()
            # 定义子进程任务：启动后执行 ModelRunner(config, i, event)，i 是进程 rank，也用于选择 GPU。
            # 此行只创建进程对象，尚未启动；spawn 会将参数传到子进程，config 不与主进程共享后续修改。
            process = ctx.Process(target=ModelRunner, args=(config, i, event))
            # 真正启动子进程；子进程初始化自己的模型分片和 KV Cache，随后进入等待命令的循环。
            # start() 不等待模型初始化完成，主进程会继续启动其他 rank；各 rank 后续通过分布式通信进行同步。
            process.start()
            # 保存进程对象，方便退出时调用 join() 等待该进程结束。
            self.ps.append(process)
            # 保存事件，供 rank 0 的 ModelRunner 通知这个子进程执行任务。
            self.events.append(event)
        # 在当前主进程中创建 rank 0 的运行器：初始化通信、绑定 GPU 0、构建模型并加载本卡权重、预热及分配 KV Cache。
        # 若未强制 eager，还会捕获 CUDA Graph；多卡时还会创建命令共享内存。此调用完成后主进程才继续向下执行。
        # 运行器会将实际 KV Cache 块数写回当前 config，后面创建调度器时会用到。
        self.model_runner = ModelRunner(config, 0, self.events)
        # 从模型目录加载分词器，用于输入文本转 token ID、输出 token ID 转文本；use_fast=True 优先使用快速实现。
        self.tokenizer = AutoTokenizer.from_pretrained(config.model, use_fast=True)
        # 记录结束 token 的 ID，供调度器判断生成是否结束；这里先赋值，下一行的调度器才能读取正确的值。
        config.eos = self.tokenizer.eos_token_id
        # 创建调度器：初始化等待/运行队列，以及管理 KV Cache 块分配、释放和前缀复用的 BlockManager。
        # GPU 上实际的 KV Cache 存储已由 ModelRunner 分配，这里主要管理块编号和使用状态。
        self.scheduler = Scheduler(config)
        # 注册正常解释器退出时的清理回调；这里只登记方法，不立即执行，退出时通知工作进程结束并等待它们退出。
        atexit.register(self.exit)

    def exit(self):
        self.model_runner.call("exit")
        del self.model_runner
        for p in self.ps:
            p.join()

    def add_request(self, prompt: str | list[int], sampling_params: SamplingParams):
        if isinstance(prompt, str):
            ##文本转tokenid
            prompt = self.tokenizer.encode(prompt)
        seq = Sequence(prompt, sampling_params)
        self.scheduler.add(seq)

    def step(self):
        seqs, is_prefill = self.scheduler.schedule() # 获取要处理的seq以及本轮是否执行prefill
        num_tokens = sum(seq.num_scheduled_tokens for seq in seqs) if is_prefill else -len(seqs)
        token_ids = self.model_runner.call("run", seqs, is_prefill)
        self.scheduler.postprocess(seqs, token_ids, is_prefill)
        outputs = [(seq.seq_id, seq.completion_token_ids) for seq in seqs if seq.is_finished]
        return outputs, num_tokens

    def is_finished(self):
        return self.scheduler.is_finished()

    def generate(
        self,
        prompts: list[str] | list[list[int]],
        sampling_params: SamplingParams | list[SamplingParams],
        use_tqdm: bool = True,
    ) -> list[str]:
        pbar = tqdm(total=len(prompts), desc="Generating", dynamic_ncols=True, disable=not use_tqdm)
        if not isinstance(sampling_params, list):
            sampling_params = [sampling_params] * len(prompts)
        for prompt, sp in zip(prompts, sampling_params):
            self.add_request(prompt, sp)
        outputs = {}
        prefill_throughput = decode_throughput = 0.
        while not self.is_finished():
            t = perf_counter()
            output, num_tokens = self.step()
            if num_tokens > 0:
                prefill_throughput = num_tokens / (perf_counter() - t)
            else:
                decode_throughput = -num_tokens / (perf_counter() - t)
            pbar.set_postfix({
                "Prefill": f"{int(prefill_throughput)}tok/s",
                "Decode": f"{int(decode_throughput)}tok/s",
            })
            for seq_id, token_ids in output:
                outputs[seq_id] = token_ids
                pbar.update(1)
        pbar.close()
        outputs = [outputs[seq_id] for seq_id in sorted(outputs.keys())]
        outputs = [{"text": self.tokenizer.decode(token_ids), "token_ids": token_ids} for token_ids in outputs]
        return outputs
