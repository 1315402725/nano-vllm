import pickle
import torch
import torch.distributed as dist
from multiprocessing.synchronize import Event
from multiprocessing.shared_memory import SharedMemory

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence
from nanovllm.models.qwen3 import Qwen3ForCausalLM
from nanovllm.layers.sampler import Sampler
from nanovllm.utils.context import set_context, get_context, reset_context
from nanovllm.utils.loader import load_model


class ModelRunner:

    def __init__(self, config: Config, rank: int, event: Event | list[Event]):
        # 保存引擎配置，后续预热、显存规划等操作会使用它；这里保存对象引用，不复制配置。
        self.config = config
        # 取出从模型目录读取的 Hugging Face 配置，包含层数、隐藏维度、注意力头数和权重数据类型等。
        hf_config = config.hf_config
        # 每个 KV Cache 块能容纳的 token 数量，用于计算缓存容量和 token 的存储位置。
        self.block_size = config.kvcache_block_size
        # True 表示不使用 CUDA Graph；它不代表禁用模型内部所有 torch.compile 优化。
        self.enforce_eager = config.enforce_eager
        # 张量并行进程总数；本项目每个进程对应一张 GPU，各进程合作执行同一个模型。
        self.world_size = config.tensor_parallel_size
        # 当前进程在通信组中的编号，从 0 开始；本项目也用它选择当前进程使用的 GPU。
        self.rank = rank
        # rank 0 接收所有子进程的 Event 列表，用于发送通知；其他 rank 只接收自己的 Event，用于等待通知。
        self.event = event

        # 建立默认分布式通信组，NCCL 用于 GPU 间的集合通信，例如 all_reduce 和 gather。
        # localhost:2333 是同机进程建立通信组的协调地址，不是对外提供推理服务的 HTTP 端口。
        # 所有 rank 都要执行此初始化；world_size 是进程总数，rank 是当前进程编号。
        dist.init_process_group("nccl", "tcp://localhost:2333", world_size=self.world_size, rank=rank)
        # 将当前 CUDA 设备设为 rank 对应的 GPU；这里的编号是当前进程可见的 GPU 编号。
        torch.cuda.set_device(rank)
        # 保存原来的默认浮点类型，初始化结束后恢复，避免影响后续没有显式指定类型的张量创建。
        default_dtype = torch.get_default_dtype()
        # 默认浮点类型改为模型配置中的类型，让后续未指定 dtype 的模型权重和缓存按此类型创建。
        torch.set_default_dtype(hf_config.dtype)
        # 默认设备改为当前 CUDA 设备，让后续未指定 device 的张量直接在本卡上创建。
        torch.set_default_device("cuda")
        # 根据配置构建 Qwen3 网络并分配参数；并行层会根据 rank/world_size 创建本卡负责的权重形状。
        # 此时只是构建模型，训练好的权重数值还需要下一行加载。
        self.model = Qwen3ForCausalLM(hf_config)
        # 从模型目录读取训练权重，通过各参数的加载方法完成名称映射、QKV 等权重合并及本卡分片加载。
        load_model(self.model, config.model)
        # 创建采样器，用于根据 logits 和温度选择下一个 token；正式运行时由 rank 0 负责采样。
        self.sampler = Sampler()
        # 使用构造的 token 序列执行预热，触发计算路径并记录显存峰值，为后续 KV Cache 容量估算提供依据。
        self.warmup_model()
        # 根据显存预算和预热开销计算块数，在本卡分配各层、本卡 KV 头的缓存，并将缓存视图绑定到注意力层。
        self.allocate_kv_cache()
        # 未强制 eager 时，准备 CUDA Graph 加速路径。
        if not self.enforce_eager:
            # 为预设的 decode 批次大小捕获 GPU 执行图，后续可回放，减少反复提交 GPU 操作的开销。
            self.capture_cudagraph()
        # 将后续默认张量创建位置设回 CPU；不会把已经创建的模型权重或 KV Cache 从 GPU 搬回 CPU。
        torch.set_default_device("cpu")
        # 恢复原来的默认浮点类型；已创建的张量仍保留各自的数据类型。
        torch.set_default_dtype(default_dtype)

        # 多卡时额外建立主进程向工作进程传递命令的通道；单卡不需要这段逻辑。
        if self.world_size > 1:
            # rank 0 负责创建共享内存，并在初始化返回后接收引擎的调用。
            if rank == 0:
                # 在主机内存中创建名为 nanovllm 的 1 MiB 共享区，存放序列化后的方法名和参数，不是 GPU KV Cache。
                self.shm = SharedMemory(name="nanovllm", create=True, size=2**20)
                # 所有 rank 在这里同步；rank 0 先创建共享内存再到达屏障，保证其他 rank 通过屏障后可以打开它。
                dist.barrier()
            # 非 0 rank 是工作进程，连接同一块共享内存并等待任务。
            else:
                # 等待包括 rank 0 在内的所有进程到达屏障，避免共享内存尚未创建就尝试打开。
                dist.barrier()
                # 按名称连接 rank 0 已创建的共享内存；不会再创建一份独立的命令存储区。
                self.shm = SharedMemory(name="nanovllm")
                # 进入常驻循环：等待 Event 通知、读取命令并执行；收到 exit 命令后才结束循环。
                # 所以工作进程会停留在这个构造方法中处理任务，而 rank 0 会正常完成初始化并返回。
                self.loop()

    def exit(self):
        if self.world_size > 1:
            self.shm.close()
            dist.barrier()
            if self.rank == 0:
                self.shm.unlink()
        if not self.enforce_eager:
            del self.graphs, self.graph_pool
        torch.cuda.synchronize()
        dist.destroy_process_group()

    def loop(self):
        while True:
            method_name, args = self.read_shm()
            self.call(method_name, *args)
            if method_name == "exit":
                break

    def read_shm(self):
        assert self.world_size > 1 and self.rank > 0
        self.event.wait()
        n = int.from_bytes(self.shm.buf[0:4], "little")
        method_name, *args = pickle.loads(self.shm.buf[4:n+4])
        self.event.clear()
        return method_name, args

    def write_shm(self, method_name, *args):
        assert self.world_size > 1 and self.rank == 0
        data = pickle.dumps([method_name, *args])#把 Python 对象序列化成字节串（bytes），方便保存或传递
        n = len(data)
        self.shm.buf[0:4] = n.to_bytes(4, "little") #将长度 n 转成 4 个字节，写入共享内存的前 4 字节
        self.shm.buf[4:n+4] = data #接收端先读取前 4 字节得到 n，再读取 [4:n+4]，就知道这条消息有多长、应该读到哪里停止。
        for event in self.event:
            event.set() #只是通知其他进程“数据已写好，可以读取了”

    def call(self, method_name, *args):
        if self.world_size > 1 and self.rank == 0:
            self.write_shm(method_name, *args)
        method = getattr(self, method_name, None)
        return method(*args)

    def warmup_model(self):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        max_num_batched_tokens, max_model_len = self.config.max_num_batched_tokens, self.config.max_model_len
        seq_len = min(max_num_batched_tokens, max_model_len)
        num_seqs = min(max_num_batched_tokens // seq_len, self.config.max_num_seqs)
        seqs = [Sequence([0] * seq_len) for _ in range(num_seqs)]
        for seq in seqs:
            seq.num_scheduled_tokens = seq_len
        self.run(seqs, True)
        torch.cuda.empty_cache()

    def allocate_kv_cache(self):
        config = self.config
        hf_config = config.hf_config
        free, total = torch.cuda.mem_get_info()
        used = total - free
        peak = torch.cuda.memory_stats()["allocated_bytes.all.peak"]
        current = torch.cuda.memory_stats()["allocated_bytes.all.current"]
        num_kv_heads = hf_config.num_key_value_heads // self.world_size
        head_dim = getattr(hf_config, "head_dim", hf_config.hidden_size // hf_config.num_attention_heads)
        block_bytes = 2 * hf_config.num_hidden_layers * self.block_size * num_kv_heads * head_dim * hf_config.dtype.itemsize
        config.num_kvcache_blocks = int(total * config.gpu_memory_utilization - used - peak + current) // block_bytes
        assert config.num_kvcache_blocks > 0
        self.kv_cache = torch.empty(2, hf_config.num_hidden_layers, config.num_kvcache_blocks, self.block_size, num_kv_heads, head_dim)
        layer_id = 0
        for module in self.model.modules():
            if hasattr(module, "k_cache") and hasattr(module, "v_cache"):
                module.k_cache = self.kv_cache[0, layer_id]
                module.v_cache = self.kv_cache[1, layer_id]
                layer_id += 1

    def prepare_block_tables(self, seqs: list[Sequence]):
        max_len = max(len(seq.block_table) for seq in seqs)
        #把每条序列的 block_table 用 -1 补齐到相同长度，再收集成一个二维列表
        block_tables = [seq.block_table + [-1] * (max_len - len(seq.block_table)) for seq in seqs]
        block_tables = torch.tensor(block_tables, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        return block_tables

    def prepare_prefill(self, seqs: list[Sequence]):
        #名字里的 q、k 表示它们分别描述 Q 侧和 K 侧的长度：
        #Q 长度：这条序列本轮新计算多少个 token，即 num_scheduled_tokens。
        #K 长度：本轮计算需要用到多少个 token 的 Key，包含已有缓存和本轮新增，即
        input_ids = []       # 本批送入模型的 token id（已剔除前缀缓存命中的部分）
        positions = []       # 每个 token 的绝对位置，用于 RoPE
        cu_seqlens_q = [0]   # query 侧的累积长度前缀和，[0]=0，每条序列追加自己长度
        cu_seqlens_k = [0]   # key 侧的累积长度前缀和，命中前缀缓存时比 q 侧长
        max_seqlen_q = 0     # 本批中单条序列的最大 q 长度，attention kernel 据此分配缓冲区
        max_seqlen_k = 0     # 本批中单条序列的最大 k 长度
        slot_mapping = []    # 每个 token 写入 paged KV cache 的物理 slot 下标
        block_tables = None  # 每条序列的 block 物理 id 映射表，仅命中前缀缓存时需要
        for seq in seqs:
            start = seq.num_cached_tokens
            seqlen_q = seq.num_scheduled_tokens
            end = start + seqlen_q
            seqlen_k = end
            input_ids.extend(seq[start:end]) # seq[start:end]) 获取要计算的tokenids数组， 加到inputids里
            positions.extend(range(start, end))
            cu_seqlens_q.append(cu_seqlens_q[-1] + seqlen_q) # 记录每个q长度的起始位置
            cu_seqlens_k.append(cu_seqlens_k[-1] + seqlen_k) # 记录每个key长度的起始位置
            # 记录这批请求最大q长度和k长度
            max_seqlen_q = max(seqlen_q, max_seqlen_q)
            max_seqlen_k = max(seqlen_k, max_seqlen_k)
            if not seq.block_table:    # warmup
                continue
            start_block = start // self.block_size #
            end_block = (end + self.block_size - 1) // self.block_size # 左闭右开区间，end_block本身不包含，最后访问的是end_block-1
            for i in range(start_block, end_block):
                slot_start = seq.block_table[i] * self.block_size
                if i == start_block:
                    slot_start += start % self.block_size
                if i != end_block - 1:
                    slot_end = seq.block_table[i] * self.block_size + self.block_size
                else:
                    slot_end = seq.block_table[i] * self.block_size + end - i * self.block_size
                slot_mapping.extend(range(slot_start, slot_end)) # 缓存写入位置列表：本轮每个 token 算出的 K、V，应该写到 KV cache 的哪个物理 slot。
        if cu_seqlens_k[-1] > cu_seqlens_q[-1]:    # prefix cache
            block_tables = self.prepare_block_tables(seqs) #每条序列的 KV 对应的物理block位置
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_q = torch.tensor(cu_seqlens_q, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_k = torch.tensor(cu_seqlens_k, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        set_context(True, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, slot_mapping, None, block_tables)
        return input_ids, positions

    def prepare_decode(self, seqs: list[Sequence]):
        input_ids = []
        positions = []
        slot_mapping = []
        context_lens = []
        for seq in seqs:
            input_ids.append(seq.last_token)
            positions.append(len(seq) - 1)
            context_lens.append(len(seq))
            slot_mapping.append(seq.block_table[-1] * self.block_size + seq.last_block_num_tokens  - 1)
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        context_lens = torch.tensor(context_lens, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        block_tables = self.prepare_block_tables(seqs)
        set_context(False, slot_mapping=slot_mapping, context_lens=context_lens, block_tables=block_tables)
        return input_ids, positions

    def prepare_sample(self, seqs: list[Sequence]):
        temperatures = [seq.temperature for seq in seqs]
        temperatures = torch.tensor(temperatures, dtype=torch.float32, pin_memory=True).cuda(non_blocking=True)
        return temperatures

    @torch.inference_mode()
    def run_model(self, input_ids: torch.Tensor, positions: torch.Tensor, is_prefill: bool):
        # enforce_eager 是否配置了强制普通执行模式；为 True 就不使用 CUDA Graph
        # input_ids 是一维张量，把本批各条 seq 本轮需要计算的 token 按顺序拼接起来。
        if is_prefill or self.enforce_eager or input_ids.size(0) > 512:
            return self.model.compute_logits(self.model(input_ids, positions))
        else:
            bs = input_ids.size(0)
            context = get_context()
            graph = self.graphs[next(x for x in self.graph_bs if x >= bs)]
            graph_vars = self.graph_vars
            graph_vars["input_ids"][:bs] = input_ids
            graph_vars["positions"][:bs] = positions
            graph_vars["slot_mapping"].fill_(-1)
            graph_vars["slot_mapping"][:bs] = context.slot_mapping
            graph_vars["context_lens"].zero_()
            graph_vars["context_lens"][:bs] = context.context_lens
            graph_vars["block_tables"][:bs, :context.block_tables.size(1)] = context.block_tables
            graph.replay()
            return self.model.compute_logits(graph_vars["outputs"][:bs])

    def run(self, seqs: list[Sequence], is_prefill: bool) -> list[int]:
        input_ids, positions = self.prepare_prefill(seqs) if is_prefill else self.prepare_decode(seqs)
        temperatures = self.prepare_sample(seqs) if self.rank == 0 else None
        logits = self.run_model(input_ids, positions, is_prefill)
        token_ids = self.sampler(logits, temperatures).tolist() if self.rank == 0 else None
        reset_context()
        return token_ids

    @torch.inference_mode()
    def capture_cudagraph(self):
        config = self.config
        hf_config = config.hf_config
        max_bs = min(self.config.max_num_seqs, 512)
        max_num_blocks = (config.max_model_len + self.block_size - 1) // self.block_size
        input_ids = torch.zeros(max_bs, dtype=torch.int64)
        positions = torch.zeros(max_bs, dtype=torch.int64)
        slot_mapping = torch.zeros(max_bs, dtype=torch.int32)
        context_lens = torch.zeros(max_bs, dtype=torch.int32)
        block_tables = torch.zeros(max_bs, max_num_blocks, dtype=torch.int32)
        outputs = torch.zeros(max_bs, hf_config.hidden_size)
        self.graph_bs = [1, 2, 4, 8] + list(range(16, max_bs + 1, 16))
        self.graphs = {}
        self.graph_pool = None

        for bs in reversed(self.graph_bs):
            graph = torch.cuda.CUDAGraph()
            set_context(False, slot_mapping=slot_mapping[:bs], context_lens=context_lens[:bs], block_tables=block_tables[:bs])
            outputs[:bs] = self.model(input_ids[:bs], positions[:bs])    # warmup
            with torch.cuda.graph(graph, self.graph_pool):
                outputs[:bs] = self.model(input_ids[:bs], positions[:bs])    # capture
            if self.graph_pool is None:
                self.graph_pool = graph.pool()
            self.graphs[bs] = graph
            torch.cuda.synchronize()
            reset_context()

        self.graph_vars = dict(
            input_ids=input_ids,
            positions=positions,
            slot_mapping=slot_mapping,
            context_lens=context_lens,
            block_tables=block_tables,
            outputs=outputs,
        )
