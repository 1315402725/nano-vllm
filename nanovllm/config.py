import os
from dataclasses import dataclass
from transformers import AutoConfig


@dataclass(slots=True)
class Config:
    # 本地模型目录路径，目录中存放模型配置、分词器和训练好的权重；这是必填参数。
    model: str
    # 每轮 prefill 最多新计算的 token 总数，所有序列合计；不是单条序列长度，超出本轮预算的输入可分块处理。
    max_num_batched_tokens: int = 16384
    # 每轮调度最多处理多少条序列（请求），用于限制批次大小；不是服务累计可接收的请求总数。
    max_num_seqs: int = 512
    # 单条序列的长度预算，单位为 token，包含输入和生成内容；初始化时不超过模型的位置上限，用于预热和 CUDA Graph 缓冲区规划，当前调度器并未据此强制截断生成。
    max_model_len: int = 4096
    # 计算 KV Cache 容量时的显存预算比例；0.9 表示按 GPU 总显存的 90% 扣除已有占用及预估计算开销，不是算力利用率，也不是显存硬上限。
    gpu_memory_utilization: float = 0.9
    # 张量并行的进程数，本项目每个进程绑定一张 GPU；1 表示单卡，大于 1 时将模型中的部分权重和计算切分到多卡，当前允许 1～8。
    tensor_parallel_size: int = 1
    # 是否强制即时执行：True 时不捕获或回放 CUDA Graph；False 时允许在适用的 decode 批次中使用 CUDA Graph，减少执行开销。
    enforce_eager: bool = False
    # 从模型目录加载的 Hugging Face 配置对象，包含隐藏维度、层数、注意力头数等；初始 None，在 __post_init__ 中自动读取并赋值。
    hf_config: AutoConfig | None = None
    # 结束 token 的 ID，用于判断生成是否结束；-1 是初始化占位值，引擎加载分词器后会替换为 eos_token_id，请求可通过 ignore_eos 忽略它。
    eos: int = -1
    # 每个 KV Cache 块可保存的 token 数量，不是字节数；用于分页管理历史 K、V，当前实现要求为 256 的倍数。
    kvcache_block_size: int = 256
    # KV Cache 池的块总数，-1 表示尚未计算；ModelRunner 根据显存预算和每块大小自动计算并覆盖，token 槽位总数为块数 × 每块 token 数。
    num_kvcache_blocks: int = -1

    def __post_init__(self):
        assert os.path.isdir(self.model)
        assert self.kvcache_block_size % 256 == 0
        assert 1 <= self.tensor_parallel_size <= 8
        self.hf_config = AutoConfig.from_pretrained(self.model)
        self.max_model_len = min(self.max_model_len, self.hf_config.max_position_embeddings)
