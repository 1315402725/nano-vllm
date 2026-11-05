import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist

from nanovllm.utils.context import get_context


class VocabParallelEmbedding(nn.Module):

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
    ):
        super().__init__()
        # 当前进程的编号，从 0 开始；本项目每个进程对应一张 GPU，用编号确定自己负责哪部分词表。
        self.tp_rank = dist.get_rank()
        # 默认通信组中的进程总数；例如两张 GPU 各运行一个进程，这里就是 2。
        self.tp_size = dist.get_world_size()
        # 词表大小必须能被进程数整除，才能平均切分；% 是取余，不满足时抛出断言错误。
        assert num_embeddings % self.tp_size == 0
        # 保存完整词表大小，即一共有多少个 token ID；例如 10000。
        self.num_embeddings = num_embeddings
        # 每个进程保存的词表行数；// 是整除，例如 10000 // 2 = 5000。
        self.num_embeddings_per_partition = self.num_embeddings // self.tp_size
        # 当前分片的起始 token ID（包含）；例如 rank 0 从 0 开始，rank 1 从 5000 开始。
        self.vocab_start_idx = self.num_embeddings_per_partition * self.tp_rank
        # 当前分片的结束边界（不包含）；例如 rank 1 负责 [5000, 10000)，即 5000～9999。
        self.vocab_end_idx = self.vocab_start_idx + self.num_embeddings_per_partition
        # 创建本进程的权重矩阵：[分片词表大小, 向量维度]，每行存一个 token 的完整向量；Parameter 将其注册为模型参数，empty 只分配空间，真实权重稍后加载。
        self.weight = nn.Parameter(torch.empty(self.num_embeddings_per_partition, embedding_dim))
        # 给权重绑定下方的加载方法，供项目加载器调用；这里只保存方法，不执行它，加载时才从完整权重中取出当前进程负责的行。
        self.weight.weight_loader = self.weight_loader

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        # 取得本进程已分配的权重张量，形状为 [分片词表大小, 隐藏维度]；这里没有复制，修改它会更新 param 的数据。
        param_data = param.data
        # size(0) 读取第 0 维（行数），即本进程需要加载多少个 token 的向量；例如 10000 个 token 分给两个进程，每份 5000 行。
        shard_size = param_data.size(0)
        # 根据进程编号计算在完整权重中的起始行；例如 rank 0 从第 0 行开始，rank 1 从第 5000 行开始。
        start_idx = self.tp_rank * shard_size
        # narrow(维度, 起点, 长度)：沿第 0 维取出当前进程负责的行，保留所有列；例如 rank 1 取 [5000, 10000) 行，得到 [5000, 隐藏维度]。
        loaded_weight = loaded_weight.narrow(0, start_idx, shard_size)
        # 将选出的训练权重复制到本进程的参数存储中；copy_ 末尾的下划线表示原地修改，若源在 CPU、目标在 GPU，也会完成设备间复制。
        param_data.copy_(loaded_weight)

    # 输入 x 是本批次待处理的 token ID，形状为 [token 总数]；输出每个 token 对应的隐藏向量。
    def forward(self, x: torch.Tensor):
        # 多进程张量并行时，每个进程只保存部分词表，需要先筛选自己负责的 token；单进程时直接查完整词表。
        if self.tp_size > 1:
            # 判断每个 ID 是否在本进程的词表范围内，得到布尔掩码；例如范围 [5000, 10000)、输入 [100, 6000]，结果为 [False, True]。
            mask = (x >= self.vocab_start_idx) & (x < self.vocab_end_idx)
            # 全局 ID 减去起始 ID，转换成本地权重的行号；不属于自己的 ID 暂时置为 0，避免查表越界，例如上例变为 [0, 1000]。
            x = mask * (x - self.vocab_start_idx)
        # 按 x 中的行号从权重矩阵取出向量，得到 y，形状为 [token 总数, 隐藏维度]；多进程时不属于自己的位置此时暂用了本地第 0 行。
        y = F.embedding(x, self.weight)
        # 多进程时还需清除占位向量，并汇总各进程的查表结果；单进程的结果已完整，无需汇总。
        if self.tp_size > 1:
            # 将 mask 从 [token 总数] 变为 [token 总数, 1]，通过广播作用于每行所有维度；属于自己的向量保留，其余整行清零。
            y = mask.unsqueeze(1) * y
            # 默认按元素求和，并将结果原地写回每个进程的 y；每个有效 ID 仅由一个进程提供向量，其他进程提供零，因此求和后各进程都得到完整结果。
            dist.all_reduce(y)
        # 返回完整的 token 向量，形状为 [token 总数, 隐藏维度]，供后续 Transformer 层使用。
        return y


# 模型输出层：把最终隐藏向量投影成词表中各 token 的分数（logits），供后续采样。
# 继承父类以复用按词表切分的权重、并行信息和权重加载逻辑；这里重写 forward，执行线性投影。
class ParallelLMHead(VocabParallelEmbedding):

    def __init__(
        self,
        num_embeddings: int,  # 完整词表大小；张量并行时每个进程只保存其中一部分词表的权重。
        embedding_dim: int,  # 输入隐藏向量的维度，即模型的 hidden_size。
        bias: bool = False,  # 是否带偏置项；当前实现仅支持 False。
    ):
        assert not bias  # 不支持偏置，传入 True 时触发断言错误。
        super().__init__(num_embeddings, embedding_dim)  # 初始化并行信息，以及形状为 [本进程词表大小, hidden_size] 的权重。

    # compute_logits(hidden_states) 调用 self.lm_head(hidden_states) 时，由 nn.Module 的调用机制进入这里。
    # x 是模型所有 Transformer 层和最终归一化的输出；prefill/decode 都会调用此方法。
    def forward(self, x: torch.Tensor):
        context = get_context()  # 获取本轮上下文，包括是否 prefill、各序列本轮输入长度的前缀和。
        if context.is_prefill:  # prefill 的 x 包含本批所有新计算 token；每条序列只选本轮最后一个位置计算 logits。
            last_indices = context.cu_seqlens_q[1:] - 1  # 各段结束边界减 1，得到末尾下标；如 [0, 3, 5] 得到 [2, 4]。
            x = x[last_indices].contiguous()  # x[[2, 4]]  指定行数取出每个序列最后一个维度的隐藏向量，选取这些隐藏向量并保证内存连续，形状变为 [序列数, hidden_size]。
        # decode 时每条序列本来就只输入一个 token，因此无需上述筛选。
        # 分块 prefill 未完成的序列也会算出 logits，但其采样结果会被 Scheduler.postprocess 丢弃。
        logits = F.linear(x, self.weight)  # 计算 x @ weight.T，无偏置；输出形状为 [序列数, 本进程词表大小]。
        if self.tp_size > 1:  # 多进程张量并行时，各进程只算了各自词表分片的分数，需要汇总。
            all_logits = [torch.empty_like(logits) for _ in range(self.tp_size)] if self.tp_rank == 0 else None  # 仅 rank 0 分配接收列表，每个进程对应一个同形状张量。
            dist.gather(logits, all_logits, 0)  # 所有进程参与，把各自 logits 收集到 rank 0；第三个参数 0 是目标 rank。
            logits = torch.cat(all_logits, -1) if self.tp_rank == 0 else None  # rank 0 沿最后一维（词表维）按 rank 顺序拼接，恢复完整词表分数；其他进程置为 None。
        return logits  # 单进程或 rank 0 返回 [序列数, 完整词表大小] 的未归一化分数；其他 rank 返回 None。
