import torch
from torch import nn
import torch.distributed as dist
from transformers import Qwen3Config

from nanovllm.layers.activation import SiluAndMul
from nanovllm.layers.attention import Attention
from nanovllm.layers.layernorm import RMSNorm
from nanovllm.layers.linear import QKVParallelLinear, MergedColumnParallelLinear, RowParallelLinear
from nanovllm.layers.rotary_embedding import get_rope
from nanovllm.layers.embed_head import VocabParallelEmbedding, ParallelLMHead


class Qwen3Attention(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        max_position: int = 4096 * 32,
        head_dim: int | None = None,
        rms_norm_eps: float = 1e-06,
        qkv_bias: bool = False,
        rope_theta: float = 10000,
        rope_scaling: dict | None = None,
    ) -> None:
        # 初始化父类 nn.Module，使 PyTorch 能登记和管理下面创建的子模块及参数。
        super().__init__()
        # 张量并行的进程数；本项目每个进程绑定一张 GPU，例如使用两张卡时为 2。
        tp_size = dist.get_world_size()
        # 保存整个注意力层的 Q 头总数，尚未按 GPU 切分。
        self.total_num_heads = num_heads
        # % 是取余：Q 头总数必须能被进程数整除，才能平均分配，否则抛出断言错误。
        assert self.total_num_heads % tp_size == 0
        # 本进程负责的 Q 头数；例如总共 32 个头、两张卡，每张卡计算 16 个头。
        self.num_heads = self.total_num_heads // tp_size
        # 保存整个注意力层的 K 头总数，也是 V 头总数；不是 K 和 V 的头数之和。
        self.total_num_kv_heads = num_kv_heads
        # 当前实现也要求 KV 头数能被进程数整除，不会自动在多卡之间复制不足的 KV 头。
        assert self.total_num_kv_heads % tp_size == 0
        # 本进程负责的 K、V 各自的头数；例如总共 8 个 KV 头、两张卡，每张卡各负责 4 个 K 头和 4 个 V 头。
        self.num_kv_heads = self.total_num_kv_heads // tp_size
        # 每个头的向量维度；使用显式配置的非零 head_dim，否则用模型隐藏维度除以全局 Q 头数推算。
        self.head_dim = head_dim or hidden_size // self.total_num_heads
        # 本卡所有 Q 头拼接后的宽度；例如 16 个头 × 每头 128 维 = 2048，后续用于从合并结果中切出 Q。
        self.q_size = self.num_heads * self.head_dim
        # 本卡所有 K 头拼接后的宽度，V 也有相同宽度；后续按 [q_size, kv_size, kv_size] 拆分 Q、K、V。
        self.kv_size = self.num_kv_heads * self.head_dim
        # 注意力分数的缩放系数 1 / sqrt(head_dim)，乘在 Q 与 K 的点积上，避免头维度增大时分数尺度过大。
        self.scaling = self.head_dim ** -0.5
        # 保存是否启用 QKV 投影偏置的配置；本实现还用它决定是否创建、执行 Q/K 的 RMSNorm。
        self.qkv_bias = qkv_bias

        # 创建合并的 QKV 投影：一次线性计算得到本卡的 Q、K、V，输出最后一维为 q_size + 2 * kv_size。
        self.qkv_proj = QKVParallelLinear(
            # 输入：每个 token 的完整隐藏向量维度。
            hidden_size,
            # 每个 Q、K、V 头的向量维度。
            self.head_dim,
            # 全局 Q 头数，投影层内部据此切分本进程的权重。
            self.total_num_heads,
            # 全局 K、V 各自的头数，允许与 Q 头数不同，以支持分组查询注意力。
            self.total_num_kv_heads,
            # True 时投影结果还会加上训练得到的偏置向量；False 时只做矩阵乘法。
            bias=qkv_bias,
        )
        # 创建注意力输出投影：各卡对自己负责的头计算部分贡献，再通过 all_reduce 求和得到完整输出。
        self.o_proj = RowParallelLinear(
            # 所有 Q 头的输出拼接后的全局宽度；该层内部会按进程数切分输入维度，每卡只接收自己的部分。
            self.total_num_heads * self.head_dim,
            # 输出恢复为每个 token 的 hidden_size 维，便于后续残差相加；不一定比输入维度小。
            hidden_size,
            # 本项目的注意力输出投影不使用偏置。
            bias=False,
        )
        # 只有 rope_scaling 是字典时才读取它；None 表示没有提供这份可选配置。
        if isinstance(rope_scaling, dict):
            # 若字典指定了 rope_theta，就覆盖原频率基数，否则保留原值；当前代码没有处理 factor 等其他缩放字段。
            rope_theta = rope_scaling.get("rope_theta", rope_theta)
        # 获取 RoPE 位置编码模块，预先准备不同位置的正弦、余弦缓存，前向时用于旋转 Q、K。
        self.rotary_emb = get_rope(
            # 单个注意力头的维度。
            self.head_dim,
            # 参与旋转的位置编码维度；这里等于整个头的维度，即头内全部分量都参与旋转。
            rotary_dim=self.head_dim,
            # 预先缓存的位置数量，对应合法位置下标 0 到 max_position - 1。
            max_position=max_position,
            # 频率基数，决定不同维度对的旋转频率；它本身不是旋转角度。
            base=rope_theta,
        )
        # 创建本卡的注意力计算模块，负责使用 Q、K、V 及 KV Cache 执行 prefill/decode 注意力计算。
        self.attn = Attention(
            # 本卡的 Q 头数，而非全局头数。
            self.num_heads,
            # 每个头的向量维度。
            self.head_dim,
            # QK 点积在 softmax 前使用的缩放系数。
            self.scaling,
            # 本卡的 K、V 各自的头数；可少于 Q 头数，此时多个 Q 头共享一组 K、V。
            self.num_kv_heads,
        )
        # 本实现仅在不使用 QKV 偏置时启用 Q/K 归一化；这是此处的模型实现约定，不是所有模型的通用规则。
        if not self.qkv_bias:
            # 对每个 token 的每个 Q 头独立做 RMSNorm，沿该头的 head_dim 维计算；eps 防止分母为零或过小。
            self.q_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)
            # 对 K 头做同样的归一化，使用独立于 Q 的缩放参数；这里没有给 V 创建归一化层。
            self.k_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)

    # 执行当前层的注意力计算；下面用 T 表示本轮输入的 token 总数（可来自多条序列）。
    def forward(
        self,
        # 每个 token 在所属序列中的位置，形状为 [T]，用于旋转位置编码。
        positions: torch.Tensor,
        # 输入的隐藏向量，形状为 [T, hidden_size]；这里存的是浮点特征，不是 token ID。
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        # 一次线性投影生成本卡负责的 Q、K、V，输出形状为 [T, q_size + 2 * kv_size]。
        qkv = self.qkv_proj(hidden_states)
        # 沿最后一维按指定长度拆出 Q、K、V；返回共享原存储的视图，不复制整份数据。
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        # 将 Q 的特征维拆成“本卡 Q 头数 × 每头维度”，形状变为 [T, num_heads, head_dim]；-1 自动推算为 T。
        q = q.view(-1, self.num_heads, self.head_dim)
        # K 使用本卡 KV 头数，形状变为 [T, num_kv_heads, head_dim]；KV 头数可以少于 Q 头数。
        k = k.view(-1, self.num_kv_heads, self.head_dim)
        # V 和 K 的头数、每头维度相同，形状也为 [T, num_kv_heads, head_dim]。
        v = v.view(-1, self.num_kv_heads, self.head_dim)
        # 本实现约定：不使用 QKV 偏置时，额外对 Q 和 K 做 RMSNorm。
        if not self.qkv_bias:
            # 对每个 Q 头的 head_dim 个元素做归一化，张量形状不变。
            q = self.q_norm(q)
            # 对每个 K 头做同样的归一化，使用独立参数；V 不做这里的归一化。
            k = self.k_norm(k)
        # 根据 token 位置对 Q、K 应用 RoPE 旋转位置编码，形状不变；V 不参与旋转。
        q, k = self.rotary_emb(positions, q, k)
        # 计算因果注意力：Q 与 K 匹配，经缩放和 softmax 后加权 V；内部还负责写入和使用 KV cache。
        # prefill 输出为 [T, num_heads, head_dim]；decode 输出为 [T, 1, num_heads, head_dim]。
        o = self.attn(q, k, v)
        # flatten 合并第 1 维到最后一维，得到 [T, q_size]，相当于拼接本卡各 Q 头的输出。
        # o_proj 将结果映射回 hidden_size；多卡时内部通过 all_reduce 求和，得到完整的投影结果。
        output = self.o_proj(o.flatten(1, -1))
        # 返回 [T, hidden_size] 的注意力模块输出；残差连接由外层 DecoderLayer 处理。
        return output


class Qwen3MLP(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
    ) -> None:
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size,
            [intermediate_size] * 2,
            bias=False,
        )
        self.down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=False,
        )
        assert hidden_act == "silu"
        self.act_fn = SiluAndMul()

    def forward(self, x):
        gate_up = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        x = self.down_proj(x)
        return x


class Qwen3DecoderLayer(nn.Module):

    def __init__(
        self,
        config: Qwen3Config,
    ) -> None:
        super().__init__()
        self.self_attn = Qwen3Attention(
            # 每个 token 输入隐藏向量的维度，也是注意力模块最终输出的维度，例如 4096。
            hidden_size=config.hidden_size,
            # 整个注意力层的 Q（查询）头总数；张量并行时再平均分给各进程。
            num_heads=config.num_attention_heads,
            # K（键）和 V（值）各自的头总数；例如 32 个 Q 头、8 个 KV 头时，每 4 个 Q 头共享一组 K、V。
            num_kv_heads=config.num_key_value_heads,
            # 用于预先生成 RoPE 旋转位置编码缓存的位置数量；例如 32768 对应位置下标 0～32767。
            max_position=config.max_position_embeddings,
            # Q、K 做 RMSNorm 时加在均方值上的小正数，避免分母为零或过小；本实现仅在 qkv_bias=False 时创建 Q、K 归一化层。
            rms_norm_eps=config.rms_norm_eps,
            # Q、K、V 的线性投影是否带偏置项；getattr 的第三个参数 True 表示配置没有 attention_bias 属性时的默认值。
            qkv_bias=getattr(config, 'attention_bias', True),
            # 每个注意力头的向量维度；未配置时传 None，由 Qwen3Attention 按 hidden_size // num_heads 推算。
            head_dim=getattr(config, 'head_dim', None),
            # RoPE 的频率基数，用于计算不同维度随位置变化的旋转角度；缺少此配置时默认使用 1000000。
            rope_theta=getattr(config, "rope_theta", 1000000),
            # 可选的 RoPE 扩展配置，缺少时传 None；当前实现只读取字典中的 rope_theta 来覆盖频率基数，并未实现通用的位置缩放策略。
            rope_scaling=getattr(config, "rope_scaling", None),
        )
        self.mlp = Qwen3MLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
        )
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    # 执行一层 Decoder：归一化、自注意力、残差相加与归一化、前馈网络；隐藏向量的最终形状不变。
    def forward(
        self,
        # 本轮各 token 在所属序列中的位置，形状为 [token 总数]，用于给 Q、K 添加 RoPE 位置信息。
        positions: torch.Tensor,
        # 形状为 [token 总数, hidden_size]；第一层接收 Embedding 输出，后续层接收上一层尚未加回残差的 MLP 输出。
        hidden_states: torch.Tensor,
        # 残差分支保存待相加的隐藏向量，形状与 hidden_states 相同；第一层开始时为 None。
        residual: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # 第一层没有已有残差，需要先保留原始输入。
        if residual is None:
            # 右侧先求值再赋值：residual 保存归一化前的输入，hidden_states 接收 RMSNorm 的输出。
            hidden_states, residual = self.input_layernorm(hidden_states), hidden_states
        else:
            # 本项目的 RMSNorm 支持先相加再归一化：residual 更新为上一层 MLP 输出 + 旧残差，hidden_states 为该和的归一化结果。
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        # 由归一化后的隐藏向量生成 Q、K、V，应用位置编码并计算自注意力，再投影回 hidden_size 维。
        hidden_states = self.self_attn(positions, hidden_states)
        # 将注意力输出加回残差；返回归一化后的和（hidden_states）以及归一化前的和（residual）。
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        # 前馈网络执行两路投影、SiLU 门控相乘和降维投影，输出仍为 [token 总数, hidden_size]。
        hidden_states = self.mlp(hidden_states)
        # 分别返回 MLP 输出和残差，此处尚未把两者相加；相加会在下一层开头，或最后一层之后的模型最终 RMSNorm 中完成。
        return hidden_states, residual


class Qwen3Model(nn.Module):

    def __init__(
        self,
        config: Qwen3Config,
    ) -> None:
        super().__init__()
        # 输入嵌入层：按 token ID 查表得到 hidden_size 维向量；张量并行时按词表切分权重。
        self.embed_tokens = VocabParallelEmbedding(config.vocab_size, config.hidden_size)
        # 创建并注册指定数量的独立 Transformer 解码器层，前向时依次处理隐藏向量。
        self.layers = nn.ModuleList([Qwen3DecoderLayer(config) for _ in range(config.num_hidden_layers)])
        # 最终归一化层：所有解码器层之后沿隐藏维度做 RMSNorm，保持形状不变；eps 用于数值稳定。
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)  

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        # 根据 token ID 查找嵌入向量：[T] → [T, hidden_size]，T 为本轮输入的 token 总数。
        # 多卡时，嵌入层内部会将各卡的局部结果求和，让每张卡获得完整的隐藏向量。
        hidden_states = self.embed_tokens(input_ids)
        # 第一层还没有单独保存的残差；传入 None，让第一层自行保留归一化前的输入。
        residual = None
        # 按顺序执行所有 Decoder 层；各层结构相同，但拥有各自的模型参数。
        for layer in self.layers:
            # 传入位置、隐藏向量和残差；每层完成注意力与 MLP 计算，返回 MLP 输出和待相加的残差。
            # 两个返回张量的形状均为 [T, hidden_size]；MLP 输出与残差的相加留到下一层或最终归一化时完成。
            hidden_states, residual = layer(positions, hidden_states, residual)
        # 将最后一层的 MLP 输出与残差相加，再做最终 RMSNorm；形状仍为 [T, hidden_size]。
        # 第二个返回值是归一化前的和，后续不再需要，用 _ 接收以表示忽略。
        hidden_states, _ = self.norm(hidden_states, residual)
        # 返回最终隐藏向量；此处还不是词表 logits，后续由外层模型的 lm_head 完成投影。
        return hidden_states


class Qwen3ForCausalLM(nn.Module):
    packed_modules_mapping = {
        "q_proj": ("qkv_proj", "q"),
        "k_proj": ("qkv_proj", "k"),
        "v_proj": ("qkv_proj", "v"),
        "gate_proj": ("gate_up_proj", 0),
        "up_proj": ("gate_up_proj", 1),
    }

    def __init__(
        self,
        config: Qwen3Config
    ) -> None:
        super().__init__()
        #里面包含输入 Embedding、多层 Transformer 和最终归一化，负责：token ID → 输入向量 → Transformer 处理 → 隐藏向量
        self.model = Qwen3Model(config)
        self.lm_head = ParallelLMHead(config.vocab_size, config.hidden_size)
        # embed_tokens.weight 存储tokenid对应的向量
        if config.tie_word_embeddings:
            self.lm_head.weight.data = self.model.embed_tokens.weight.data

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        return self.model(input_ids, positions)

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        return self.lm_head(hidden_states)
