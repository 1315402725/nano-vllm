from functools import lru_cache
import torch
from torch import nn


def apply_rotary_emb(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    # x 是 Q 或 K，形状为 [T, H, D]：T 为 token 数，H 为本卡对应的头数，D 为每头维度。
    # 先转为 float32 进行旋转计算，再沿最后一维平分成前后两半，各为 [T, H, D/2]。
    # 两半的对应元素配成一对，例如 D=4 时，[a, b, c, d] 拆成 [a, b] 和 [c, d]，配对为 (a, c)、(b, d)。
    x1, x2 = torch.chunk(x.float(), 2, dim=-1)
    # 二维旋转的第一个分量：y1 = x1*cos(角度) - x2*sin(角度)，这里都是逐元素乘法。
    # cos、sin 的形状为 [T, 1, D/2]，会沿头数维广播到 H 个头；y1 的形状为 [T, H, D/2]。
    y1 = x1 * cos - x2 * sin
    # 二维旋转的第二个分量：y2 = x2*cos(角度) + x1*sin(角度)，形状同样为 [T, H, D/2]。
    y2 = x2 * cos + x1 * sin
    # 将旋转后的前后两半按 y1 在前、y2 在后的顺序拼回 [T, H, D]，再转回输入 x 原本的数据类型。
    return torch.cat((y1, y2), dim=-1).to(x.dtype)


class RotaryEmbedding(nn.Module):

    def __init__(
        self,
        head_size: int,
        rotary_dim: int,
        max_position_embeddings: int,
        base: float,
    ) -> None:
        # 初始化 nn.Module，使当前模块能够注册 buffer 等，并随模型一起迁移设备。
        super().__init__()
        # 保存每个注意力头的维度；下面用 D 表示 head_size，用 P 表示最大位置数。
        self.head_size = head_size
        # 此实现要求整个头的所有维度都参与旋转，不支持只旋转头内的一部分维度。
        assert rotary_dim == head_size
        # 为 D/2 对特征生成不同的旋转频率：第 i 对的频率为 base^(-2*i/D)，形状为 [D/2]。
        # 本文件的旋转函数将前半段与后半段的对应元素配成一对；D 需要是偶数。
        inv_freq = 1.0 / (base**(torch.arange(0, rotary_dim, 2, dtype=torch.float) / rotary_dim))
        # 生成位置编号 [0, 1, ..., P-1]，形状为 [P]；使用浮点数方便后续计算角度。
        t = torch.arange(max_position_embeddings, dtype=torch.float)
        # 计算外积：freqs[p, i] = t[p] * inv_freq[i]，得到各位置、各特征对的旋转角度，形状为 [P, D/2]。
        freqs = torch.einsum("i,j -> ij", t, inv_freq)
        # 预先计算各旋转角度的余弦，形状为 [P, D/2]。
        cos = freqs.cos()
        # 预先计算各旋转角度的正弦，形状为 [P, D/2]。
        sin = freqs.sin()
        # 先拼接为 [P, D]，再在第 1 维插入大小为 1 的维度，得到 [P, 1, D]。
        # 中间的 1 用于后续广播到所有注意力头；unsqueeze_ 的下划线表示原地改变张量形状。
        cache = torch.cat((cos, sin), dim=-1).unsqueeze_(1)
        # 注册为无需训练的缓存：随模块的 .to(...) 迁移设备/转换类型；persistent=False 表示不写入 state_dict。
        self.register_buffer("cos_sin_cache", cache, persistent=False)

    # 使用 PyTorch 编译机制优化此方法的执行；具体编译和优化在调用时进行。
    @torch.compile
    def forward(
        self,
        # 本轮每个 token 在所属序列中的位置，形状为 [T]；T 表示本轮 token 总数。
        positions: torch.Tensor,
        # Q 向量，形状为 [T, 本卡 Q 头数, D]。
        query: torch.Tensor,
        # K 向量，形状为 [T, 本卡 KV 头数, D]；KV 头数可以和 Q 头数不同。
        key: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # 按 token 的实际位置查找预计算结果，得到 [T, 1, D]；不是直接取前 T 个位置。
        cos_sin = self.cos_sin_cache[positions]
        # 沿最后一维平分为余弦和正弦两部分，各自形状为 [T, 1, D/2]。
        cos, sin = cos_sin.chunk(2, dim=-1)
        # 对 Q 的每对特征做二维旋转；cos/sin 沿头数维广播，输出形状仍为 [T, 本卡 Q 头数, D]。
        query = apply_rotary_emb(query, cos, sin)
        # 对 K 应用相同位置、相同频率的旋转，输出形状仍为 [T, 本卡 KV 头数, D]。
        key = apply_rotary_emb(key, cos, sin)
        # 返回加入旋转位置信息后的 Q、K，供后续注意力计算使用；此方法不处理 V。
        return query, key


@lru_cache(1)
def get_rope(
    head_size: int,
    rotary_dim: int,
    max_position: int,
    base: float,
):
    rotary_emb = RotaryEmbedding(head_size, rotary_dim, max_position, base)
    return rotary_emb
