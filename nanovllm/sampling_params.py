from dataclasses import dataclass


@dataclass(slots=True)
class SamplingParams:
    # 采样温度：先用 logits 除以该值，再计算概率；1.0 不改变分布，小于 1 更偏向高分 token，大于 1 更随机。
    # 本实现要求温度大于 1e-10，不支持设为 0 来进行每次选择最高分 token 的贪心生成。
    temperature: float = 1.0
    # 本次请求最多新生成的 token 数，不包含输入提示词；默认 64，达到上限就停止，token 数不等于汉字数或单词数。
    max_tokens: int = 64
    # 是否忽略 EOS（结束 token）：默认 False，生成 EOS 时提前停止；True 时遇到 EOS 仍继续，但仍受 max_tokens 限制。
    ignore_eos: bool = False

    def __post_init__(self):
        assert self.temperature > 1e-10, "greedy sampling is not permitted"
