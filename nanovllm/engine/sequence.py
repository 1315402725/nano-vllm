from copy import copy
from enum import Enum, auto
from itertools import count

from nanovllm.sampling_params import SamplingParams


class SequenceStatus(Enum):
    WAITING = auto()
    RUNNING = auto()
    FINISHED = auto()


class Sequence:
    block_size = 256
    counter = count()

    def __init__(self, token_ids: list[int], sampling_params = SamplingParams()):
        self.seq_id = next(Sequence.counter)  # 当前进程内递增的请求编号，用于关联结果并恢复请求顺序
        self.status = SequenceStatus.WAITING  # 调度状态：等待、运行或结束；新请求先进入等待状态
        self.token_ids = copy(token_ids)  # 输入及后续生成的 token ID；浅拷贝避免追加时修改调用方列表
        self.last_token = token_ids[-1]  # 当前最后一个 token ID；decode 时作为本轮模型输入
        self.num_tokens = len(self.token_ids)  # 当前 token 总数，随生成结果追加而增加
        self.num_prompt_tokens = len(token_ids)  # 原始输入的 token 数，用于区分输入与生成内容，后续保持不变
        self.num_cached_tokens = 0  # 已有可用 K/V 的 token 数，包括复用的前缀缓存和本请求已完成的计算
        self.num_scheduled_tokens = 0  # 本轮计划计算的 token 数；prefill 可为多个，decode 为 1，处理后清零
        self.is_prefill = True  # 是否处于 prefill 阶段；序列化时据此传完整 token 列表或仅最后一个 token
        self.block_table = []  # 逻辑块到物理 KV 缓存块编号的映射；列表下标是逻辑块号，元素是物理块号
        self.temperature = sampling_params.temperature  # 采样温度，用于缩放 logits，调整输出概率分布的平坦程度
        self.max_tokens = sampling_params.max_tokens  # 最多生成的 token 数，不包含原始输入
        self.ignore_eos = sampling_params.ignore_eos  # 是否忽略 EOS 结束标记；为 True 时仍受 max_tokens 限制

    def __len__(self):
        return self.num_tokens

    def __getitem__(self, key):
        return self.token_ids[key]

    @property
    def is_finished(self):
        return self.status == SequenceStatus.FINISHED

    @property
    def num_completion_tokens(self):
        return self.num_tokens - self.num_prompt_tokens

    @property
    def prompt_token_ids(self):
        return self.token_ids[:self.num_prompt_tokens]

    @property
    def completion_token_ids(self):
        return self.token_ids[self.num_prompt_tokens:]

    @property
    def num_blocks(self):
        return (self.num_tokens + self.block_size - 1) // self.block_size

    @property
    def last_block_num_tokens(self):
        return self.num_tokens - (self.num_blocks - 1) * self.block_size

    def block(self, i):
        assert 0 <= i < self.num_blocks
        return self.token_ids[i*self.block_size: (i+1)*self.block_size]

    def append_token(self, token_id: int):
        self.token_ids.append(token_id)
        self.last_token = token_id
        self.num_tokens += 1

    def __getstate__(self):
        last_state = self.last_token if not self.is_prefill else self.token_ids
        return (self.num_tokens, self.num_prompt_tokens, self.num_cached_tokens, self.num_scheduled_tokens, self.block_table, last_state)

    def __setstate__(self, state):
        self.num_tokens, self.num_prompt_tokens, self.num_cached_tokens, self.num_scheduled_tokens, self.block_table, last_state = state
        if isinstance(last_state, list):
            self.token_ids = last_state
            self.last_token = self.token_ids[-1]
        else:
            self.token_ids = []
            self.last_token = last_state
