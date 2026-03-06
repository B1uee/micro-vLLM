from dataclasses import dataclass


@dataclass
class SamplingParams:
    # Sequence/Sampler 使用的最小逐请求采样参数。
    temperature: float = 1.0
    max_tokens: int = 64
    ignore_eos: bool = False

    def __post_init__(self):
        assert self.temperature > 1e-10, "greedy sampling is not permitted"
