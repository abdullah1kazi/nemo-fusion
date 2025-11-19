from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import math


@dataclass
class ModelConfig:
    num_params: int
    num_layers: int
    hidden_size: int
    num_attention_heads: int
    sequence_length: int = 2048
    vocab_size: int = 50257

    def __post_init__(self):
        assert self.num_params > 0 and self.num_layers > 0 and self.hidden_size > 0


@dataclass
class HardwareConfig:
    num_gpus: int
    gpu_memory_gb: float
    interconnect: str = "NVLink"
    gpu_type: str = "H100"

    def __post_init__(self):
        assert self.num_gpus > 0 and self.gpu_memory_gb > 0


@dataclass
class ParallelStrategy:
    tensor_parallel: int
    pipeline_parallel: int
    data_parallel: int
    context_parallel: int = 1
    memory_per_gpu_gb: float = 0.0
    expected_throughput: float = 0.0
    efficiency_score: float = 0.0

    def __str__(self) -> str:
        return (
            f"TP={self.tensor_parallel} PP={self.pipeline_parallel} "
            f"DP={self.data_parallel} CP={self.context_parallel} | "
            f"Mem={self.memory_per_gpu_gb:.1f}GB "
            f"Throughput={self.expected_throughput:.0f}tok/s "
            f"Eff={self.efficiency_score:.2%}"
        )

    def to_nemo_config(self) -> Dict[str, int]:
        return {
            "tensor_model_parallel_size": self.tensor_parallel,
            "pipeline_model_parallel_size": self.pipeline_parallel,
            "data_parallel_size": self.data_parallel,
            "context_parallel_size": self.context_parallel,
        }


class AutoParallelOptimizer:
    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.strategies: List[ParallelStrategy] = []

    def optimize(
        self,
        num_params: int,
        num_layers: int,
        hidden_size: int,
        num_gpus: int,
        gpu_memory_gb: float,
        batch_size: int = 1,
        sequence_length: int = 2048,
        num_attention_heads: Optional[int] = None,
        use_context_parallel: bool = True,
    ) -> ParallelStrategy:
        if num_attention_heads is None:
            num_attention_heads = hidden_size // 128

        model_memory_gb = self._calc_model_mem(num_params, num_layers, hidden_size)
        activation_memory_gb = self._calc_activation_mem(
            batch_size, sequence_length, hidden_size, num_layers
        )

        if self.verbose:
            print(f"Model: {model_memory_gb:.2f}GB, Activation: {activation_memory_gb:.2f}GB")

        candidates = []
        for tp in self._get_valid_tp_sizes(num_gpus, num_attention_heads):
            for pp in self._get_valid_pp_sizes(num_gpus, num_layers, tp):
                for cp in ([1, 2, 4, 8] if use_context_parallel else [1]):
                    if tp * pp * cp > num_gpus:
                        continue

                    dp = num_gpus // (tp * pp * cp)

                    memory_per_gpu = self._calc_memory(
                        model_memory_gb,
                        activation_memory_gb,
                        tp, pp, dp, cp,
                        batch_size, sequence_length
                    )

                    if memory_per_gpu <= gpu_memory_gb * 0.9:
                        throughput = self._calc_throughput(
                            tp, pp, dp, cp, num_layers, hidden_size, num_gpus
                        )
                        efficiency = self._calc_efficiency(
                            tp, pp, dp, cp, memory_per_gpu, gpu_memory_gb
                        )
                        candidates.append(ParallelStrategy(
                            tensor_parallel=tp,
                            pipeline_parallel=pp,
                            data_parallel=dp,
                            context_parallel=cp,
                            memory_per_gpu_gb=memory_per_gpu,
                            expected_throughput=throughput,
                            efficiency_score=efficiency,
                        ))

        if not candidates:
            raise ValueError(
                f"No valid parallelism strategy found! "
                f"Model requires {model_memory_gb:.1f} GB but only "
                f"{gpu_memory_gb:.1f} GB available per GPU."
            )

        self.strategies = sorted(candidates, key=lambda s: s.efficiency_score, reverse=True)
        return self.strategies[0]

    def _get_valid_tp_sizes(self, num_gpus: int, num_attention_heads: int) -> List[int]:
        return [tp for tp in [1, 2, 4, 8, 16]
                if tp <= num_gpus and num_attention_heads % tp == 0]

    def _get_valid_pp_sizes(self, num_gpus: int, num_layers: int, tp: int) -> List[int]:
        return [pp for pp in [1, 2, 4, 8, 16]
                if pp * tp <= num_gpus and num_layers % pp == 0]

    def _calc_model_mem(self, num_params: int, num_layers: int, hidden_size: int) -> float:
        param_memory = num_params * 2 / 1e9
        optimizer_memory = num_params * 8 / 1e9
        gradient_memory = num_params * 2 / 1e9
        return param_memory + optimizer_memory + gradient_memory

    def _calc_activation_mem(
        self, batch_size: int, seq_len: int, hidden_size: int, num_layers: int
    ) -> float:
        attention_memory = 4 * batch_size * seq_len * hidden_size * 2 / 1e9
        mlp_memory = batch_size * seq_len * hidden_size * 4 * 2 / 1e9
        return (attention_memory + mlp_memory) * num_layers

    def _calc_memory(
        self,
        model_memory: float,
        activation_memory: float,
        tp: int, pp: int, dp: int, cp: int,
        batch_size: int, sequence_length: int,
    ) -> float:
        model_per_gpu = model_memory / (tp * pp)
        micro_batch_size = max(1, batch_size // (dp * pp))
        activation_scale = micro_batch_size / batch_size
        sequence_scale = 1.0 / cp if cp > 1 else 1.0
        activation_per_gpu = (activation_memory / tp) * activation_scale * sequence_scale
        buffer_memory = (model_per_gpu + activation_per_gpu) * 0.2
        return model_per_gpu + activation_per_gpu + buffer_memory

    def _calc_throughput(
        self, tp: int, pp: int, dp: int, cp: int,
        num_layers: int, hidden_size: int, num_gpus: int
    ) -> float:
        base_throughput = 10000 / (hidden_size / 4096)
        tp_efficiency = 0.9 ** (tp - 1) if tp > 1 else 1.0
        num_microbatches = max(4, pp * 4)
        pp_efficiency = num_microbatches / (num_microbatches + pp - 1) if pp > 1 else 1.0
        dp_efficiency = 0.98 ** (dp - 1) if dp > 1 else 1.0
        cp_efficiency = 0.95 ** (cp - 1) if cp > 1 else 1.0
        total_efficiency = tp_efficiency * pp_efficiency * dp_efficiency * cp_efficiency
        return base_throughput * dp * total_efficiency

    def _calc_efficiency(
        self, tp: int, pp: int, dp: int, cp: int,
        memory_per_gpu: float, gpu_memory_gb: float
    ) -> float:
        memory_util = memory_per_gpu / gpu_memory_gb
        if memory_util < 0.5:
            memory_score = memory_util / 0.5
        elif memory_util < 0.85:
            memory_score = 1.0
        else:
            memory_score = (0.95 - memory_util) / 0.1
        memory_score = max(0.0, min(1.0, memory_score))

        comm_score = 0.95 ** (tp - 1) * 0.97 ** (pp - 1) * 0.98 ** (cp - 1)
        total_parallel = tp * pp * cp
        balance_score = min(dp, total_parallel) / max(dp, total_parallel)
        return 0.4 * memory_score + 0.4 * comm_score + 0.2 * balance_score

    def get_top_strategies(self, n: int = 5) -> List[ParallelStrategy]:
        return self.strategies[:n]
