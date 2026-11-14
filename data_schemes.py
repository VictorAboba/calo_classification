from pydantic import BaseModel, Field, field_validator
from typing import Generator, Tuple, Any, Literal


class HardwareConfig(BaseModel):
    """`hardware` section of experiment_config.yaml; "auto" = pick for detected hardware."""

    device: Literal["auto", "cpu"] = "auto"
    mixed_precision: Literal["auto", "no", "fp16", "bf16"] = "auto"
    compile: bool | Literal["auto"] = "auto"
    tf32: bool | Literal["auto"] = "auto"
    num_workers: int | Literal["auto"] = Field(default="auto")

    @field_validator("mixed_precision", mode="before")
    @classmethod
    def _yaml_no_is_false(cls, v):
        # YAML 1.1 parses a bare `no` as False
        return "no" if v is False else v


class ViTConfig(BaseModel):
    input_dim: int = Field(default=200)
    out_dim: int = Field(default=3)
    h_split: int = Field(default=10)
    v_split: int = Field(default=10)
    n_heads: int = Field(default=8)
    hid_dim: int = Field(default=128)
    p: float = Field(default=0.05)
    up_scale: int = Field(default=4)
    n_layers: int = Field(default=10)


class OptimizerConfig(BaseModel):
    lr: float = Field(default=3e-4)
    beta1: float = Field(default=0.9)
    beta2: float = Field(default=0.999)
    weight_decay: float = Field(default=1e-2)
    fused: bool = True

    def __iter__(self) -> Generator[Tuple[str, Any], None, None]:
        for_return = {
            "lr": self.lr,
            "betas": (self.beta1, self.beta2),
            "weight_decay": self.weight_decay,
            "fused": self.fused,
        }
        for k, v in for_return.items():
            yield k, v

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__iter__()}


if __name__ == "__main__":
    print(OptimizerConfig().to_dict())
    import pickle

    print(pickle.dumps(OptimizerConfig()))
