from pydantic import BaseModel, Field


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
