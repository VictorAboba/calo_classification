import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionBlock(nn.Module):
    def __init__(self, n_heads, hid_dim, p, *args, **kwargs) -> None:
        assert hid_dim % n_heads == 0, "hid_dim must be divisible by n_heads"
        super().__init__(*args, **kwargs)

        self.q_proj = nn.Linear(hid_dim, hid_dim, bias=False)
        self.k_proj = nn.Linear(hid_dim, hid_dim, bias=False)
        self.v_proj = nn.Linear(hid_dim, hid_dim, bias=False)
        self.o_proj = nn.Linear(hid_dim, hid_dim, bias=False)

        self.q_norm = nn.RMSNorm(hid_dim // n_heads)
        self.k_norm = nn.RMSNorm(hid_dim // n_heads)

        self.p = p
        self.n_heads = n_heads
        self.head_dim = hid_dim // n_heads

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, hid_dim = x.shape

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q = self.q_norm(q.view(B, L, self.n_heads, self.head_dim)).permute(0, 2, 1, 3)
        kT = self.k_norm(k.view(B, L, self.n_heads, self.head_dim)).permute(0, 2, 3, 1)
        v = v.view(B, L, self.n_heads, self.head_dim).permute(0, 2, 1, 3)

        scale = self.head_dim**0.5

        atten_weights = F.softmax((q @ kT) / scale, dim=-1)

        output = self.o_proj(
            (F.dropout(atten_weights, p=self.p, training=self.training) @ v)
            .permute(0, 2, 1, 3)
            .reshape(B, L, hid_dim)
        )

        return output


class FFN(nn.Module):
    def __init__(self, hid_dim, up_scale, p, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.up_proj = nn.Linear(hid_dim, hid_dim * up_scale, bias=False)
        self.down_proj = nn.Linear(hid_dim * up_scale, hid_dim, bias=False)
        self.gate = nn.Linear(hid_dim, hid_dim * up_scale, bias=False)
        self.p = p

    def forward(self, x) -> torch.Tensor:
        return self.down_proj(
            F.dropout(
                (self.up_proj(x) * F.silu(self.gate(x))),
                self.p,
                training=self.training,
            )
        )


class CustomViTEncoderLayer(nn.Module):
    def __init__(self, n_heads, hid_dim, p, up_scale, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.atten_prenorm = nn.RMSNorm(hid_dim)
        self.attention = AttentionBlock(n_heads, hid_dim, p)
        self.ffn_prenorm = nn.RMSNorm(hid_dim)
        self.ffn = FFN(hid_dim, up_scale, p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        atten_output = self.attention(self.atten_prenorm(x)) + x
        return self.ffn(self.ffn_prenorm(atten_output)) + atten_output


class CustomViT(nn.Module):
    def __init__(
        self,
        input_dim,
        out_dim,
        h_split,
        v_split,
        n_heads,
        hid_dim,
        p,
        up_scale,
        n_layers,
        *args,
        **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)

        self.hid_dim = hid_dim
        self.cls_token = nn.Parameter(torch.zeros(1, 1, hid_dim), requires_grad=True)
        self.embedding = nn.Linear(input_dim, hid_dim)
        self.h_split, self.v_split = h_split, v_split
        self.v_pos_emb = nn.Parameter(torch.zeros(v_split, hid_dim), requires_grad=True)
        self.h_pos_emb = nn.Parameter(torch.zeros(h_split, hid_dim), requires_grad=True)
        self.vit_encoder = nn.ModuleList(
            [
                CustomViTEncoderLayer(n_heads, hid_dim, p, up_scale)
                for _ in range(n_layers)
            ]
        )
        self.out = nn.Linear(hid_dim, out_dim)

        nn.init.normal_(self.cls_token, std=0.02)
        nn.init.normal_(self.v_pos_emb, std=0.02)
        nn.init.normal_(self.h_pos_emb, std=0.02)

    def forward(
        self, x: torch.Tensor, need_hidden=False
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        assert (
            x.shape[1] == self.h_split * self.v_split
        ), "product of splits must be equal sequence length"
        B, L = x.shape[:2]
        x = self.embedding(x)

        pos_emb = (
            (self.v_pos_emb[:, None, :] + self.h_pos_emb[None, :, :])
            .view(self.h_split * self.v_split, -1)
            .contiguous()
        )
        x += pos_emb.unsqueeze(0)
        x = torch.concat([self.cls_token.expand(B, -1, -1), x], dim=1)
        hidden_state = x

        for layer in self.vit_encoder:
            hidden_state = layer(hidden_state)

        if need_hidden:
            return self.out(hidden_state[:, 0, :]), hidden_state

        return self.out(hidden_state[:, 0, :])


if __name__ == "__main__":
    from einops import rearrange

    x = torch.randn(100, 200)
    x = rearrange(x, "(H p1) (W p2) -> (H W) (p1 p2)", p1=5, p2=10).unsqueeze(0)
    print(x.shape)
    model = CustomViT(50, 2, 20, 20, 5, 10, 0.05, 4, 10)
    print(model)
    y = model(x)
    print(y.shape)
