# https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/switch_transformers/modeling_switch_transformers.py#L285C27-L285C27
from dataclasses import dataclass

import torch


@dataclass
class Config:
    device: torch.device
    n_experts: int
    d_model: int
    k: int
    batch_size: int
    n_ctx: int


class Top1Router(torch.nn.Module):
    def __init__(self, config: Config):
        super(Top1Router, self).__init__()
        self._config = config
        self._classifier = torch.nn.Linear(config.d_model, config.n_experts, device=config.device)

    def forward(self, x: torch.Tensor):  # x: [batch_size, n_ctx, d_model]
        logits = self._classifier(x)  # [batch_size, n_ctx, n_experts]
        probs = torch.nn.functional.softmax(logits, dim=-1)  # [batch_size, n_ctx, n_experts]
        weights, top1_experts = torch.max(probs, dim=-1)  # [batch_size, n_ctx]
        assert top1_experts.shape == (self._config.batch_size, self._config.n_ctx)
        return weights, top1_experts


class Top1MoE(torch.nn.Module):
    def __init__(self, config: Config):
        super(Top1MoE, self).__init__()
        self._config = config
        self._router = Top1Router(config)
        self._experts = torch.nn.ModuleList(
            [
                torch.nn.Linear(config.d_model, config.d_model, device=config.device)
                for _ in range(config.n_experts)
            ]
        )

    def forward(self, x: torch.Tensor):  # x: [batch_size, n_ctx, d_model]
        weights, expert_indices = self._router(x)  # [batch_size, n_ctx]
        out = torch.zeros_like(x)  # [batch_size, n_ctx, d_model]
        for i, expert in enumerate(self._experts):
            mask = expert_indices == i  # [batch_size, n_ctx]
            expert_mask = torch.nn.functional.one_hot(
                expert_indices, num_classes=self._config.n_experts
            )  # [batch_size, n_ctx, n_experts]
            token_indices = expert_mask[:, :, i].bool()  # [batch_size, n_ctx]
            assert torch.equal(mask, token_indices)
            masked_weights = weights[mask].unsqueeze_(-1)  # [batch_size, n_ctx, 1]
            out[token_indices] = (
                expert(x[token_indices]) * masked_weights
            )  # [batch_size, n_ctx, d_model]
        assert out.shape == x.shape
        return out


def test_top1():
    config = Config(
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        n_experts=3,
        d_model=8,
        k=1,
        batch_size=4,
        n_ctx=32,
    )
    moe = Top1MoE(config)
    x = torch.randn(config.batch_size, config.n_ctx, config.d_model, device=config.device)
    out = moe(x)
    assert out.shape == x.shape
    print("test_top1 passed")


class TopkRouter(torch.nn.Module):
    def __init__(self, config: Config):
        super(TopkRouter, self).__init__()
        self._config = config
        self._classifier = torch.nn.Linear(config.d_model, config.n_experts, device=config.device)

    def forward(self, x: torch.Tensor):  # x: [batch_size, n_ctx, d_model]
        logits = self._classifier(x)  # [batch_size, n_ctx, n_experts]
        probs = torch.nn.functional.softmax(logits, dim=-1)  # [batch_size, n_ctx, n_experts]
        topk_probs, topk_experts = torch.topk(probs, k=self._config.k, dim=-1)
        assert topk_probs.shape == (self._config.batch_size, self._config.n_ctx, self._config.k)
        assert topk_experts.shape == (self._config.batch_size, self._config.n_ctx, self._config.k)
        topk_expert_masks = torch.nn.functional.one_hot(
            topk_experts, num_classes=self._config.n_experts
        )  # [batch_size, n_ctx, k, n_experts]
        topk_expert_masks = topk_expert_masks.sum(dim=-2).bool()  # [batch_size, n_ctx, n_experts]
        topk_expert_probs = probs
        assert topk_expert_masks.shape == (
            self._config.batch_size,
            self._config.n_ctx,
            self._config.n_experts,
        ), topk_expert_masks.shape
        assert topk_expert_probs.shape == (
            self._config.batch_size,
            self._config.n_ctx,
            self._config.n_experts,
        )
        return topk_expert_masks, topk_expert_probs


class TopkMoE(torch.nn.Module):
    def __init__(self, config: Config):
        super(TopkMoE, self).__init__()
        self._config = config
        self._router = TopkRouter(config)
        self._experts = torch.nn.ModuleList(
            [
                torch.nn.Linear(config.d_model, config.d_model, device=config.device)
                for _ in range(config.n_experts)
            ]
        )

    def forward(self, x: torch.Tensor):  # x: [batch_size, n_ctx, d_model]
        out = torch.zeros_like(x)  # [batch_size, n_ctx, d_model]
        topk_expert_masks, topk_expert_probs = self._router(x)  # [batch_size, n_ctx, k, n_experts]
        for i, expert in enumerate(self._experts):
            mask = topk_expert_masks[:, :, i].bool()
            assert mask.shape == (self._config.batch_size, self._config.n_ctx)
            weight = topk_expert_probs[mask][:, i].unsqueeze_(-1)
            out[mask] = expert(x[mask]) * weight
        return out


def test_topk():
    config = Config(
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        n_experts=3,
        d_model=8,
        k=2,
        batch_size=4,
        n_ctx=32,
    )
    moe = TopkMoE(config)
    x = torch.randn(config.batch_size, config.n_ctx, config.d_model, device=config.device)
    out = moe(x)
    assert out.shape == x.shape
    print("test_topk passed")


def test_cmp():
    config = Config(
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        n_experts=3,
        d_model=8,
        k=1,
        batch_size=4,
        n_ctx=32,
    )
    torch.manual_seed(0)
    top1_moe = Top1MoE(config)
    torch.manual_seed(0)
    topk_moe = TopkMoE(config)

    assert torch.equal(
        top1_moe._router._classifier.weight, topk_moe._router._classifier.weight
    ), "weight"
    assert torch.equal(top1_moe._router._classifier.bias, topk_moe._router._classifier.bias), "bias"
    for i in range(config.n_experts):
        assert torch.equal(top1_moe._experts[i].weight, topk_moe._experts[i].weight), f"weight {i}"
        assert torch.equal(top1_moe._experts[i].bias, topk_moe._experts[i].bias), f"bias {i}"

    x = torch.randn(config.batch_size, config.n_ctx, config.d_model, device=config.device)
    top1_out = top1_moe(x)
    topk_out = topk_moe(x)
    assert torch.equal(top1_out, topk_out)
    print("test_cmp passed")


def main():
    test_top1()
    test_topk()
    test_cmp()


if __name__ == "__main__":
    main()
