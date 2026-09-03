"""
Test UDLM (uniform diffusion language model) pieces:
- time sampling (isotropic/anisotropic mixture, log-SNR range)
- uniform corruption
- GIDD loss parity with Sumi's reference implementation + sanity checks
- bidirectional attention in the GPT model

Run: python -m pytest tests/test_udlm.py -v -s
"""
import torch
import pytest

import nanochat.flash_attention as fa_module
import nanochat.udlm as udlm
from nanochat.flash_attention import HAS_FA3
from nanochat.gpt import GPT, GPTConfig


def set_impl(impl):
    fa_module._override_impl = impl
    fa_module.USE_FA3 = fa_module._resolve_use_fa3()


# =============================================================================
# Sumi's reference uniform_gidd_loss (verbatim port from
# https://huggingface.co/tohoku-nlp/sumi-7b/blob/main/modeling_sumi.py, Apache-2.0)
# =============================================================================
def sumi_reference_gidd_loss(logits, z_t, labels, t, vocab_size, beta_is=1.0, z_loss_strength=1e-5, eps=1e-12):
    logits = logits[..., :vocab_size].float()

    t = t.float()
    if t.dim() == 1:
        t = t[:, None]
    alpha = (1.0 - t).clamp(min=eps, max=1.0 - eps).expand_as(z_t)
    u = (1.0 - alpha) / float(vocab_size)

    x_hat = torch.softmax(logits, dim=-1)
    log_q_v = torch.log(alpha.unsqueeze(-1) * x_hat + u.unsqueeze(-1))

    log_q_at_x = log_q_v.gather(-1, labels.unsqueeze(-1)).squeeze(-1)
    sum_log_q = log_q_v.sum(dim=-1)
    x_hat_at_zt = x_hat.gather(-1, z_t.unsqueeze(-1)).squeeze(-1)

    h_ce = -alpha * log_q_at_x - u * sum_log_q
    alpha_plus_u = alpha + u
    h_q = -(
        alpha_plus_u * torch.log(alpha_plus_u.clamp(min=eps))
        + (vocab_size - 1) * u * torch.log(u.clamp(min=eps))
    )
    kl = h_ce - h_q

    is_zt_eq_x = (z_t == labels).to(alpha.dtype)
    q_zt_x = alpha * is_zt_eq_x + u
    q_zt_x_hat = alpha * x_hat_at_zt + u
    log_ratio = torch.log(q_zt_x.clamp(min=eps)) - torch.log(q_zt_x_hat.clamp(min=eps))
    is_div = torch.exp(log_ratio) - log_ratio - 1.0

    w_kept = (1.0 - alpha) / (1.0 + (vocab_size - 1) * alpha)
    w = torch.where(z_t == labels, w_kept, torch.ones_like(w_kept))

    loss = w * kl + beta_is * w * is_div
    if z_loss_strength is not None and z_loss_strength > 0.0:
        log_z = torch.logsumexp(logits, dim=-1)
        loss = loss + float(z_loss_strength) * log_z * log_z
    return loss


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float32 if not torch.cuda.is_available() else torch.bfloat16
V = 512  # small test vocab


# =============================================================================
# Time sampling and corruption
# =============================================================================
class TestSampleTimes:
    def test_range_and_shape(self):
        torch.manual_seed(0)
        t = udlm.sample_times((4, 16), DEVICE)
        assert t.shape == (4, 16)
        assert t.min() >= udlm.T_MIN - 1e-7
        assert t.max() <= udlm.T_MAX + 1e-7

    def test_isotropic_rows_constant(self):
        t = udlm.sample_times((8, 64), DEVICE, aniso_frac=0.0)
        # every row must be exactly constant (isotropic time per sequence)
        row_std = t.std(dim=1)
        assert (row_std == 0).all()

    def test_anisotropic_rows_vary(self):
        torch.manual_seed(0)
        t = udlm.sample_times((8, 64), DEVICE, aniso_frac=1.0)
        row_std = t.std(dim=1)
        assert (row_std > 0).all()

    def test_mixture_fraction(self):
        torch.manual_seed(0)
        n_rows = 4000
        t = udlm.sample_times((n_rows, 8), DEVICE, aniso_frac=0.5)
        constant_rows = (t.std(dim=1) == 0).float().mean().item()
        assert abs(constant_rows - 0.5) < 0.05  # loose binomial tolerance


class TestCorrupt:
    def test_clean_at_t_zero(self):
        x = torch.randint(0, V, (4, 16), device=DEVICE)
        t = torch.zeros(4, 16, device=DEVICE)
        z = udlm.corrupt(x, t, V)
        assert torch.equal(z, x)

    def test_all_noised_at_t_one(self):
        torch.manual_seed(0)
        x = torch.randint(0, V, (4, 1024), device=DEVICE)
        t = torch.ones(4, 1024, device=DEVICE)
        z = udlm.corrupt(x, t, V)
        assert (z < V).all()
        # with independent uniform draws at V=512, almost all positions differ from x
        matching_frac = (z == x).float().mean().item()
        assert matching_frac < 0.05

    def test_replace_rate_scales_with_t(self):
        torch.manual_seed(0)
        x = torch.randint(0, V, (64, 4096), device=DEVICE)
        t = torch.full((64, 4096), 0.5, device=DEVICE)
        z = udlm.corrupt(x, t, V)
        diff_frac = (z != x).float().mean().item()
        # expected diff rate = t * (1 - 1/V) ≈ 0.499
        assert abs(diff_frac - 0.5) < 0.02


# =============================================================================
# GIDD loss
# =============================================================================
class TestGiddLoss:
    def _make_inputs(self, B=2, T=16, seed=0):
        g = torch.Generator(device=DEVICE).manual_seed(seed)
        x = torch.randint(0, V, (B, T), generator=g, device=DEVICE)
        t = torch.rand((B, T), generator=g, device=DEVICE) * (udlm.T_MAX - udlm.T_MIN) + udlm.T_MIN
        z = udlm.corrupt(x, t, V)
        logits = torch.randn(B, T, V, generator=g, device=DEVICE)
        return logits, z, x, t

    def test_matches_sumi_reference(self):
        logits, z, x, t = self._make_inputs()
        ours = udlm.uniform_gidd_loss(logits, z, x, t, V)
        ref = sumi_reference_gidd_loss(logits, z, x, t, V)
        max_diff = (ours - ref).abs().max().item()
        print(f"loss parity: max_diff={max_diff:.3e}")
        assert torch.allclose(ours, ref, atol=1e-5, rtol=1e-5)

    def test_matches_sumi_reference_isotropic_t(self):
        logits, z, x, t = self._make_inputs()
        t = t[:, :1].expand_as(x).contiguous()  # one time per sequence
        ours = udlm.uniform_gidd_loss(logits, z, x, t, V)
        ref = sumi_reference_gidd_loss(logits, z, x, t, V)
        assert torch.allclose(ours, ref, atol=1e-5, rtol=1e-5)

    def test_perfect_prediction_low_loss(self):
        B, T = 2, 16
        x = torch.randint(0, V, (B, T), device=DEVICE)
        t = torch.full((B, T), 0.5, device=DEVICE)
        z = udlm.corrupt(x, t, V)
        # strongly peaked logits at the clean token
        logits = torch.zeros(B, T, V, device=DEVICE)
        logits.scatter_(-1, x.unsqueeze(-1), 20.0)
        loss_good = udlm.uniform_gidd_loss(logits, z, x, t, V, z_loss_strength=0.0).mean().item()
        loss_flat = udlm.uniform_gidd_loss(torch.zeros(B, T, V, device=DEVICE), z, x, t, V, z_loss_strength=0.0).mean().item()
        print(f"peaked loss={loss_good:.4f}, flat loss={loss_flat:.4f}")
        assert loss_good < 0.1 * loss_flat

    def test_z_loss_grows_with_logit_scale(self):
        logits, z, x, t = self._make_inputs()
        small = udlm.uniform_gidd_loss(logits, z, x, t, V, z_loss_strength=1e-5).mean().item()
        large = udlm.uniform_gidd_loss(5 * logits, z, x, t, V, z_loss_strength=1e-5).mean().item()
        assert large > small

    def test_backward(self):
        logits, z, x, t = self._make_inputs()
        logits.requires_grad_(True)
        loss = udlm.uniform_gidd_loss(logits, z, x, t, V).mean()
        loss.backward()
        assert logits.grad is not None
        assert not torch.isnan(logits.grad).any()
        assert logits.grad.abs().sum() > 0


# =============================================================================
# Bidirectional attention in the GPT model
# =============================================================================
def build_tiny_model(bidirectional, device):
    config = GPTConfig(
        sequence_len=64, vocab_size=V, n_layer=2, n_head=2, n_kv_head=2, n_embd=64,
        window_pattern="L", bidirectional=bidirectional,
    )
    with torch.device("meta"):
        model = GPT(config)
    model.to_empty(device=device)
    model.init_weights()
    # init_weights zeros every c_proj, which at init disconnects the attention/MLP
    # outputs from the residual stream entirely (logits then depend only on the
    # position's own token) - replace them with small random weights so that the
    # cross-position information flow probed below actually reaches the logits.
    torch.manual_seed(123)
    with torch.no_grad():
        for block in model.transformer.h:
            block.attn.c_proj.weight.normal_(0, 0.5)
            block.mlp.c_proj.weight.normal_(0, 0.5)
    model.eval()
    return model


class TestBidirectionalAttention:
    DEVICE = DEVICE

    def run_flip_probe(self, impl, device):
        """Flip late tokens; measure how much logits change at an earlier position."""
        set_impl(impl)
        try:
            torch.manual_seed(0)
            x = torch.randint(0, V, (2, 64), device=device)
            x_flip = x.clone()
            x_flip[:, 40:] = (x_flip[:, 40:] + 17) % V  # flip the whole tail

            model_bi = build_tiny_model(bidirectional=True, device=device)
            model_ar = build_tiny_model(bidirectional=False, device=device)
            with torch.no_grad():
                logits_bi = model_bi(x)
                logits_bi_flip = model_bi(x_flip)
                logits_ar = model_ar(x)
                logits_ar_flip = model_ar(x_flip)

            early = 5  # position strictly before the flipped token at 40
            delta_bi = (logits_bi[:, early] - logits_bi_flip[:, early]).abs().max().item()
            delta_ar = (logits_ar[:, early] - logits_ar_flip[:, early]).abs().max().item()
            return delta_bi, delta_ar
        finally:
            set_impl(None)

    def test_bidirectional_attention(self):
        """Causal model must be invariant to future tokens; bidirectional must not."""
        for impl in (["fa3", "sdpa"] if HAS_FA3 else ["sdpa"]):
            delta_bi, delta_ar = self.run_flip_probe(impl, self.DEVICE)
            print(f"[{impl}] delta bidirectional={delta_bi:.6f}, delta causal={delta_ar:.6f}")
            assert delta_ar < 1e-5, f"[{impl}] causal attention leaked future information"
            assert delta_bi > 1e-3, f"[{impl}] bidirectional attention ignored a future token"


if __name__ == "__main__":
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"HAS_FA3: {HAS_FA3}")
    pytest.main([__file__, "-v", "-s"])
