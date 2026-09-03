"""
Uniform diffusion language model (UDLM) training pieces.

Implements the pretraining recipe of Sumi (https://huggingface.co/tohoku-nlp/sumi-7b,
arXiv:2606.19005), which is GIDD (von Rütte et al. 2025) in its SNR-reparameterized form
(von Rütte et al. 2026, arXiv:2512.10858) instantiated under pure uniform noise:

- forward process: q(z_t = v | x) = alpha * 1[v = x] + (1 - alpha) / V, alpha = 1 - t,
  i.e. each token is replaced by a uniform vocab draw with probability t,
- per-token loss (unweighted ELBO surrogate): w * KL(q(.|x) || q(.|x_hat)) + w * D_IS(z_t),
  with w the posterior probability that the position was noised,
- log-SNR restricted to lambda in [-9, 9], i.e. t ~ U[sigmoid(-9), sigmoid(9)],
- 50% of sequences use one isotropic (per-sequence) time t, 50% use independent per-token
  times (diffusion forcing, von Rütte et al. 2026),
- the model is bidirectional and time-agnostic: t enters only via the loss, never as input,
- auxiliary z-loss logsumexp(logits)^2 with coefficient 1e-5.

uniform_gidd_loss is a faithful port of Sumi's reference implementation in
modeling_sumi.py at https://huggingface.co/tohoku-nlp/sumi-7b (Apache License 2.0).
"""

import torch
import torch.distributed as dist

# log-SNR lambda in [-9, 9] with lambda = log(alpha / (1 - alpha)) = -logit(t), t = sigma(-lambda)
T_MIN = torch.sigmoid(torch.tensor(-9.0)).item()  # ~1.23e-4: nearly clean
T_MAX = torch.sigmoid(torch.tensor(9.0)).item()   # ~1 - 1.23e-4: nearly pure noise


def sample_times(shape, device, aniso_frac=0.5):
    """
    Sample diffusion times t ~ U[T_MIN, T_MAX]. For each sequence (row), with probability
    aniso_frac sample independent per-token times (anisotropic noise / diffusion forcing),
    otherwise sample a single isotropic time shared by the whole sequence.
    Returns a float tensor of the given (B, T) shape.
    """
    B, T = shape
    t_tokens = torch.rand((B, T), device=device) * (T_MAX - T_MIN) + T_MIN
    t_seqs = torch.rand((B, 1), device=device) * (T_MAX - T_MIN) + T_MIN
    aniso = torch.rand((B, 1), device=device) < aniso_frac
    return torch.where(aniso, t_tokens, t_seqs)


def corrupt(x, t, vocab_size):
    """
    Uniform-noise corruption: replace each token with a uniform draw from the vocabulary
    with probability t (alpha = 1 - t is the signal strength). x stays unchanged otherwise.
    Note replacement with the original token can happen by chance (that's correct).
    """
    replace = torch.rand_like(t) < t
    noise = torch.randint(0, vocab_size, x.shape, device=x.device)
    return torch.where(replace, noise, x)


def uniform_gidd_loss(logits, z_t, labels, t, vocab_size, beta_is=1.0, z_loss_strength=1e-5, eps=1e-12):
    """
    Per-token uniform-only GIDD training loss (port of Sumi's uniform_gidd_loss).

    Args:
        logits: (B, T, V) denoiser logits over the clean token (already fp32, e.g. softcapped).
        z_t: (B, T) noised tokens actually fed to the model.
        labels: (B, T) clean target tokens x. All positions must be valid.
        t: diffusion time in (0, 1), broadcastable to (B, T) (per-sequence or per-token).
        vocab_size: real vocabulary size V (uniform noise is over the existing vocab).
        beta_is: weight of the Itakura-Saito reconstruction term (1.0 in pretraining).
        z_loss_strength: coefficient of the logsumexp(logits)^2 z-loss (1e-5 in pretraining).
        eps: numerical clamp for log / ratios.

    Returns:
        per_token_loss (B, T) in fp32: w * KL + beta_is * w * D_IS (+ z-loss).
    """
    logits = logits[..., :vocab_size].float()

    t = t.float()
    if t.dim() == 1:
        t = t[:, None]
    alpha = (1.0 - t).clamp(min=eps, max=1.0 - eps)
    u = (1.0 - alpha) / float(vocab_size)

    x_hat = torch.softmax(logits, dim=-1)
    log_q_v = torch.log(alpha.unsqueeze(-1) * x_hat + u.unsqueeze(-1))

    log_q_at_x = log_q_v.gather(-1, labels.unsqueeze(-1)).squeeze(-1)
    sum_log_q = log_q_v.sum(dim=-1)
    x_hat_at_zt = x_hat.gather(-1, z_t.unsqueeze(-1)).squeeze(-1)

    # KL[q(.|x) || q(.|x_hat)]: cross entropy collapses since q(.|x) has only two values (alpha + u, u)
    h_ce = -alpha * log_q_at_x - u * sum_log_q
    alpha_plus_u = alpha + u
    h_q = -(
        alpha_plus_u * torch.log(alpha_plus_u.clamp(min=eps))
        + (vocab_size - 1) * u * torch.log(u.clamp(min=eps))
    )
    kl = h_ce - h_q

    # Itakura-Saito divergence at the observed noised token z_t
    is_zt_eq_x = (z_t == labels).to(alpha.dtype)
    q_zt_x = alpha * is_zt_eq_x + u
    q_zt_x_hat = alpha * x_hat_at_zt + u
    log_ratio = torch.log(q_zt_x.clamp(min=eps)) - torch.log(q_zt_x_hat.clamp(min=eps))
    is_div = torch.exp(log_ratio) - log_ratio - 1.0

    # Per-token GIDD weight = posterior probability the position was noised
    w_kept = (1.0 - alpha) / (1.0 + (vocab_size - 1) * alpha)
    w = torch.where(z_t == labels, w_kept, torch.ones_like(w_kept))

    loss = w * kl + beta_is * w * is_div
    if z_loss_strength is not None and z_loss_strength > 0.0:
        log_z = torch.logsumexp(logits, dim=-1)
        loss = loss + float(z_loss_strength) * log_z * log_z
    return loss


@torch.no_grad()
def evaluate_nelbo(model, batches, steps, vocab_size, beta_is=1.0, z_loss_strength=1e-5, aniso_frac=0.5):
    """
    Monte Carlo estimate of the validation NELBO (negative ELBO, nats per token): run the
    training-time corruption on val batches and average the per-token GIDD loss.
    Mirrors evaluate_bpb's structure (per-rank sums, all-reduce), without byte normalization
    (diffusion nats/token are not directly comparable to autoregressive bits-per-byte anyway).
    """
    total_nats = torch.tensor(0.0, dtype=torch.float32, device=model.get_device())
    total_tokens = torch.tensor(0, dtype=torch.int64, device=model.get_device())
    batch_iter = iter(batches)
    for _ in range(steps):
        x, _ = next(batch_iter) # inputs are the clean tokens; the AR-shifted targets are unused
        t = sample_times(x.size(), x.device, aniso_frac=aniso_frac)
        z_t = corrupt(x, t, vocab_size)
        logits = model(z_t) # (B, T, vocab_size) fp32
        total_nats += uniform_gidd_loss(logits, z_t, x, t, vocab_size, beta_is, z_loss_strength).sum()
        total_tokens += x.numel()
    # sum reduce across all ranks
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    if world_size > 1:
        dist.all_reduce(total_nats, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_tokens, op=dist.ReduceOp.SUM)
    return (total_nats / total_tokens.clamp(min=1)).item()
