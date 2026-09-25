"""Verify the sufficient condition from Agent-report/Future Paper/prove-suboptimality.md:

The residual-power family w_k(alpha) = ||r_bar_k||^alpha / E_p[||r_bar_k||^alpha] has

    d F(w(alpha))/d alpha |_{alpha=0} = C * sum_k t_k * log||r_bar_k||,   C > 0   (F = S_w^2/G_w)

so "a small alpha>0 improves the one-step proxy F over alpha=0" holds iff

    boxed condition:  sum_k t_k * log||r_bar_k||  >  0

where, at the unbiased point w=1 (region k has mass p_k = N_k/N):

    a_k = p_k <g, gbar_k>            (= p_k * b_k, b_k from task2-alignment-analysis.py)
    c_k = p_k^2 sigma_k^2 / n_k      (Monte-Carlo sampling-variance contribution of region k's
                                       mean-gradient estimator; sigma_k^2 = population variance,
                                       trace over parameters, of the per-POINT gradient within
                                       region k; n_k = the region's actual expected sample count
                                       under the CURRENT (w=1) sampler)
    V   = sum_k c_k
    t_k = a_k V - ||g||^2 c_k        (sum_k t_k == 0 identically -- checked numerically)

sigma_k^2 is estimated by an unbiased pilot, mirroring the project's existing pilot-variance
convention (taylor_estimation.py): for out_dim=1 and squared loss, the per-point gradient is
g_i = 2 r_i J_i (J_i = d pred_i / d theta), so ||g_i||^2 = 4 r_i^2 ||J_i||^2, computed exactly per
pilot point with vmap(jacrev(...)) (a per-sample NTK-diagonal trick) over `--pilot` points drawn
uniformly at random (with replacement) per region. sigma_k^2 = E_pilot[||g_i||^2] - ||gbar_k||^2,
using the EXACT gbar_k (computed from the full region, not the pilot, by task2-alignment-analysis.py).

n_k is drawn from the same `adaptive_cell_counts` used throughout task2-alignment-energy.py, so
results are directly comparable to that script's S_w^2/G_w numbers (matched --iters/--pct/--modes).
"""
import argparse
import glob
import importlib.util
import json
import os
import sys
from pathlib import Path

import torch
from scipy.stats import spearmanr
from torch.func import functional_call, jacrev, vmap

REPO = Path(__file__).resolve().parents[2]
sys.path.append(str(REPO))
from train_utility_sampling.SamplerWrapper import (  # noqa: E402
    adaptive_cell_counts,
    sample_multiple_from_2d_intervals,
)

_spec = importlib.util.spec_from_file_location("task2_align", REPO / "script/inr_sample/task2-alignment-analysis.py")
align = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(align)


def per_point_grad_sqnorm(inr, params, names, coords, chunk):
    """Exact ||J_i||^2 = sum_theta (d pred_i / d theta)^2 per point. coords: [M, in_dim]."""
    def f(x_single, p):
        pred = functional_call(inr, p, (x_single.unsqueeze(0),))
        return pred.reshape(-1).sum()  # out_dim == 1: scalar

    jac_fn = vmap(jacrev(f, argnums=1), in_dims=(0, None))
    out = torch.empty(coords.shape[0], device=coords.device, dtype=torch.float64)
    for i in range(0, coords.shape[0], chunk):
        xb = coords[i:i + chunk]
        jac = jac_fn(xb, params)  # dict: name -> [B, *param_shape]
        sq = sum((jac[n].reshape(xb.shape[0], -1).double() ** 2).sum(1) for n in names)
        out[i:i + xb.shape[0]] = sq
    return out


@torch.no_grad()
def region_sigma2(ctx, pilot, chunk, seed):
    """Unbiased MC estimate of sigma_k^2 = trace Var_i(g_i) per region.

    torch.no_grad() here only stops an (unwanted) outer autograd graph from building on top of
    the jacrev output -- torch.func transforms manage their own internal autograd regardless.
    """
    inr, graph, bounds, W = ctx["inr"], ctx["graph"], ctx["bounds"], ctx["W"]
    device = graph.space_emb.device
    params = {n: p for n, p in inr.named_parameters() if p.requires_grad}
    names = list(params)
    K = bounds.shape[0]

    Nk = ctx["counts"].to(torch.float64)
    pilot_k = torch.minimum(torch.full_like(Nk, float(pilot)), Nk).clamp_min(1).long()
    k_max = int(pilot_k.max())
    torch.manual_seed(seed)
    xy = sample_multiple_from_2d_intervals(bounds.to(device), k_max, device=device)  # [K, k_max, 2]
    valid = torch.arange(k_max, device=device)[None, :] < pilot_k.to(device)[:, None]
    cell_of = torch.arange(K, device=device)[:, None].expand(-1, k_max)[valid]
    xy_flat = xy[valid]
    idx = xy_flat[:, 1].long() * W + xy_flat[:, 0].long()

    coords = graph.space_emb[idx]
    with torch.no_grad():
        r = (graph.feat[idx] - inr(coords)).reshape(-1).double()
    sqJ = per_point_grad_sqnorm(inr, params, names, coords, chunk)
    g_sqnorm = 4.0 * r ** 2 * sqJ  # ||g_i||^2 per pilot point, exact

    sum_g2 = torch.zeros(K, device=device, dtype=torch.float64)
    sum_g2.index_add_(0, cell_of, g_sqnorm)
    cnt = torch.zeros(K, device=device, dtype=torch.float64)
    cnt.index_add_(0, cell_of, torch.ones_like(g_sqnorm))
    E_g2 = sum_g2 / cnt.clamp_min(1)  # unbiased 2nd-moment estimate (population, i.i.d. pilot draws)

    gk_norm2 = (ctx["gk_norm"].double()) ** 2  # exact, from the full region (not the pilot)
    sigma2 = E_g2 - gk_norm2
    return sigma2, cnt  # cnt: pilot points actually used per region (<= pilot when N_k < pilot)


def analyse_tk(ckpt, device, seed, pilot, chunk, count_mode, frac):
    summary, ctx = align.analyse(ckpt, device, seed, return_ctx=True)
    sigma2, pilot_cnt = region_sigma2(ctx, pilot, chunk, seed)

    values = ctx["sampler"].cached_values
    _, expected = adaptive_cell_counts(values, max(int(int(ctx["counts"].sum()) * ckpt["cfg"].sampling.rate), 1),
                                        count_mode, frac)
    n_k = expected.double().clamp_min(1e-9)

    p = ctx["p"].double()
    b = ctx["b"].double()
    gnorm2 = ctx["gnorm2"]
    a_mean = ctx["a"]["mean"].double()  # ||mean r||_k, the task2.md statistic

    a_k = p * b
    c_k = (p ** 2) * sigma2 / n_k
    V = float(c_k.sum())
    t_k = a_k * V - gnorm2 * c_k

    sum_t = float(t_k.sum())
    scale = float(t_k.abs().sum()) + 1e-30
    log_a = torch.log(a_mean.clamp_min(1e-30))
    boxed_stat = float((t_k * log_a).sum())

    npy = lambda x: x.detach().cpu().numpy()
    rho_t_loga = float(spearmanr(npy(t_k), npy(log_a)).statistic)
    rho_t_a = float(spearmanr(npy(t_k), npy(a_mean)).statistic)
    ascent_energy = float((t_k ** 2 / p.clamp_min(1e-30)).sum())  # should be > 0 (background Prop. check)

    out = {
        "K": int(ctx["bounds"].shape[0]), "gnorm2": gnorm2, "V": V,
        "sum_t_over_scale": sum_t / scale,  # identity check: should be ~0
        "boxed_stat": boxed_stat, "boxed_positive": boxed_stat > 0,
        "rho_t_logA": rho_t_loga, "rho_t_A": rho_t_a,
        "ascent_energy_positive": ascent_energy > 0,
        "frac_t_positive": float((t_k > 0).double().mean()),
        "min_pilot_cnt": int(pilot_cnt.min()), "pilot_requested": pilot,
        "frac_sigma2_negative": float((sigma2 < 0).double().mean()),  # MC noise diagnostic, not an error
        "test_rel_loss": float(ckpt["loss"]),
    }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--epochs", nargs="+", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--pilot", type=int, default=16, help="pilot points per region for sigma_k^2 (with replacement)")
    ap.add_argument("--chunk", type=int, default=1024, help="vmap(jacrev) batch size")
    ap.add_argument("--count-mode", default="min_one", choices=["min_one", "soft"])
    ap.add_argument("--floor-frac", type=float, default=0.1)
    ap.add_argument("--iters", type=int, default=None)
    ap.add_argument("--pct", type=float, default=None)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.set_default_dtype(torch.float32)
    run_dirs = sorted({d for pat in args.runs for d in glob.glob(pat)})
    assert run_dirs
    results = json.load(open(args.out)) if os.path.exists(args.out) else {}
    for rd in run_dirs:
        run = os.path.basename(rd.rstrip("/"))
        for ep in args.epochs:
            key = f"{run}|{ep}"
            path = os.path.join(rd, f"eval_{ep}.pt")
            if key in results or not os.path.exists(path):
                continue
            ckpt = torch.load(path, map_location="cpu", weights_only=False)
            if args.iters is not None or args.pct is not None:
                from omegaconf import OmegaConf
                OmegaConf.set_struct(ckpt["cfg"], False)
                if args.iters is not None:
                    ckpt["cfg"].sampling.adaptive_iterations = args.iters
                if args.pct is not None:
                    ckpt["cfg"].sampling.subdivision_percentage = args.pct
            res = analyse_tk(ckpt, args.device, args.seed, args.pilot, args.chunk, args.count_mode, args.floor_frac)
            res.update(run=run, epoch=ep)
            results[key] = res
            print(f"{key}: K={res['K']} sum_t/scale={res['sum_t_over_scale']:+.1e} "
                  f"boxed_stat={res['boxed_stat']:+.3e} ({'POS' if res['boxed_positive'] else 'neg'}) "
                  f"rho(t,logA)={res['rho_t_logA']:+.3f} frac_t>0={res['frac_t_positive']:.2f}", flush=True)
            json.dump(results, open(args.out, "w"))


if __name__ == "__main__":
    main()
