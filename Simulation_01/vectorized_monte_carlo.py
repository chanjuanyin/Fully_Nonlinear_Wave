"""Vectorized (batched) Monte Carlo engine for the branching wave-equation solver.

Runs the same stochastic branching algorithm as the recursive
{real,complex}_d{1,2,3}_monte_carlo.py scripts, but processes an entire batch of
tree samples simultaneously as flat tensors ("frontier" of active nodes), so it
can exploit the GPU and reach 10^7 samples.

Design notes
------------
* The random functional H is a PRODUCT of per-node factors (eq. GlobalWeight in
  the paper).  We accumulate log-factors per sample with index_add_ (safe with
  duplicate indices) and exponentiate at the end:  exp(sum log z_k) == prod z_k
  for ANY branch choice of the complex log, so branch cuts are harmless.
* Boundary weights need tilde_phi / tilde_psi, which require arbitrary-order
  autograd derivatives.  Boundary nodes are grouped by their (identical) code
  vector and each group is evaluated in one batched autograd call, reusing the
  exact tilde_phi / tilde_psi implementations of the audited per-dimension
  modules, so both engines share one source of truth for the boundary data.
* Offspring sampling reproduces QC exactly, vectorized with masked
  acceptance-rejection loops.

Usage:
    python vectorized_monte_carlo.py real_d1 [--num-samples 10000000]
                                             [--chunk-size 1000000]
                                             [--device cuda]
"""

import argparse
import csv
import math
import os
import sys
import time

import torch

SIM_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SIM_DIR)

import real_d1_monte_carlo as mod_d1  # noqa: E402
import real_d2_monte_carlo as mod_d2  # noqa: E402
import real_d3_monte_carlo as mod_d3  # noqa: E402

OMEGA = 0.5 + 0.0j
K = 4 * OMEGA / 3
LAMBDA = 1.0

MODULES = {1: mod_d1, 2: mod_d2, 3: mod_d3}

# rot = 1 for the real cases, i for the complex cases (initial data are
# functions of rot*(z_1+...+z_d)); a is the wave-speed factor.
CASES = {
    "real_d1":    dict(d=1, a=1.0 + 0.0j,                 rot=1.0 + 0.0j),
    "real_d2":    dict(d=2, a=(1.0 / math.sqrt(2)) + 0j,  rot=1.0 + 0.0j),
    "real_d3":    dict(d=3, a=(1.0 / math.sqrt(3)) + 0j,  rot=1.0 + 0.0j),
    "complex_d1": dict(d=1, a=1j,                         rot=1j),
    "complex_d2": dict(d=2, a=(1.0 / math.sqrt(2)) * 1j,  rot=1j),
    "complex_d3": dict(d=3, a=(1.0 / math.sqrt(3)) * 1j,  rot=1j),
}


def make_initial_data(rot):
    """phi, psi, f exactly as in the per-case __main__ blocks (fixed versions)."""
    def phi(*zs):
        s = sum(zs)
        return 4 * torch.arctan(torch.exp(K * rot * s))

    def psi(*zs):
        s = sum(zs)
        return (8 * OMEGA / 3) * (torch.exp(K * rot * s) / (1 + torch.exp(2 * K * rot * s)))

    def f(u):
        return -(4 * (OMEGA ** 2) / 3) * torch.sin(u)

    return phi, psi, f


# --------------------------------------------------------------------------
# vectorized sampling helpers
# --------------------------------------------------------------------------

def rand_unif_int(high_inclusive):
    """Per-element uniform integers in {0, ..., high_inclusive[k]} (int64 tensor)."""
    u = torch.rand(high_inclusive.shape, device=high_inclusive.device)
    return torch.minimum((u * (high_inclusive + 1).to(u.dtype)).long(), high_inclusive)


def ar_type1_beta(alpha, i_idx):
    """Acceptance-rejection for c = partial^alpha (alpha != 0), vectorized.

    alpha: (m, d) int64; i_idx: (m,) first index with alpha_i > 0.
    beta_i ~ Unif{0..alpha_i-1} accepted w.p. (alpha_i - beta_i)/alpha_i;
    other coordinates uniform on {0..alpha_k}.
    Returns beta (m, d) and gamma1 (m,) float (= (alpha_i-beta_i)/alpha_i).
    """
    m, d = alpha.shape
    dev = alpha.device
    ar = torch.arange(m, device=dev)
    alpha_i = alpha[ar, i_idx]
    beta = rand_unif_int(alpha)  # provisional for all coords
    beta_i = torch.zeros(m, dtype=torch.int64, device=dev)
    pending = torch.ones(m, dtype=torch.bool, device=dev)
    for _ in range(10000):
        idx = pending.nonzero(as_tuple=True)[0]
        if idx.numel() == 0:
            break
        ai = alpha_i[idx]
        prop = rand_unif_int(ai - 1)  # {0..alpha_i-1}
        acc_p = (ai - prop).to(torch.float32) / ai.to(torch.float32)
        acc = torch.rand(idx.shape, device=dev) <= acc_p
        beta_i[idx[acc]] = prop[acc]
        pending[idx[acc]] = False
    else:
        raise RuntimeError("type-1 acceptance-rejection did not converge")
    beta[ar, i_idx] = beta_i
    gamma1 = (alpha_i - beta_i).to(torch.float32) / alpha_i.to(torch.float32)
    return beta, gamma1


def ar_family_beta(alpha, i_idx):
    """AR for the i-indexed families of types 2/3, vectorized.

    beta_i ~ Unif{0..alpha_i} accepted w.p. 4(alpha_i-beta_i+1)(beta_i+1)/(2+alpha_i)^2;
    other coordinates uniform.  Returns beta (m, d) and rho = (beta_i+1)(alpha_i-beta_i+1).
    """
    m, d = alpha.shape
    dev = alpha.device
    ar = torch.arange(m, device=dev)
    alpha_i = alpha[ar, i_idx]
    beta = rand_unif_int(alpha)
    beta_i = torch.zeros(m, dtype=torch.int64, device=dev)
    pending = torch.ones(m, dtype=torch.bool, device=dev)
    for _ in range(10000):
        idx = pending.nonzero(as_tuple=True)[0]
        if idx.numel() == 0:
            break
        ai = alpha_i[idx]
        prop = rand_unif_int(ai)  # {0..alpha_i}
        acc_p = (4.0 * (ai - prop + 1) * (prop + 1)).to(torch.float32) / ((2 + ai) ** 2).to(torch.float32)
        acc = torch.rand(idx.shape, device=dev) <= acc_p
        beta_i[idx[acc]] = prop[acc]
        pending[idx[acc]] = False
    else:
        raise RuntimeError("family acceptance-rejection did not converge")
    beta[ar, i_idx] = beta_i
    rho = ((beta_i + 1) * (alpha_i - beta_i + 1)).to(torch.float32)
    return beta, rho


# --------------------------------------------------------------------------
# boundary evaluation (grouped by code)
# --------------------------------------------------------------------------

def eval_boundary(d, module, phi, psi, f, a_t, codes, z, trem, U, theta, lam):
    """Return log complex boundary factors e^{lam*trem} * W^b for each node.

    codes: (g, d+2) int64 (all rows identical within a call!),
    z: (g, d) complex, trem/U/theta: (g,) float.
    """
    code = [int(v) for v in codes[0].tolist()]
    a = a_t

    if d == 1:
        zp = (z[:, 0] + a * trem).detach().requires_grad_(True)
        zm = (z[:, 0] - a * trem).detach().requires_grad_(True)
        zpsi = (z[:, 0] + a * trem * (2 * U - 1)).detach().requires_grad_(True)
        i1 = 0.5 * module.tilde_phi(code, phi, psi, f, zp, a)
        i2 = 0.5 * module.tilde_phi(code, phi, psi, f, zm, a)
        i3 = trem * module.tilde_psi(code, phi, psi, f, zpsi, a)
        wb = (i1 + i2 + i3).detach()
    else:
        if d == 2:
            r = trem * torch.sqrt(1 - (1 - U) ** 2)
            y = [a * r * torch.cos(theta), a * r * torch.sin(theta)]
        else:
            eta = torch.arccos(1 - 2 * U)
            y = [a * trem * torch.sin(eta) * torch.cos(theta),
                 a * trem * torch.sin(eta) * torch.sin(theta),
                 a * trem * torch.cos(eta)]
        pts = [(z[:, k] + y[k]).detach().requires_grad_(True) for k in range(d)]
        i1 = module.tilde_phi(code, phi, psi, f, *pts, a)
        i2 = 0
        coord_names = ["z1", "z2", "z3"]
        for k in range(d):
            i2 = i2 + y[k] * module.gradient_tilde_phi(coord_names[k], code, phi, psi, f, *pts, a)
        i3 = trem * module.tilde_psi(code, phi, psi, f, *pts, a)
        wb = (i1 + i2 + i3).detach()

    factor = torch.exp(torch.tensor(lam, device=wb.device) * trem) * wb
    return torch.log(factor.to(torch.complex128))


# --------------------------------------------------------------------------
# offspring mechanism (vectorized QC + interior weights)
# --------------------------------------------------------------------------

def interior_step(d, a_sq, codes, lam, tau):
    """Vectorized QC + interior node weight.

    codes: (m, d+2) int64 for interior nodes; tau: (m,) their lifetimes.
    Returns (log_factors (m,) complex128, child1 codes (m, d+2),
             child2 codes (m, d+2) or -1 rows where absent, has2 (m,) bool).
    """
    m = codes.shape[0]
    dev = codes.device
    ctype = codes[:, 0]
    alpha = codes[:, 1:1 + d]
    j = codes[:, -1]

    prod1a = torch.ones(m, dtype=torch.float32, device=dev)
    for k in range(d):
        prod1a = prod1a * (1 + alpha[:, k]).to(torch.float32)

    factor = torch.zeros(m, dtype=torch.complex128, device=dev)
    child1 = torch.zeros_like(codes)
    child2 = torch.full_like(codes, -1)
    has2 = torch.ones(m, dtype=torch.bool, device=dev)

    # ---- type 0: id -> (1, f, void) --------------------------------------
    mask = ctype == 0
    if mask.any():
        idx = mask.nonzero(as_tuple=True)[0]
        factor[idx] = 1.0
        c1 = torch.zeros(idx.numel(), d + 2, dtype=torch.int64, device=dev)
        c1[:, 0] = 2  # f^(0)
        c1[:, -1] = 0
        child1[idx] = c1
        has2[idx] = False

    # ---- type 1: (1/alpha!) d^alpha, alpha != 0 --------------------------
    mask = ctype == 1
    if mask.any():
        idx = mask.nonzero(as_tuple=True)[0]
        al = alpha[idx]
        i_idx = torch.argmax((al > 0).to(torch.int64), dim=1)  # first nonzero coord
        beta, _g1 = ar_type1_beta(al, i_idx)
        factor[idx] = (prod1a[idx] / 2.0).to(torch.complex128)
        c1 = torch.zeros(idx.numel(), d + 2, dtype=torch.int64, device=dev)
        c1[:, 0] = 2
        c1[:, 1:1 + d] = beta
        c1[:, -1] = 1
        c2 = torch.zeros_like(c1)
        c2[:, 0] = 1
        c2[:, 1:1 + d] = al - beta
        c2[:, -1] = -1
        child1[idx] = c1
        child2[idx] = c2

    # ---- types 2 and 3 (share family structure) --------------------------
    for tval in (2, 3):
        mask = ctype == tval
        if not mask.any():
            continue
        idx = mask.nonzero(as_tuple=True)[0]
        al = alpha[idx]
        jj = j[idx]
        n = idx.numel()
        mfam = torch.randint(1, d + 3, (n,), device=dev)  # {1..d+2}

        base = prod1a[idx] * (d + 2)
        if tval == 3:
            base_uniform = 2.0 * base  # gamma1 = 2 in the first two families
        else:
            base_uniform = base

        # families 1 and 2: uniform beta over the box
        for fam in (1, 2):
            fmask = mfam == fam
            if not fmask.any():
                continue
            fi = idx[fmask]
            beta = rand_unif_int(al[fmask])
            c1 = torch.zeros(fi.numel(), d + 2, dtype=torch.int64, device=dev)
            c2 = torch.zeros_like(c1)
            c1[:, 1:1 + d] = beta
            c2[:, 1:1 + d] = al[fmask] - beta
            if tval == 2:
                if fam == 1:
                    c1[:, 0] = 2; c1[:, -1] = jj[fmask] + 2
                    c2[:, 0] = 3; c2[:, -1] = -1
                else:
                    c1[:, 0] = 2; c1[:, -1] = jj[fmask] + 1
                    c2[:, 0] = 2; c2[:, -1] = 0
            else:
                if fam == 1:
                    c1[:, 0] = 5; c1[:, -1] = -1
                    c2[:, 0] = 5; c2[:, -1] = -1
                else:
                    c1[:, 0] = 3; c1[:, -1] = -1
                    c2[:, 0] = 2; c2[:, -1] = 1
            factor[fi] = base_uniform[fmask].to(torch.complex128)
            child1[fi] = c1
            child2[fi] = c2

        # families 3..d+2: coordinate i = mfam - 2, non-uniform beta_i
        fmask = mfam >= 3
        if fmask.any():
            fi = idx[fmask]
            i_idx = (mfam[fmask] - 3).to(torch.int64)  # 0-based coordinate
            alf = al[fmask]
            beta, _rho = ar_family_beta(alf, i_idx)
            arng = torch.arange(fi.numel(), device=dev)
            ai = alf[arng, i_idx].to(torch.float32)
            coef = base[fmask] * (2 + ai) * (3 + ai) / 6.0
            if tval == 3:
                coef = coef * 2.0
            fac = (-a_sq) * coef.to(torch.complex128)
            factor[fi] = fac
            e_i = torch.zeros(fi.numel(), d, dtype=torch.int64, device=dev)
            e_i[arng, i_idx] = 1
            c1 = torch.zeros(fi.numel(), d + 2, dtype=torch.int64, device=dev)
            c2 = torch.zeros_like(c1)
            c1[:, 1:1 + d] = beta + e_i
            c2[:, 1:1 + d] = alf - beta + e_i
            if tval == 2:
                c1[:, 0] = 2; c1[:, -1] = jj[fmask] + 1
                c2[:, 0] = 1; c2[:, -1] = -1
            else:
                c1[:, 0] = 4; c1[:, -1] = -1
                c2[:, 0] = 4; c2[:, -1] = -1
            child1[fi] = c1
            child2[fi] = c2

    # ---- type 4 ----------------------------------------------------------
    mask = ctype == 4
    if mask.any():
        idx = mask.nonzero(as_tuple=True)[0]
        beta = rand_unif_int(alpha[idx])
        factor[idx] = prod1a[idx].to(torch.complex128)
        c1 = torch.zeros(idx.numel(), d + 2, dtype=torch.int64, device=dev)
        c1[:, 0] = 2
        c1[:, 1:1 + d] = beta
        c1[:, -1] = 1
        c2 = torch.zeros_like(c1)
        c2[:, 0] = 4
        c2[:, 1:1 + d] = alpha[idx] - beta
        c2[:, -1] = -1
        child1[idx] = c1
        child2[idx] = c2

    # ---- type 5 ----------------------------------------------------------
    mask = ctype == 5
    if mask.any():
        idx = mask.nonzero(as_tuple=True)[0]
        beta = rand_unif_int(alpha[idx])
        mfam = torch.randint(1, 3, (idx.numel(),), device=dev)
        factor[idx] = (2.0 * prod1a[idx]).to(torch.complex128)
        c1 = torch.zeros(idx.numel(), d + 2, dtype=torch.int64, device=dev)
        c2 = torch.zeros_like(c1)
        c1[:, 1:1 + d] = beta
        c2[:, 1:1 + d] = alpha[idx] - beta
        m1 = mfam == 1
        # family 1: (f^(2), (d_t .)^2); family 2: (f^(1), d_tt)
        c1[:, 0] = 2
        c1[m1, -1] = 2
        c1[~m1, -1] = 1
        c2[m1, 0] = 3
        c2[m1, -1] = -1
        c2[~m1, 0] = 5
        c2[~m1, -1] = -1
        child1[idx] = c1
        child2[idx] = c2

    # multiply by the universal interior factor tau * e^{lam tau} / lam
    time_factor = (tau.to(torch.float64) * torch.exp(lam * tau.to(torch.float64)) / lam)
    log_factor = torch.log(factor * time_factor.to(torch.complex128))
    return log_factor, child1, child2, has2


# --------------------------------------------------------------------------
# main driver
# --------------------------------------------------------------------------

def run_chunk(case_cfg, module, phi, psi, f, z0, t, lam, n, device, max_depth=10000):
    d = case_cfg["d"]
    a = torch.tensor(case_cfg["a"], dtype=torch.complex64, device=device)
    a_sq = torch.tensor(case_cfg["a"], dtype=torch.complex128, device=device) ** 2

    # per-sample log-accumulators (complex128 as two float64 channels)
    logacc = torch.zeros(n, dtype=torch.complex128, device=device)

    # frontier: root node per sample, code = id
    z = torch.full((n, d), z0, dtype=torch.complex64, device=device)
    trem = torch.full((n,), float(t), dtype=torch.float32, device=device)
    codes = torch.zeros(n, d + 2, dtype=torch.int64, device=device)
    codes[:, -1] = -1  # id code vector [0, 0...0, -1]
    sidx = torch.arange(n, dtype=torch.int64, device=device)

    # boundary nodes are buffered during the tree sweep and evaluated once at
    # the end, grouped by code (one batched autograd call per distinct code)
    bz, btrem, bU, btheta, bcodes_l, bsidx = [], [], [], [], [], []

    for _depth in range(max_depth):
        m = z.shape[0]
        if m == 0:
            break
        tau = -torch.log1p(-torch.rand(m, device=device)) / lam
        U = torch.rand(m, device=device)
        theta = torch.rand(m, device=device) * (2 * math.pi)

        bmask = tau >= trem
        if bmask.any():
            bidx = bmask.nonzero(as_tuple=True)[0]
            bz.append(z[bidx])
            btrem.append(trem[bidx])
            bU.append(U[bidx])
            btheta.append(theta[bidx])
            bcodes_l.append(codes[bidx])
            bsidx.append(sidx[bidx])

        # ----- interior nodes: offspring + weights ------------------------
        imask = ~bmask
        if not imask.any():
            break
        iidx = imask.nonzero(as_tuple=True)[0]
        logf, c1, c2, has2 = interior_step(d, a_sq, codes[iidx], lam, tau[iidx])
        logacc.index_add_(0, sidx[iidx], logf)

        # children positions (same displacement for both children)
        ti = tau[iidx]
        Ui = U[iidx]
        thi = theta[iidx]
        zi = z[iidx]
        if d == 1:
            disp = (a * (ti * (2 * Ui - 1)).to(torch.complex64)).unsqueeze(1)
        elif d == 2:
            r = ti * torch.sqrt(1 - (1 - Ui) ** 2)
            disp = torch.stack([a * (r * torch.cos(thi)).to(torch.complex64),
                                a * (r * torch.sin(thi)).to(torch.complex64)], dim=1)
        else:
            eta = torch.arccos(1 - 2 * Ui)
            disp = torch.stack([a * (ti * torch.sin(eta) * torch.cos(thi)).to(torch.complex64),
                                a * (ti * torch.sin(eta) * torch.sin(thi)).to(torch.complex64),
                                a * (ti * torch.cos(eta)).to(torch.complex64)], dim=1)
        zc = zi + disp
        tc = trem[iidx] - ti
        sc = sidx[iidx]

        # stack child1 (all) + child2 (where present)
        z = torch.cat([zc, zc[has2]], dim=0)
        trem = torch.cat([tc, tc[has2]], dim=0)
        codes = torch.cat([c1, c2[has2]], dim=0)
        sidx = torch.cat([sc, sc[has2]], dim=0)
    else:
        raise RuntimeError("tree depth guard exceeded")

    # ----- evaluate all buffered boundary nodes, grouped by code ----------
    if bz:
        z_b = torch.cat(bz)
        trem_b = torch.cat(btrem)
        U_b = torch.cat(bU)
        theta_b = torch.cat(btheta)
        codes_b = torch.cat(bcodes_l)
        sidx_b = torch.cat(bsidx)
        uniq, inv = torch.unique(codes_b, dim=0, return_inverse=True)
        for g in range(uniq.shape[0]):
            gsel = (inv == g).nonzero(as_tuple=True)[0]
            logf = eval_boundary(d, module, phi, psi, f, a,
                                 codes_b[gsel], z_b[gsel], trem_b[gsel],
                                 U_b[gsel], theta_b[gsel], lam)
            logacc.index_add_(0, sidx_b[gsel], logf)

    return torch.exp(logacc)  # (n,) complex128


def run_case(case, num_samples, chunk_size, device_str, seed=1234, out_file=None):
    cfg = CASES[case]
    d = cfg["d"]
    module = MODULES[d]
    phi, psi, f = make_initial_data(cfg["rot"])
    device = torch.device(device_str)
    torch.manual_seed(seed)

    t_values = [round(0.1 * k, 1) for k in range(11)]
    if out_file is None:
        out_dir = os.path.join(SIM_DIR, f"{case}_results")
        os.makedirs(out_dir, exist_ok=True)
        out_file = os.path.join(out_dir, "monte_carlo.csv")

    # Rows: mean real, mean imag, stderr real, stderr imag
    rows = [[0.0] * len(t_values) for _ in range(4)]
    with open(out_file, "w", newline="") as fh:
        csv.writer(fh).writerows(rows)

    for it, t in enumerate(t_values):
        start = time.perf_counter()
        n_done = 0
        s_re = s_im = ss_re = ss_im = 0.0
        while n_done < num_samples:
            n = min(chunk_size, num_samples - n_done)
            H = run_chunk(cfg, module, phi, psi, f, 1.0 + 0.0j, t, LAMBDA, n, device)
            hr = H.real
            hi = H.imag
            s_re += hr.sum().item()
            s_im += hi.sum().item()
            ss_re += (hr * hr).sum().item()
            ss_im += (hi * hi).sum().item()
            n_done += n
        mean_re = s_re / num_samples
        mean_im = s_im / num_samples
        var_re = max(ss_re / num_samples - mean_re ** 2, 0.0)
        var_im = max(ss_im / num_samples - mean_im ** 2, 0.0)
        se_re = math.sqrt(var_re / num_samples)
        se_im = math.sqrt(var_im / num_samples)
        elapsed = time.perf_counter() - start
        print(f"t={t:.1f}, Real part: {mean_re:.6f} (SE {se_re:.6f}), "
              f"Imaginary part: {mean_im:.6f} (SE {se_im:.6f}), Time taken: {elapsed:.3f}s",
              flush=True)

        rows[0][it] = str(mean_re)
        rows[1][it] = str(mean_im)
        rows[2][it] = str(se_re)
        rows[3][it] = str(se_im)
        with open(out_file, "w", newline="") as fh:
            csv.writer(fh).writerows(rows)

    print(f"All results saved to {out_file}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("case", choices=sorted(CASES.keys()))
    parser.add_argument("--num-samples", type=int, default=10_000_000)
    parser.add_argument("--chunk-size", type=int, default=1_000_000)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--out", default=None, help="override output CSV path")
    args = parser.parse_args()
    print(f"case={args.case}, num_samples={args.num_samples}, "
          f"chunk_size={args.chunk_size}, device={args.device}", flush=True)
    run_case(args.case, args.num_samples, args.chunk_size, args.device, args.seed,
             out_file=args.out)
