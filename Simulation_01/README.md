# Simulation 01 — Branching Monte Carlo for a fully nonlinear wave equation

This folder contains six numerical experiments for the stochastic branching
representation of the nonlinear wave equation

$$u_{tt}(z,t) - a^2 \Delta u(z,t) = f(u) = -\frac{4\omega^2}{3}\sin\big(u(z,t)\big),
\qquad u(z,0)=\phi(z),\quad u_t(z,0)=\psi(z),$$

with $z \in \mathbb{C}^d$, $d \in \{1,2,3\}$, and $\omega = 0.5$, so that
$k := \tfrac{4\omega}{3} = \tfrac{2}{3}$. The Monte Carlo estimator is the
branching-tree functional $\mathcal{H}$ of the paper (§4–§5 and Appendix A.1):
codes are attached to tree nodes, offspring are drawn from the code algebra
$\mathcal{M}(c)$ with probabilities $q_c$, interior nodes contribute weights
$e^{\lambda\tau}\tau\gamma_1/(\lambda q_c(\gamma))$, and boundary nodes contribute
the initial-data weights $W^{\rm b}$ built from $\tilde\phi(c)$, $\tilde\psi(c)$.
The empirical mean of i.i.d. copies of $\mathcal{H}_{\rm Id}(z,t)$ converges to
$u(z,t)$.

Throughout: branching rate $\lambda = 1$, evaluation point $z = (1,\dots,1)$,
times $t \in \{0, 0.1, \dots, 1.0\}$.

## The six cases and their analytical solutions

All six use sine–Gordon-type kink solutions. Write $s = z_1 + \cdots + z_d$.

| Case | $\phi$ | $\psi$ | analytical solution $u$ | $a$ |
|------|--------|--------|--------------------------|-----|
| real $d{=}1$ | $4\arctan\!\big(e^{k z}\big)$ | $\dfrac{2k\,e^{k z}}{1+e^{2k z}}$ | $4\arctan\!\big(e^{k(z+t/2)}\big)$ | $1$ |
| real $d{=}2$ | $4\arctan\!\big(e^{k s}\big)$ | $\dfrac{2k\,e^{k s}}{1+e^{2k s}}$ | $4\arctan\!\big(e^{k(s+t/2)}\big)$ | $\tfrac{1}{\sqrt2}$ |
| real $d{=}3$ | $4\arctan\!\big(e^{k s}\big)$ | $\dfrac{2k\,e^{k s}}{1+e^{2k s}}$ | $4\arctan\!\big(e^{k(s+t/2)}\big)$ | $\tfrac{1}{\sqrt3}$ |
| complex $d{=}1$ | $4\arctan\!\big(e^{ik z}\big)$ | $\dfrac{2k\,e^{ik z}}{1+e^{2ik z}}$ | $4\arctan\!\big(e^{k(iz+t/2)}\big)$ | $i$ |
| complex $d{=}2$ | $4\arctan\!\big(e^{ik s}\big)$ | $\dfrac{2k\,e^{ik s}}{1+e^{2ik s}}$ | $4\arctan\!\big(e^{k(is+t/2)}\big)$ | $\tfrac{i}{\sqrt2}$ |
| complex $d{=}3$ | $4\arctan\!\big(e^{ik s}\big)$ | $\dfrac{2k\,e^{ik s}}{1+e^{2ik s}}$ | $4\arctan\!\big(e^{k(is+t/2)}\big)$ | $\tfrac{i}{\sqrt3}$ |

Note $2k = \tfrac{8\omega}{3}$, matching the $\psi$ used in the code.

**Why these solve the PDE.** The profile $w(\theta) = 4\arctan(e^\theta)$
satisfies $w'' = \sin w$. For $u = w\big(k(cs + t/2)\big)$ with $c = 1$ (real)
or $c = i$ (complex):

$$u_{tt} = \tfrac{k^2}{4}\,w'', \qquad
\Delta u = c^2 d\, k^2\, w'',$$

so $u_{tt} - a^2\Delta u = k^2 w''\big(\tfrac14 - a^2 c^2 d\big)$. In every row
of the table $a^2 c^2 d = 1$ (e.g. real $d{=}2$: $\tfrac12 \cdot 1 \cdot 2$;
complex $d{=}3$: $-\tfrac13 \cdot (-1) \cdot 3$), hence

$$u_{tt} - a^2\Delta u = -\tfrac{3k^2}{4}\, w'' = -\tfrac{3k^2}{4}\sin u
= -\tfrac{4\omega^2}{3}\sin u = f(u). \checkmark$$

## Files

| File(s) | Purpose |
|---------|---------|
| `{real,complex}_d{1,2,3}_analytical.py` | Evaluate the closed-form solution on $t \in [0,1]$, step $0.01$ → `*_results/analytic.csv` (2 rows: Re, Im). |
| `{real,complex}_d{1,2,3}_monte_carlo.py` | **Reference implementation**: recursive, one tree per sample, mirrors Algorithm A.1 line by line. Also hosts the shared building blocks `tilde_phi`, `tilde_psi`, `QC`, `nth_derivative_scalar`, `mixed_partial_orders`. Runs on CPU with $10^5$ samples. |
| `vectorized_monte_carlo.py` | **Production engine**: batched GPU implementation of the same algorithm, ~2000× faster per sample. Produced the committed $10^7$-sample results. |
| `Plot_history.ipynb` | Loads the CSVs and plots Monte Carlo vs analytical for all six cases. |
| `*_results/monte_carlo.csv` | 4 rows × 11 columns ($t=0,\dots,1$): mean Re, mean Im, standard error Re, standard error Im. |

## Computational considerations: the two engines

There are two implementations of the same algorithm, with different roles:

| | Recursive scripts (6 files) | `vectorized_monte_carlo.py` |
|---|---|---|
| How it works | One tree at a time, Python recursion | Millions of trees at once, tensor ops |
| Speed | ≈ 0.7 ms/sample (CPU) | ≈ 0.4 µs/sample (GPU) |
| $10^7$ samples would take | ≈ 200 hours ❌ | ≈ 1.5 h for all six cases ✅ |
| Configured for | $10^5$ samples | $10^7$ samples |
| Role | **reference** — readable, matches Algorithm A.1 line by line | **production** — produced the committed CSVs |

The $10^7$ numbers in `monte_carlo.csv` come from the vectorized engine. The
recursive drivers are kept at $10^5$ because that is the largest run that
finishes in reasonable time (≈ 2 h for all six), and that run is what
cross-validates the fast engine: both implementations agree within one
standard error at every time point (and both reproduce $u(z,0)=\phi(z)$
exactly at $t=0$). That agreement is what certifies the fast engine computes
the same quantity as the reference code.

**Why the recursion is slow — and why it prefers the CPU.** Each tree has only
~5–10 nodes, and every node performs a handful of torch operations on a
*single complex scalar*. Each op costs a fixed overhead (Python dispatch, and
on the GPU a kernel launch) that dwarfs the actual arithmetic:
$10^7$ trees × ~10 nodes × dozens of ops = billions of overhead-dominated
calls. Measured: ≈ 0.7 ms/sample on CPU vs ≈ 2.9 ms/sample on GPU — the GPU is
**4× slower** here because single-scalar kernels are pure launch overhead.
The recursive drivers therefore pin `device = "cpu"`.

**How the vectorized engine restructures the work.** The same overhead is paid
once per *million* nodes instead of once per node, by going breadth-first over
a whole batch of trees. Three ideas:

1. **The frontier.** Instead of "finish tree 1, then tree 2, …", keep one flat
   tensor of all *currently alive* nodes across all samples: positions $z$,
   remaining times $t_{\rm rem}$, code vectors, and the owning sample index.
   One loop iteration processes an entire generation: draw all lifetimes
   $\tau$ and spatial marks at once, split into boundary ($\tau \ge t_{\rm rem}$)
   and interior ($\tau < t_{\rm rem}$) nodes, sample all offspring at once
   (acceptance–rejection runs on masked sub-arrays until every node accepts),
   and assemble the next generation's tensors. ~30 iterations empty the
   frontier for the whole batch.
2. **Products as sums of logs.** $\mathcal{H}$ is a *product* of per-node
   factors — $e^{\lambda\tau}\tau\gamma_1/(\lambda q_c)$ for interior nodes,
   $e^{\lambda t}W^{\rm b}$ for boundary nodes — but one sample's nodes surface
   at different loop iterations, interleaved with every other sample's.
   Multiplying into "sample $m$'s slot" with duplicate indices is not a safe
   tensor operation, but *adding* is (`index_add_`). So each factor is stored
   as its complex logarithm and summed into its sample's accumulator; at the
   end $\mathcal{H} = e^{\sum\log(\text{factors})}$. This is exact for any
   branch choice of the complex log, since
   $e^{\log z_1 + \log z_2} = z_1 z_2$ regardless of which branch each log
   lands on (negative real factors such as $-a^2(\cdots)$ included).
3. **Boundary evaluation grouped by code.** The expensive part is
   $\tilde\phi(c), \tilde\psi(c)$ — nested autograd derivatives. Two boundary
   nodes carrying the *same code* need the *same* derivative computation, only
   at different evaluation points. So all boundary nodes of a chunk are
   buffered during the sweep, grouped by code vector at the end, and each
   distinct code triggers **one** batched autograd call over all its nodes'
   points — e.g. $5\times10^5$ evaluations of
   $\tfrac{1}{2!}\partial^2 f^{(3)}(\phi)$ in a single graph. Crucially these
   calls go through the *same* `tilde_phi` / `tilde_psi` imported from the
   recursive modules (autograd is batch-agnostic for elementwise functions),
   so both engines share one source of truth for the initial-data machinery —
   any change there automatically applies to both.

Same algorithm, same randomness structure, same boundary functions — just
reorganized from "depth-first, one tree" to "breadth-first, all trees", which
is what lets the GPU actually deliver $10^7$ samples.

**Regenerating results.** The high-precision CSVs are produced by

```
python vectorized_monte_carlo.py real_d1 --num-samples 10000000 --chunk-size 1000000
```

(cases: `real_d1`, `real_d2`, `real_d3`, `complex_d1`, `complex_d2`,
`complex_d3`). ⚠ Running a recursive driver directly, e.g.
`python real_d1_monte_carlo.py`, **overwrites** the corresponding
`monte_carlo.csv` with a $10^5$-sample run — use it for cross-checking, then
regenerate with the vectorized engine.

## Results ($10^7$ samples per time point)

RMS error of the MC mean against the analytical solution over each case's
convergence window, and the largest deviation in units of the standard error:

| Case | window | RMS error | max $\|z\|$-score | verdict |
|------|--------|-----------|--------------------|---------|
| real $d{=}1$ | $t\le 1.0$ | $9\times10^{-4}$ | 1.3 | unbiased |
| real $d{=}2$ | $t\le 1.0$ | $1.0\times10^{-3}$ | 1.2 | unbiased |
| real $d{=}3$ | $t\le 1.0$ | $1.8\times10^{-3}$ | 1.8 | unbiased |
| complex $d{=}1$ | $t\le 1.0$ | $5.5\times10^{-4}$ | 1.3 | unbiased |
| complex $d{=}2$ | $t\le 0.4$ | $4.1\times10^{-3}$ | 1.7 | unbiased on window |
| complex $d{=}3$ | $t\le 0.7$ | $2.3\times10^{-3}$ | 1.9 | unbiased on window |

**Convergence windows.** For complex $d{=}2$ beyond $t\approx0.4$ and complex
$d{=}3$ beyond $t\approx0.7$ the empirical standard error explodes (comparable
to the mean itself, growing to $10^{11}$ by $t=1$): the estimator leaves its
variance-integrability horizon — the phenomenon controlled by the sufficient
conditions of §7 of the paper. This is a property of the estimator, not an
implementation error; `Plot_history.ipynb` therefore plots the MC curves for
these two cases only on their windows.

## Implementation pitfalls found and fixed (2026-08-13)

1. **Interior weight of the code $\tfrac{1}{\alpha!}\partial^\alpha\big((\partial_t\cdot)^2\big)$**:
   the prefactor for the first two offspring families is $(d{+}2)\prod_k(1+\alpha_k)$
   (with the $i$-family correction $\tfrac{|a|^2(2+\alpha_i)(3+\alpha_i)}{3|\gamma_1|}$),
   per Appendix A.1 — it was doubled.
2. **Missing $a^2$** on the $\sum_i \partial^{\alpha+2\mathbf{1}_i}$ terms in
   $\tilde\phi(\partial^\alpha\partial_{tt})$, $\tilde\psi(\partial^\alpha\partial_t)$
   and $\tilde\psi(\partial^\alpha\partial_{tt})$ (harmless only when $a^2=1$).
3. **$d{=}3$ drivers** used $\psi = \tfrac83(\cdots)$ instead of
   $\tfrac{8\omega}{3}(\cdots)$, doubling the initial velocity.
4. **PyTorch complex autograd returns the *conjugate* Wirtinger derivative**:
   for holomorphic $f$, `torch.autograd.grad` yields $\overline{f'(z)}$, not
   $f'(z)$. Every derivative in `nth_derivative_scalar` /
   `mixed_partial_orders` is therefore followed by `.conj()`. On the real axis
   derivatives are real and the conjugation is invisible — which is exactly why
   only the complex cases' imaginary parts were biased before the fix
   (a smooth $t$-dependent offset that did **not** shrink from $10^5$ to $10^7$
   samples). *Any future complex-analytic autograd code in this project needs
   the same treatment.*
