# Simulation 01

For Simulation 01, the wave equation is:

$u_{tt} - a^2 \Delta u = -\frac{4w^2}{3}\sin(u)$

We let $u(z,0) = \phi(z)$ and $u_t(z,0)=\psi(z)$. The following table gives a list of experiments we have conducted:

| Case | $\phi(z)$ or $\phi(z_1,z_2)$ or $\phi(z_1,z_2,z_3)$ | $\psi(z)$ or $\psi(z_1,z_2)$ or $\psi(z_1,z_2,z_3)$ | Analytical Solution | $a$ | Initial Point(s) |
|------|------|------|------|------|------|
| $d=1$ (real) | $4\arctan\Big(\exp\big(\frac{4w}{3}z\big)\Big)$ | $\frac{8w\exp\big(\frac{4w}{3}z\big)}{3\big(1+\exp\big(\frac{8w}{3}z\big)\big)}$ | $u(z,t) = 4\arctan\Big(\exp\big(\frac{4w}{3}\big(z+\frac{t}{2}\big)\big)\Big)$ | $1$ | $1$ |
| $d=2$ (real) | $4\arctan\Big(\exp\Big(\frac{4w}{3}\big(z_1+z_2\big)\Big)\Big)$ | $\frac{8w\exp\big(\frac{4w}{3}(z_1+z_2)\big)}{3\big(1+\exp\big(\frac{8w}{3}(z_1+z_2)\big)\big)}$ | $u(z_1,z_2,t) = 4\arctan\Big(\exp\Big(\frac{4w}{3}\big(z_1+z_2+\frac{t}{2}\big)\Big)\Big)$ | $\frac{1}{\sqrt{2}}$ | $(1,1)$ |
| $d=3$ (real) | $4\arctan\Big(\exp\Big(\frac{4w}{3}\big(z_1+z_2+z_3\big)\Big)\Big)$ | $\frac{8w\exp\big(\frac{4w}{3}(z_1+z_2+z_3)\big)}{3\big(1+\exp\big(\frac{8w}{3}(z_1+z_2+z_3)\big)\big)}$ | $u(z_1,z_2,z_3,t) = 4\arctan\Big(\exp\Big(\frac{4w}{3}\big(z_1+z_2+z_3+\frac{t}{2}\big)\Big)\Big)$ | $\frac{1}{\sqrt{3}}$ | $(1,1,1)$ |
| $d=1$ (complex) | $4\arctan\Big(\exp\big(\frac{4w}{3}(iz)\big)\Big)$ | $\frac{8w\exp\big(\frac{4w}{3}(iz)\big)}{3\big(1+\exp\big(\frac{8w}{3}(iz)\big)\big)}$ | $u(z,t) = 4\arctan\Big(\exp\big(\frac{4w}{3}\big(iz+\frac{t}{2}\big)\big)\Big)$ | $i$ | $i$ |
| $d=2$ (complex) | $4\arctan\Big(\exp\Big(\frac{4w}{3}\big(iz_1+iz_2\big)\Big)\Big)$ | $\frac{8w\exp\big(\frac{4w}{3}(iz_1+iz_2)\big)}{3\big(1+\exp\big(\frac{8w}{3}(iz_1+iz_2)\big)\big)}$ | $u(z_1,z_2,t) = 4\arctan\Big(\exp\Big(\frac{4w}{3}\big(iz_1+iz_2+\frac{t}{2}\big)\Big)\Big)$ | $\frac{i}{\sqrt{2}}$ | $(i,i)$ |
| $d=3$ (complex) | $4\arctan\Big(\exp\Big(\frac{4w}{3}\big(iz_1+iz_2+iz_3\big)\Big)\Big)$ | $\frac{8w\exp\big(\frac{4w}{3}(iz_1+iz_2+iz_3)\big)}{3\big(1+\exp\big(\frac{8w}{3}(iz_1+iz_2+iz_3)\big)\big)}$ | $u(z_1,z_2,z_3,t) = 4\arctan\Big(\exp\Big(\frac{4w}{3}\big(iz_1+iz_2+iz_3+\frac{t}{2}\big)\Big)\Big)$ | $\frac{i}{\sqrt{3}}$ | $(i,i,i)$ |



In all cases, we set $w = 0.5 + 0.0i$, $\lambda = 1.0$, and sample $t \in [0,1.0]$ in steps of $0.1$ (11 time points).




