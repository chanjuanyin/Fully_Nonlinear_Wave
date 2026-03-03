# Simulation 01

For Simulation 01, the wave equation is:

$\frac{\partial^2 u}{\partial t^2}(z,t)-(1)^2\Big[\frac{\partial^2 u}{\partial z^2}(z,t)\Big] = -\frac{1}{3}\sin(u(z,t))$

$u(z,0) = \phi(z) = 4\arctan\big(\exp\big(\frac{2}{3}z\big)\big)$

$\frac{\partial u}{\partial t}(z,0) = \psi(z) = \frac{4\exp\big(\frac{2}{3}z\big)}{3\big(1+\exp\big(\frac{4}{3}z\big)\big)}$

The analytical solution is:

$u(z,t) = 4\arctan\Big(\exp\big(\frac{2}{3}\big(z+\frac{t}{2}\big)\big)\Big)$

We set $a=1+0i$, $z=1+0i$, $\lambda = 1$, and sample $t \in [0,1.0]$ in steps of $0.1$ (11 time points).
