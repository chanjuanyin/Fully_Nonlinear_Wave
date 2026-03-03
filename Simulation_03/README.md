# Simulation 03

For Simulation 03, the wave equation is:

$\frac{\partial^2 u}{\partial t^2}(z_1,z_2,z_3,t)-(1)^2\Big[\frac{\partial^2 u}{\partial z_1^2}(z_1,z_2,z_3,t)+\frac{\partial^2 u}{\partial z_2^2}(z_1,z_2,z_3,t)+\frac{\partial^2 u}{\partial z_3^2}(z_1,z_2,z_3,t)\Big] = -\frac{11}{9}\sin(u(z_1,z_2,z_3,t))$

$u(z_1,z_2,z_3,0) = \phi(z_1,z_2,z_3) = 4\arctan\Big(\exp\Big(\frac{2}{3}\big(z_1+z_2+z_3\big)\Big)\Big)$

$\frac{\partial u}{\partial t}(z_1,z_2,z_3,0) = \psi(z_1,z_2,z_3) = \frac{4\exp\big(\frac{2}{3}(z_1+z_2+z_3)\big)}{3\big(1+\exp\big(\frac{4}{3}(z_1+z_2+z_3)\big)\big)}$

The analytical solution is:

$u(z_1,z_2,z_3,t) = 4\arctan\Big(\exp\Big(\frac{2}{3}\big(z_1+z_2+z_3+\frac{t}{2}\big)\Big)\Big)$

We set $a=1+0i$, $z=(1+0i,1+0i,1+0i)$, $\lambda = 1$, and sample $t \in [0,1.0]$ in steps of $0.1$ (11 time points).
