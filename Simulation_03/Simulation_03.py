import torch
import copy

def nth_derivative_scalar(f, u, order):
    """Compute f^{(j)}(u)
    Compute the order-th derivative of a scalar->scalar function f at u.
    u must require grad; result keeps graph so it can be differentiated w.r.t inputs of u."""
    y = f(u)
    for k in range(order):
        (y,) = torch.autograd.grad(y, u, create_graph=True, retain_graph=k < order - 1)
    return y

def mixed_partial_orders(g, inputs, orders):
    """Compute \partial_{z_1}^{\alpha_1} \partial_{z_2}^{\alpha_2} of some functions."""
    # For example inputs = (x, y, z), so inputs[0] = x, inputs[1] = y, inputs[2] = z
    # orders: list of (var_index, order), e.g. [(0, 2), (1, 3), (2, 4)] means d^2/dx^2, d^3/dy^3, d^4/dz^4
    y = g(*inputs)
    steps = []
    for idx, k in orders:
        steps.extend([idx] * k)  # expand into sequential steps
    last = len(steps) - 1
    for j, idx in enumerate(steps):
        (y,) = torch.autograd.grad(
            y, inputs[idx],
            create_graph= j < last, # only create graph if we need to do more steps
            retain_graph= j < last  # only retain graph if we need to do more steps
        )
    return y

# code is of the form [int, int, int, int, int]
# [1, \alpha_1, \alpha_2, \alpha_3, -1] means \frac{1}{\alpha_1! \alpha_2! \alpha_3!} \partial_{z_1}^{\alpha_1} \partial_{z_2}^{\alpha_2} \partial_{z_2}^{\alpha_3}
# [2, \alpha_1, \alpha_2, \alpha_3, j] means \frac{1}{\alpha_1! \alpha_2! \alpha_3!} \partial_{z_1}^{\alpha_1} \partial_{z_2}^{\alpha_2} \partial_{z_2}^{\alpha_3} f^{(j)}
# [3, \alpha_1, \alpha_2, \alpha_3, -1] means \frac{1}{\alpha_1! \alpha_2! \alpha_3!} \partial_{z_1}^{\alpha_1} \partial_{z_2}^{\alpha_2} \partial_{z_2}^{\alpha_3} ((\partial_t(\cdot))^2)
# [4, \alpha_1, \alpha_2, \alpha_3, -1] means \frac{1}{\alpha_1! \alpha_2! \alpha_3!} \partial_{z_1}^{\alpha_1} \partial_{z_2}^{\alpha_2} \partial_{z_2}^{\alpha_3} \partial_t
# [5, \alpha_1, \alpha_2, \alpha_3, -1] means \frac{1}{\alpha_1! \alpha_2! \alpha_3!} \partial_{z_1}^{\alpha_1} \partial_{z_2}^{\alpha_2} \partial_{z_2}^{\alpha_3} \partial_{tt}

def tilde_phi(code, phi, psi, f, z1, z2, z_3):
    form = code[0]
    alpha_1 = code[1]
    alpha_2 = code[2]
    alpha_3 = code[3]
    j = code[4]
    if form == 1:
        return 1. / (torch.factorial(torch.tensor(alpha_1)) * torch.factorial(torch.tensor(alpha_2)) * torch.factorial(torch.tensor(alpha_3))) * \
            mixed_partial_orders(g = phi, inputs = (z1, z2, z_3), orders = [(0, alpha_1), (1, alpha_2), (2, alpha_3)])
    elif form == 2:
        return 1. / (torch.factorial(torch.tensor(alpha_1)) * torch.factorial(torch.tensor(alpha_2)) * torch.factorial(torch.tensor(alpha_3))) * \
            mixed_partial_orders(g = lambda x_in, y_in, z_in: nth_derivative_scalar(f, phi(x_in, y_in, z_in), j), 
                                 inputs = (z1, z2, z_3), 
                                 orders = [(0, alpha_1), (1, alpha_2), (2, alpha_3)])
    elif form == 3:
        return 1. / (torch.factorial(torch.tensor(alpha_1)) * torch.factorial(torch.tensor(alpha_2)) * torch.factorial(torch.tensor(alpha_3))) * \
            mixed_partial_orders(g = lambda x_in, y_in, z_in: psi(x_in, y_in, z_in)**2,
                                 inputs = (z1, z2, z_3), 
                                 orders = [(0, alpha_1), (1, alpha_2), (2, alpha_3)])
    elif form == 4:
        return 1. / (torch.factorial(torch.tensor(alpha_1)) * torch.factorial(torch.tensor(alpha_2)) * torch.factorial(torch.tensor(alpha_3))) * \
            mixed_partial_orders(g = lambda x_in, y_in, z_in: psi(x_in, y_in, z_in),
                                 inputs = (z1, z2, z_3), 
                                 orders = [(0, alpha_1), (1, alpha_2), (2, alpha_3)])
    elif form == 5:
        return 1. / (torch.factorial(torch.tensor(alpha_1)) * torch.factorial(torch.tensor(alpha_2)) * torch.factorial(torch.tensor(alpha_3))) * \
            (
                mixed_partial_orders(g = phi, inputs=(z1, z2, z_3), orders=[(0, alpha_1 + 2), (1, alpha_2), (2, alpha_3)]) + \
                mixed_partial_orders(g = phi, inputs=(z1, z2, z_3), orders=[(0, alpha_1), (1, alpha_2 + 2), (2, alpha_3)]) + \
                mixed_partial_orders(g = phi, inputs=(z1, z2, z_3), orders=[(0, alpha_1), (1, alpha_2), (2, alpha_3 + 2)]) + \
                mixed_partial_orders(g = lambda x_in, y_in, z_in: f(phi(x_in, y_in, z_in)), inputs=(z1, z2, z_3), orders=[(0, alpha_1), (1, alpha_2), (2, alpha_3)])
            )

def gradient_tilde_phi(coordinate, code, phi, psi, f, z1, z2, z_3):
    if coordinate=='z1':
        new_code = copy.deepcopy(code)
        new_code[1] += 1 # increase alpha_1 by 1
        return tilde_phi(new_code, phi, psi, f, z1, z2, z_3)
    elif coordinate=='z2':
        new_code = copy.deepcopy(code)
        new_code[2] += 1 # increase alpha_2 by 1
        return tilde_phi(new_code, phi, psi, f, z1, z2, z_3)
    elif coordinate=='z3':
        new_code = copy.deepcopy(code)
        new_code[3] += 1 # increase alpha_3 by 1
        return tilde_phi(new_code, phi, psi, f, z1, z2, z_3)

def tilde_psi(code, phi, psi, f, z1, z2, z_3, a):
    form = code[0]
    alpha_1 = code[1]
    alpha_2 = code[2]
    alpha_3 = code[3]
    j = code[4]
    if form == 1:
        return 1. / (torch.factorial(torch.tensor(alpha_1)) * torch.factorial(torch.tensor(alpha_2)) * torch.factorial(torch.tensor(alpha_3))) * \
            mixed_partial_orders(g = psi, inputs = (z1, z2, z_3), orders = [(0, alpha_1), (1, alpha_2), (2, alpha_3)])
    elif form == 2:
        return 1. / (torch.factorial(torch.tensor(alpha_1)) * torch.factorial(torch.tensor(alpha_2)) * torch.factorial(torch.tensor(alpha_3))) * \
            mixed_partial_orders(g = lambda x_in, y_in, z_in: psi(x_in, y_in, z_in) * nth_derivative_scalar(f, phi(x_in, y_in, z_in), j+1), 
                                 inputs = (z1, z2, z_3), 
                                 orders = [(0, alpha_1), (1, alpha_2), (2, alpha_3)])
    elif form == 3:
        return 2. / (torch.factorial(torch.tensor(alpha_1)) * torch.factorial(torch.tensor(alpha_2)) * torch.factorial(torch.tensor(alpha_3))) * \
            mixed_partial_orders(g = lambda x_in, y_in, z_in: a**2 * psi(x_in, y_in, z_in) * (
                                                                mixed_partial_orders(g=phi,
                                                                                     inputs=(x_in, y_in, z_in), 
                                                                                     orders=[(0, 2), (1, 0), (2, 0)]) + \
                                                                mixed_partial_orders(g=phi,
                                                                                     inputs=(x_in, y_in, z_in), 
                                                                                     orders=[(0, 0), (1, 2), (2, 0)]) + \
                                                                mixed_partial_orders(g=phi,
                                                                                     inputs=(x_in, y_in, z_in), 
                                                                                     orders=[(0, 0), (1, 0), (2, 2)])
                                                                )
                                                        + psi(x_in, y_in, z_in) * f(phi(x_in, y_in, z_in)),
                                 inputs = (z1, z2, z_3),
                                 orders = [(0, alpha_1), (1, alpha_2), (2, alpha_3)])
    elif form == 4:
        return 1. / (torch.factorial(torch.tensor(alpha_1)) * torch.factorial(torch.tensor(alpha_2)) * torch.factorial(torch.tensor(alpha_3))) * \
            (
                mixed_partial_orders(g = phi, inputs=(z1, z2, z_3), orders=[(0, alpha_1 + 2), (1, alpha_2), (2, alpha_3)]) + \
                mixed_partial_orders(g = phi, inputs=(z1, z2, z_3), orders=[(0, alpha_1), (1, alpha_2 + 2), (2, alpha_3)]) + \
                mixed_partial_orders(g = phi, inputs=(z1, z2, z_3), orders=[(0, alpha_1), (1, alpha_2), (2, alpha_3 + 2)]) + \
                mixed_partial_orders(g = lambda x_in, y_in, z_in: f(phi(x_in, y_in, z_in)), inputs=(z1, z2, z_3), orders=[(0, alpha_1), (1, alpha_2), (2, alpha_3)])
            )
    elif form == 5:
        return 1. / (torch.factorial(torch.tensor(alpha_1)) * torch.factorial(torch.tensor(alpha_2)) * torch.factorial(torch.tensor(alpha_3))) * \
            (
                mixed_partial_orders(g = psi, inputs=(z1, z2, z_3), orders=[(0, alpha_1 + 2), (1, alpha_2), (2, alpha_3)]) + \
                mixed_partial_orders(g = psi, inputs=(z1, z2, z_3), orders=[(0, alpha_1), (1, alpha_2 + 2), (2, alpha_3)]) + \
                mixed_partial_orders(g = psi, inputs=(z1, z2, z_3), orders=[(0, alpha_1), (1, alpha_2), (2, alpha_3 + 2)]) + \
                mixed_partial_orders(g = lambda x_in, y_in, z_in: psi(x_in, y_in, z_in) * nth_derivative_scalar(f, phi(x_in, y_in, z_in), 1), 
                                     inputs=(z1, z2, z_3), 
                                     orders=[(0, alpha_1), (1, alpha_2), (2, alpha_3)]
                                     )
            )

def QC(code, a):
    """Probabilistically generate the next code based on the code algebra"""
    form = code[0]
    alpha_1 = code[1]
    alpha_2 = code[2]
    alpha_3 = code[3]
    j = code[4]
    if form == 1:
        if alpha_1==0 and alpha_2==0 and alpha_3==0:
            return 1, [2, 0, 0, 0, 0], None, None # Return \{(1, f, \emptyset)\}
        elif alpha_1 > 0: # i which is the first index such that \alpha_i\neq 0 is 1
            while True: # accept-reject sampling
                beta_1 = torch.randint(0, alpha_1, (1,)).item() # uniform random integer in [0, alpha_1-1] inclusive
                beta_2 = torch.randint(0, alpha_2 + 1, (1,)).item() # uniform random integer in [0, alpha_2] inclusive
                beta_3 = torch.randint(0, alpha_3 + 1, (1,)).item() # uniform random integer in [0, alpha_3] inclusive
                gamma_1 = (alpha_1 - beta_1) / alpha_1
                if torch.rand(1).item() <= torch.abs(gamma_1):
                    break
            return gamma_1, [2, beta_1, beta_2, beta_3, 1], [1, alpha_1 - beta_1, alpha_2 - beta_2, alpha_3 - beta_3, -1], None # Return \{( gamma_1, \frac{1}{\beta!} \partial^{\beta} \circ f^{(1)}, \frac{1}{(alpha_1-beta_1)!} \partial^{\alpha-\beta} )\}
        elif alpha_2 > 0: # i which is the first index such that \alpha_i\neq 0 is 2
            while True: # accept-reject sampling
                beta_1 = torch.randint(0, alpha_1 + 1, (1,)).item() # uniform random integer in [0, alpha_1] inclusive
                beta_2 = torch.randint(0, alpha_2, (1,)).item() # uniform random integer in [0, alpha_2-1] inclusive
                beta_3 = torch.randint(0, alpha_3 + 1, (1,)).item() # uniform random integer in [0, alpha_3] inclusive
                gamma_1 = (alpha_2 - beta_2) / alpha_2
                if torch.rand(1).item() <= torch.abs(gamma_1):
                    break
            return gamma_1, [2, beta_1, beta_2, beta_3, 1], [1, alpha_1 - beta_1, alpha_2 - beta_2, alpha_3 - beta_3, -1], None # Return \{( gamma_2, \frac{1}{\beta!} \partial^{\beta} \circ f^{(1)}, \frac{1}{(alpha_2-beta_2)!} \partial^{\alpha-\beta} )\}
        elif alpha_3 > 0: # i which is the first index such that \alpha_i\neq 0 is 3
            while True: # accept-reject sampling
                beta_1 = torch.randint(0, alpha_1 + 1, (1,)).item() # uniform random integer in [0, alpha_1] inclusive
                beta_2 = torch.randint(0, alpha_2 + 1, (1,)).item() # uniform random integer in [0, alpha_2] inclusive
                beta_3 = torch.randint(0, alpha_3, (1,)).item() # uniform random integer in [0, alpha_3-1] inclusive
                gamma_1 = (alpha_3 - beta_3) / alpha_3
                if torch.rand(1).item() <= torch.abs(gamma_1):
                    break
            return gamma_1, [2, beta_1, beta_2, beta_3, 1], [1, alpha_1 - beta_1, alpha_2 - beta_2, alpha_3 - beta_3, -1], None # Return \{( gamma_2, \frac{1}{\beta!} \partial^{\beta} \circ f^{(1)}, \frac{1}{(alpha_2-beta_2)!} \partial^{\alpha-\beta} )\}
    elif form == 2:
        m = torch.randint(1, 6, (1,)).item() # uniform random integer in [1, 5] inclusive
        beta_1 = torch.randint(0, alpha_1 + 1, (1,)).item() # uniform random integer in [0, alpha_1] inclusive
        beta_2 = torch.randint(0, alpha_2 + 1, (1,)).item() # uniform random integer in [0, alpha_2] inclusive
        beta_3 = torch.randint(0, alpha_3 + 1, (1,)).item() # uniform random integer in [0, alpha_3] inclusive
        if m==1:
            return 1, [2, beta_1, beta_2, beta_3, j+2], [3, alpha_1 - beta_1, alpha_2 - beta_2, alpha_3 - beta_3, -1], None # Return \{(1, \frac{1}{\beta!} \partial^{\beta} \circ f^{(j+2)}, \frac{1}{(\alpha-\beta)!} \partial^{\alpha-\beta} ((\partial_t(\cdot))^2) )\}
        elif m==2:
            return 1, [2, beta_1, beta_2, beta_3, j+1], [2, alpha_1 - beta_1, alpha_2 - beta_2, alpha_3 - beta_3, 0], None # Return \{(1, \frac{1}{\beta!} \partial^{\beta} \circ f^{(j+1)}, \frac{1}{(\alpha-\beta)!} \partial^{\alpha-\beta} f )\}
        elif m==3:
            i = 1 # coordinate in concern will be z_1
            while True: # accept-reject sampling
                beta_1 = torch.randint(0, alpha_1 + 1, (1,)).item() # uniform random integer in [0, alpha_1] inclusive
                beta_2 = torch.randint(0, alpha_2 + 1, (1,)).item() # uniform random integer in [0, alpha_2] inclusive
                beta_3 = torch.randint(0, alpha_3 + 1, (1,)).item() # uniform random integer in [0, alpha_3] inclusive
                L = ( 4 * (alpha_1 - beta_1 + 1) * (beta_1 + 1) ) / ((2+alpha_1)**2)
                if torch.rand(1).item() <= torch.abs(L):
                    break
            return -a**2 * (beta_1+1) * (alpha_1-beta_1+1), [2, beta_1+1, beta_2, beta_3, j+1], [1, alpha_1-beta_1+1, alpha_2-beta_2, alpha_3-beta_3, -1], i # Return \{( -a^2 * (beta_1+1) * (alpha_1-beta_1+1), \frac{1}{(\beta+e_i)!} \partial^{\beta+e_i} \circ f^{(j+1)}, \frac{1}{(\alpha-\beta+e_i)!} \partial^{\alpha-\beta+e_i} )\}
        elif m==4:
            i = 2 # coordinate in concern will be z_2
            while True: # accept-reject sampling
                beta_1 = torch.randint(0, alpha_1 + 1, (1,)).item() # uniform random integer in [0, alpha_1] inclusive
                beta_2 = torch.randint(0, alpha_2 + 1, (1,)).item() # uniform random integer in [0, alpha_2] inclusive
                beta_3 = torch.randint(0, alpha_3 + 1, (1,)).item() # uniform random integer in [0, alpha_3] inclusive
                L = ( 4 * (alpha_2 - beta_2 + 1) * (beta_2 + 1) ) / ((2+alpha_2)**2)
                if torch.rand(1).item() <= torch.abs(L):
                    break
            return -a**2 * (beta_2+1) * (alpha_2-beta_2+1), [2, beta_1, beta_2+1, beta_3, j+1], [1, alpha_1-beta_1, alpha_2-beta_2+1, alpha_3-beta_3, -1], i # Return \{( -a^2 * (beta_2+1) * (alpha_2-beta_2+1), \frac{1}{(\beta+e_i)!} \partial^{\beta+e_i} \circ f^{(j+1)}, \frac{1}{(\alpha-\beta+e_i)!} \partial^{\alpha-\beta+e_i} )\}
        elif m==5:
            i = 3 # coordinate in concern will be z_3
            while True: # accept-reject sampling
                beta_1 = torch.randint(0, alpha_1 + 1, (1,)).item() # uniform random integer in [0, alpha_1] inclusive
                beta_2 = torch.randint(0, alpha_2 + 1, (1,)).item() # uniform random integer in [0, alpha_2] inclusive
                beta_3 = torch.randint(0, alpha_3 + 1, (1,)).item() # uniform random integer in [0, alpha_3] inclusive
                L = ( 4 * (alpha_3 - beta_3 + 1) * (beta_3 + 1) ) / ((2+alpha_3)**2)
                if torch.rand(1).item() <= torch.abs(L):
                    break
            return -a**2 * (beta_3+1) * (alpha_3-beta_3+1), [2, beta_1, beta_2, beta_3+1, j+1], [1, alpha_1-beta_1, alpha_2-beta_2, alpha_3-beta_3+1, -1], i # Return \{( -a^2 * (beta_3+1) * (alpha_3-beta_3+1), \frac{1}{(\beta+e_i)!} \partial^{\beta+e_i} \circ f^{(j+1)}, \frac{1}{(\alpha-\beta+e_i)!} \partial^{\alpha-\beta+e_i} )\}
    elif form == 3:
        m = torch.randint(1, 6, (1,)).item() # uniform random integer in [1, 5] inclusive
        beta_1 = torch.randint(0, alpha_1 + 1, (1,)).item() # uniform random integer in [0, alpha_1] inclusive
        beta_2 = torch.randint(0, alpha_2 + 1, (1,)).item() # uniform random integer in [0, alpha_2] inclusive
        beta_3 = torch.randint(0, alpha_3 + 1, (1,)).item() # uniform random integer in [0, alpha_3] inclusive
        if m==1:
            return 2, [5, beta_1, beta_2, beta_3, -1], [5, alpha_1 - beta_1, alpha_2 - beta_2, alpha_3 - beta_3, -1], None # Return \{(2, \frac{1}{\beta!} \partial^{\beta} \circ \partial_{tt}, \frac{1}{(\alpha-\beta)!} \partial^{\alpha-\beta} \partial_{tt} )\}
        elif m==2:
            return 2, [3, beta_1, beta_2, beta_3, -1], [2, alpha_1 - beta_1, alpha_2 - beta_2, alpha_3 - beta_3, 1], None # Return \{(2, \frac{1}{\beta!} \partial^{\beta} \circ ((\partial_t(\cdot))^2), \frac{1}{(\alpha-\beta)!} \partial^{\alpha-\beta} f^{(1)} )\}
        elif m==3:
            i = 1 # coordinate in concern will be z_1
            while True: # accept-reject sampling
                beta_1 = torch.randint(0, alpha_1 + 1, (1,)).item() # uniform random integer in [0, alpha_1] inclusive
                beta_2 = torch.randint(0, alpha_2 + 1, (1,)).item() # uniform random integer in [0, alpha_2] inclusive
                beta_3 = torch.randint(0, alpha_3 + 1, (1,)).item() # uniform random integer in [0, alpha_3] inclusive
                L = ( 4 * (alpha_1 - beta_1 + 1) * (beta_1 + 1) ) / ((2+alpha_1)**2)
                if torch.rand(1).item() <= torch.abs(L):
                    break
            return -2*a**2 * (beta_1+1) * (alpha_1-beta_1+1), [4, beta_1+1, beta_2, beta_3, -1], [4, alpha_1-beta_1+1, alpha_2-beta_2, alpha_3-beta_3, -1], i # Return \{( -2*a^2 * (beta_1+1) * (alpha_1-beta_1+1), \frac{1}{(\beta+e_i)!} \partial^{\beta+e_i} \circ \partial_t, \frac{1}{(\alpha-\beta+e_i)!} \partial^{\alpha-\beta+e_i} \partial_t )\}
        elif m==4:
            i = 2 # coordinate in concern will be z_2
            while True: # accept-reject sampling
                beta_1 = torch.randint(0, alpha_1 + 1, (1,)).item() # uniform random integer in [0, alpha_1] inclusive
                beta_2 = torch.randint(0, alpha_2 + 1, (1,)).item() # uniform random integer in [0, alpha_2] inclusive
                beta_3 = torch.randint(0, alpha_3 + 1, (1,)).item() # uniform random integer in [0, alpha_3] inclusive
                L = ( 4 * (alpha_2 - beta_2 + 1) * (beta_2 + 1) ) / ((2+alpha_2)**2)
                if torch.rand(1).item() <= torch.abs(L):
                    break
            return -2*a**2 * (beta_2+1) * (alpha_2-beta_2+1), [4, beta_1, beta_2+1, beta_3, -1], [4, alpha_1-beta_1, alpha_2-beta_2+1, alpha_3-beta_3, -1], i # Return \{( -2*a^2 * (beta_2+1) * (alpha_2-beta_2+1), \frac{1}{(\beta+e_i)!} \partial^{\beta+e_i} \circ \partial_t, \frac{1}{(\alpha-\beta+e_i)!} \partial^{\alpha-\beta+e_i} \partial_t )\}
        elif m==5:
            i = 3 # coordinate in concern will be z_3
            while True: # accept-reject sampling
                beta_1 = torch.randint(0, alpha_1 + 1, (1,)).item() # uniform random integer in [0, alpha_1] inclusive
                beta_2 = torch.randint(0, alpha_2 + 1, (1,)).item() # uniform random integer in [0, alpha_2] inclusive
                beta_3 = torch.randint(0, alpha_3 + 1, (1,)).item() # uniform random integer in [0, alpha_3] inclusive
                L = ( 4 * (alpha_3 - beta_3 + 1) * (beta_3 + 1) ) / ((2+alpha_3)**2)
                if torch.rand(1).item() <= torch.abs(L):
                    break
            return -2*a**2 * (beta_3+1) * (alpha_3-beta_3+1), [4, beta_1, beta_2, beta_3+1, -1], [4, alpha_1-beta_1, alpha_2-beta_2, alpha_3-beta_3+1, -1], i # Return \{( -2*a^2 * (beta_3+1) * (alpha_3-beta_3+1), \frac{1}{(\beta+e_i)!} \partial^{\beta+e_i} \circ \partial_t, \frac{1}{(\alpha-\beta+e_i)!} \partial^{\alpha-\beta+e_i} \partial_t )\}
    elif form == 4:
        beta_1 = torch.randint(0, alpha_1 + 1, (1,)).item() # uniform random integer in [0, alpha_1] inclusive
        beta_2 = torch.randint(0, alpha_2 + 1, (1,)).item() # uniform random integer in [0, alpha_2] inclusive
        beta_3 = torch.randint(0, alpha_3 + 1, (1,)).item() # uniform random integer in [0, alpha_3] inclusive
        return 1, [2, beta_1, beta_2, beta_3, 1], [4, alpha_1 - beta_1, alpha_2 - beta_2, alpha_3 - beta_3, -1], None # Return \{(1, \frac{1}{\beta!} \partial^{\beta} \circ f^{(1)}, \frac{1}{(\alpha-\beta)!} \partial^{\alpha-\beta} \partial_t )\}
    elif form == 5:
        beta_1 = torch.randint(0, alpha_1 + 1, (1,)).item() # uniform random integer in [0, alpha_1] inclusive
        beta_2 = torch.randint(0, alpha_2 + 1, (1,)).item() # uniform random integer in [0, alpha_2] inclusive
        beta_3 = torch.randint(0, alpha_3 + 1, (1,)).item() # uniform random integer in [0, alpha_3] inclusive
        m = torch.randint(1, 3, (1,)).item() # uniform random integer in [1, 2] inclusive
        if m==1:
            return 1, [2, beta_1, beta_2, beta_3, 2], [3, alpha_1 - beta_1, alpha_2 - beta_2, alpha_3 - beta_3, -1], None # Return \{(1, \frac{1}{\beta!} \partial^{\beta} \circ f^{(2)}, \frac{1}{(\alpha-\beta)!} \partial^{\alpha-\beta} ((\partial_t(\cdot))^2) )\}
        elif m==2:
            return 1, [2, beta_1, beta_2, beta_3, 1], [5, alpha_1 - beta_1, alpha_2 - beta_2, alpha_3 - beta_3, -1], None # Return \{(1, \frac{1}{\beta!} \partial^{\beta} \circ f^{(1)}, \frac{1}{(\alpha-\beta)!} \partial^{\alpha-\beta} \partial_{tt} )\}

def branching3D(code, phi, psi, f, z1, z2, z3, t, a, lambda_):
    tau = torch.distributions.Exponential(lambda_).sample() # sample waiting time tau from exponential distribution with rate lambda
    uniform = torch.rand(1).item()  # sample U uniformly from [0, 1]
    theta = torch.rand(1).item() * 2 * torch.pi # sample theta uniformly from [0, 2*pi]
    if tau >= t:
        eta = torch.acos(1-2*uniform) # eta = arccos(1-2U)
        y_1 = a * t * torch.sin(eta) * torch.cos(theta) # spatial point z_1 = a * r * cos(theta)
        y_2 = a * t * torch.sin(eta) * torch.sin(theta) # spatial point z_2 = a * r * sin(theta)
        y_3 = a * t * torch.cos(eta) # spatial point z_3 = a * r * cos(eta)
        i_1 = tilde_phi(code, phi, psi, f, z1+y_1, z2+y_2, z3+y_3) # compute \tilde{\phi}(code, phi, psi, f, z1+y_1, z2+y_2, z_3+y_3)
        i_2 = y_1 * gradient_tilde_phi('z1', code, phi, psi, f, z1+y_1, z2+y_2, z3+y_3) + y_2 * gradient_tilde_phi('z2', code, phi, psi, f, z1+y_1, z2+y_2, z3+y_3) + y_3 * gradient_tilde_phi('z3', code, phi, psi, f, z1+y_1, z2+y_2,z3+y_3) # compute y_1 * \nabla_{z_1} \tilde{\phi}(code, phi, psi, f, z1+y_1,z2+y_2,z3+y_3) + y_2 * \nabla_{z_2} \tilde{\phi}(code, phi, psi,f,z1+y_1,z2+y_2,z3+y_3) + y_3 * \nabla_{z_3} \tilde{\phi}(code,\phi,\psi,f,z1+y_1,z2+y_2,z3+y_3)
        i_3 = t * tilde_psi(code, phi, psi,f,z1+y_1,z2+y_2,z3+y_3,a) # compute \tilde{\psi}(code,\phi,\psi,f,z1+z_y,z2+z_y,z3+z_z,a)
        return ( 1./(lambda_*torch.exp(-lambda_*t)) ) * (i_1 + i_2 + i_3)
    else:
        gamma_1, new_code_1, new_code_2, i = QC(code, a) # compute the next codes and coefficient using the code algebra
        H = tau * gamma_1 / (lambda_ * torch.exp(-lambda_ * tau)) # compute the coefficient H for the next codes
        alpha_1 = code[1] # alpha_1 of current code
        alpha_2 = code[2] # alpha_2 of current code
        alpha_3 = code[3] # alpha_3 of current code
        if code[0]==2: # if current code is of form \partial^{\alpha} \circ f^{(j)}
            H *= 5 * (1 + alpha_1)* (1 + alpha_2)* (1 + alpha_3)
            if i==1: # if the coordinate in concern is z_1
                H *= ( torch.abs(a)**2 * (2 + alpha_1) * (3 + alpha_1) ) / (6 * torch.abs(gamma_1))
            if i==2: # if the coordinate in concern is z_2
                H *= ( torch.abs(a)**2 * (2 + alpha_2) * (3 + alpha_2) ) / (6 * torch.abs(gamma_1))
            if i==3: # if the coordinate in concern is z_3
                H *= ( torch.abs(a)**2 * (2 + alpha_3) * (3 + alpha_3) ) / (6 * torch.abs(gamma_1))
        elif code[0]==3: # if current code is of form \partial^{\alpha} \circ ((\partial_t(\cdot))^2)
            H *= 10 * (1 + alpha_1)* (1 + alpha_2)* (1 + alpha_3)
            if i==1: # if the coordinate in concern is z_1
                H *= ( torch.abs(a)**2 * (2 + alpha_1) * (3 + alpha_1) ) / (6 * torch.abs(gamma_1))
            if i==2: # if the coordinate in concern is z_2
                H *= ( torch.abs(a)**2 * (2 + alpha_2) * (3 + alpha_2) ) / (6 * torch.abs(gamma_1))
            if i==3: # if the coordinate in concern is z_3
                H *= ( torch.abs(a)**2 * (2 + alpha_3) * (3 + alpha_3) ) / (6 * torch.abs(gamma_1))
        elif code[0]==4 or code[0]==5: # if current code is of form \partial^{\alpha} \circ \partial_t or \partial^{\alpha} \circ \partial_{tt}
            H *= (1 + alpha_1)* (1 + alpha_2)* (1 + alpha_3)
        eta = torch.acos(1-2*uniform) # eta = arccos(1-2U)
        y_1 = a * tau * torch.sin(eta) * torch.cos(theta) # spatial point z_1 = a * r * cos(theta)
        y_2 = a * tau * torch.sin(eta) * torch.sin(theta) # spatial point z_2 = a * r * sin(theta)
        y_3 = a * tau * torch.cos(eta) # spatial point z_3 = a * r * cos(eta)
        H *= branching3D(new_code_1, phi, psi, f, z1+y_1, z2+y_2, z3+y_3, t-tau, a, lambda_) # compute the contribution from the next code
        if new_code_2 is not None: # if there is a second next code
            H *= branching3D(new_code_2, phi, psi, f, z1+y_1, z2+y_2, z3+y_3, t-tau, a, lambda_) # compute the contribution from the second next code
        return H
            
def monte_carlo_simulation(phi, psi, f, z1, z2, z3, t, a, lambda_, num_samples=1000):
    # Help me implement a multi-threading version of this function to speed up the simulation
    from concurrent.futures import ThreadPoolExecutor
    code = [1, 0, 0, -1] # start with the code representing the identity operator Id
    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(branching3D, code, phi, psi, f, z1, z2, z3, t, a, lambda_) for _ in range(num_samples)]
        results = [future.result() for future in futures]
    return torch.mean(torch.tensor(results))

if __name__ == "__main__":
    import os
    import csv
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    phi = lambda x, y, z: torch.sin(x) * torch.cos(y) * torch.cos(z)
    psi = lambda x, y, z: torch.cos(x) * torch.sin(y) * torch.sin(z)
    f = lambda u: u**3
    z1 = torch.tensor(0.5 + 0j, requires_grad=True, device=device) # complex number with requires_grad=True to enable differentiation, representing the initial spatial point z_1
    z2 = torch.tensor(0.5 + 0j, requires_grad=True, device=device) # complex number with requires_grad=True to enable differentiation, representing the initial spatial point z_2
    z3 = torch.tensor(0.5 + 0j, requires_grad=True, device=device) # complex number with requires_grad=True to enable differentiation, representing the initial spatial point z_3
    a = 1.0 + 0j # complex number representing the spatial scaling factor
    lambda_ = 1.0 # real number, rate parameter for the exponential distribution governing the waiting times in the branching process
    t_values = torch.arange(0, 1.1, 0.1) # list of t values from 0 to 1 with step 0.1
    real_results = []
    imag_results = []
    num_samples = 10000 # number of Monte Carlo samples to use for each t
    for t in t_values:
        result = monte_carlo_simulation(phi, psi, f, z1, z2, z3, t.item(), a, lambda_, num_samples)
        real_results.append(result.real.item())
        imag_results.append(result.imag.item())
        print(f"t={t.item():.1f}, Real part: {result.real.item():.6f}, Imaginary part: {result.imag.item():.6f}")
    # Create directory if it does not exist
    os.makedirs("results", exist_ok=True)
    # Save results to CSV file
    with open("results/results.csv", mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(real_results) # write real parts in the first row
        writer.writerow(imag_results) # write imaginary parts in the second row
    