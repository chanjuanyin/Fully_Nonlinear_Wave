import torch
import math

def nth_derivative_scalar(f, u, order):
    """Compute f^{(j)}(u)
    Compute the order-th derivative of a scalar->scalar function f at u.
    u must require grad; result keeps graph so it can be differentiated w.r.t inputs of u.
    If f is polynomial and we exceed its degree, returns 0 (via u * 0) so computation graph stays connected."""
    # Ensure u requires grad - only works on leaf tensors
    if not u.requires_grad:
        if not u.is_leaf:
            raise RuntimeError(
                f"nth_derivative_scalar: u is a non-leaf tensor without requires_grad. "
                f"Cannot enable gradients on non-leaf tensors. "
                f"u.shape={u.shape}, u.dtype={u.dtype}, u.device={u.device}. "
                f"This indicates the computation graph isn't properly set up in the calling code."
            )
        else:
            u.requires_grad_(True)
    
    y = f(u)
    for k in range(order):
        if not y.requires_grad:
            # Derivative sequence has terminated to a constant (polynomial case)
            # All higher derivatives are mathematically 0
            # Return u * 0 to keep it connected to the computation graph
            return u * 0.0
        
        grad_out = torch.ones_like(y, dtype=y.dtype, device=y.device)
        (y,) = torch.autograd.grad(y, u, grad_outputs=grad_out, create_graph=True, retain_graph=True)
    return y

def mixed_partial_orders(g, inputs, orders):
    """Compute \partial_{z_1}^{\alpha_1} of some functions."""
    # For example inputs = (x, y, z), so inputs[0] = x, inputs[1] = y, inputs[2] = z
    # orders: list of (var_index, order), e.g. [(0, 2), (1, 3), (2, 4)] means d^2/dx^2, d^3/dy^3, d^4/dz^4
    # Ensure all inputs require gradients
    inputs_list = []
    for inp in inputs:
        if inp.requires_grad:
            inputs_list.append(inp)
        elif inp.is_leaf:
            inp.requires_grad_(True)
            inputs_list.append(inp)
        else:
            # Non-leaf tensor without requires_grad - cannot enable it
            raise RuntimeError(
                f"mixed_partial_orders: input is a non-leaf tensor without requires_grad. "
                f"shape={inp.shape}, dtype={inp.dtype}, device={inp.device}. "
                f"Cannot enable gradients on non-leaf tensors. "
                f"This indicates the computation graph isn't properly set up."
            )
    inputs = tuple(inputs_list)
    y = g(*inputs)
    steps = []
    for idx, k in orders:
        steps.extend([idx] * k)  # expand into sequential steps
    for idx in steps:
        grad_out = torch.ones_like(y, dtype=y.dtype, device=y.device)
        (y,) = torch.autograd.grad(
            y, inputs[idx],
            grad_outputs=grad_out,
            create_graph=True, # always create graph for chained derivatives
            retain_graph=True  # always retain graph for multiple derivative paths
        )
    return y

def torch_factorial_int(alpha_1: int, device=None) -> torch.Tensor:
    if not isinstance(alpha_1, int):
        raise TypeError("alpha_1 must be a Python int")
    if alpha_1 < 0:
        raise ValueError("alpha_1 must be non-negative")

    value = math.factorial(alpha_1)  # exact Python integer
    if value > torch.iinfo(torch.int64).max:
        print(f"Factorial of {alpha_1} is too large for torch.int64. Returning infinity.")
        raise OverflowError("factorial too large for torch.int64")

    return torch.tensor(value, dtype=torch.int64, device=device)

# code is of the form [int, int, int, int]
# [1, \alpha_1, \alpha_2, -1] means \frac{1}{\alpha_1! \alpha_2!} \partial_{z_1}^{\alpha_1} \partial_{z_2}^{\alpha_2}
# [2, \alpha_1, \alpha_2, j] means \frac{1}{\alpha_1! \alpha_2!} \partial_{z_1}^{\alpha_1} \partial_{z_2}^{\alpha_2} f^{(j)}
# [3, \alpha_1, \alpha_2, -1] means \frac{1}{\alpha_1! \alpha_2!} \partial_{z_1}^{\alpha_1} \partial_{z_2}^{\alpha_2} ((\partial_t(\cdot))^2)
# [4, \alpha_1, \alpha_2, -1] means \frac{1}{\alpha_1! \alpha_2!} \partial_{z_1}^{\alpha_1} \partial_{z_2}^{\alpha_2} \partial_t
# [5, \alpha_1, \alpha_2, -1] means \frac{1}{\alpha_1! \alpha_2!} \partial_{z_1}^{\alpha_1} \partial_{z_2}^{\alpha_2} \partial_{tt}

def tilde_phi(code, phi, psi, f, z1, z2):
    form = code[0]
    alpha_1 = code[1]
    alpha_2 = code[2]
    j = code[3]
    if form == 0:
        return phi(z1,z2)
    elif form == 1:
        return 1. / (torch_factorial_int(alpha_1) * torch_factorial_int(alpha_2)) * \
            mixed_partial_orders(g = phi, inputs = (z1, z2), orders = [(0, alpha_1), (1, alpha_2)])
    elif form == 2:
        return 1. / (torch_factorial_int(alpha_1) * torch_factorial_int(alpha_2)) * \
            mixed_partial_orders(g = lambda x_in, y_in: nth_derivative_scalar(f, phi(x_in, y_in), j), 
                                 inputs = (z1, z2), 
                                 orders = [(0, alpha_1), (1, alpha_2)])
    elif form == 3:
        return 1. / (torch_factorial_int(alpha_1) * torch_factorial_int(alpha_2)) * \
            mixed_partial_orders(g = lambda x_in, y_in: psi(x_in, y_in)**2,
                                 inputs = (z1, z2), 
                                 orders = [(0, alpha_1), (1, alpha_2)])
    elif form == 4:
        return 1. / (torch_factorial_int(alpha_1) * torch_factorial_int(alpha_2)) * \
            mixed_partial_orders(g = lambda x_in, y_in: psi(x_in, y_in),
                                 inputs = (z1, z2), 
                                 orders = [(0, alpha_1), (1, alpha_2)])
    elif form == 5:
        return 1. / (torch_factorial_int(alpha_1) * torch_factorial_int(alpha_2)) * \
            (
                mixed_partial_orders(g = phi, inputs=(z1, z2), orders=[(0, alpha_1 + 2), (1, alpha_2)]) + \
                mixed_partial_orders(g = phi, inputs=(z1, z2), orders=[(0, alpha_1), (1, alpha_2 + 2)]) + \
                mixed_partial_orders(g = lambda x_in, y_in: f(phi(x_in, y_in)), inputs=(z1, z2), orders=[(0, alpha_1), (1, alpha_2)])
            )

def gradient_tilde_phi(coordinate, code, phi, psi, f, z1, z2):
    if coordinate=='z1':
        return mixed_partial_orders(g = lambda x_in, y_in: tilde_phi(code, phi, psi, f, x_in, y_in), inputs=(z1, z2), orders=[(0, 1), (1, 0)])
    elif coordinate=='z2':
        return mixed_partial_orders(g = lambda x_in, y_in: tilde_phi(code, phi, psi, f, x_in, y_in), inputs=(z1, z2), orders=[(0, 0), (1, 1)])
    
def tilde_psi(code, phi, psi, f, z1, z2, a):
    form = code[0]
    alpha_1 = code[1]
    alpha_2 = code[2]
    j = code[3]
    if form == 0:
        return psi(z1,z2)
    elif form == 1:
        return 1. / (torch_factorial_int(alpha_1) * torch_factorial_int(alpha_2)) * \
            mixed_partial_orders(g = psi, inputs = (z1, z2), orders = [(0, alpha_1), (1, alpha_2)])
    elif form == 2:
        return 1. / (torch_factorial_int(alpha_1) * torch_factorial_int(alpha_2)) * \
            mixed_partial_orders(g = lambda x_in, y_in: psi(x_in, y_in) * nth_derivative_scalar(f, phi(x_in, y_in), j+1), 
                                 inputs = (z1, z2), 
                                 orders = [(0, alpha_1), (1, alpha_2)])
    elif form == 3:
        return 2. / (torch_factorial_int(alpha_1) * torch_factorial_int(alpha_2)) * \
            mixed_partial_orders(g = lambda x_in, y_in: a**2 * psi(x_in, y_in) * (
                                                                mixed_partial_orders(g=phi,
                                                                                     inputs=(x_in, y_in), 
                                                                                     orders=[(0, 2), (1, 0)]) + \
                                                                mixed_partial_orders(g=phi,
                                                                                     inputs=(x_in, y_in), 
                                                                                     orders=[(0, 0), (1, 2)])
                                                                )
                                                        + psi(x_in, y_in) * f(phi(x_in, y_in)),
                                 inputs = (z1, z2),
                                 orders = [(0, alpha_1), (1, alpha_2)])
    elif form == 4:
        return 1. / (torch_factorial_int(alpha_1) * torch_factorial_int(alpha_2)) * \
            (
                mixed_partial_orders(g = phi, inputs=(z1, z2), orders=[(0, alpha_1 + 2), (1, alpha_2)]) + \
                mixed_partial_orders(g = phi, inputs=(z1, z2), orders=[(0, alpha_1), (1, alpha_2 + 2)]) + \
                mixed_partial_orders(g = lambda x_in, y_in: f(phi(x_in, y_in)), inputs=(z1, z2), orders=[(0, alpha_1), (1, alpha_2)])
            )
    elif form == 5:
        return 1. / (torch_factorial_int(alpha_1) * torch_factorial_int(alpha_2)) * \
            (
                mixed_partial_orders(g = psi, inputs=(z1, z2), orders=[(0, alpha_1 + 2), (1, alpha_2)]) + \
                mixed_partial_orders(g = psi, inputs=(z1, z2), orders=[(0, alpha_1), (1, alpha_2 + 2)]) + \
                mixed_partial_orders(g = lambda x_in, y_in: psi(x_in, y_in) * nth_derivative_scalar(f, phi(x_in, y_in), 1), 
                                     inputs=(z1, z2), 
                                     orders=[(0, alpha_1), (1, alpha_2)]
                                     )
            )

def QC(code, a):
    """Probabilistically generate the next code based on the code algebra"""
    form = code[0]
    alpha_1 = code[1]
    alpha_2 = code[2]
    j = code[3]
    if form == 0:
            return 1, [2, 0, 0, 0], None, None # Return \{(1, f, \emptyset)\}
    elif form == 1:
        if alpha_1 > 0: # i which is the first index such that \alpha_i\neq 0 is 1
            while True: # accept-reject sampling
                beta_1 = torch.randint(0, alpha_1, (1,)).item() # uniform random integer in [0, alpha_1-1] inclusive
                beta_2 = torch.randint(0, alpha_2 + 1, (1,)).item() # uniform random integer in [0, alpha_2] inclusive
                gamma_1 = (alpha_1 - beta_1) / alpha_1
                if torch.rand(1).item() <= abs(gamma_1):
                    break
            return gamma_1, [2, beta_1, beta_2, 1], [1, alpha_1 - beta_1, alpha_2 - beta_2, -1], None # Return \{( gamma_1, \frac{1}{\beta!} \partial^{\beta} \circ f^{(1)}, \frac{1}{(alpha_1-beta_1)!} \partial^{\alpha-\beta} )\}
        elif alpha_2 > 0: # i which is the first index such that \alpha_i\neq 0 is 2
            while True: # accept-reject sampling
                beta_1 = torch.randint(0, alpha_1 + 1, (1,)).item() # uniform random integer in [0, alpha_1] inclusive
                beta_2 = torch.randint(0, alpha_2, (1,)).item() # uniform random integer in [0, alpha_2-1] inclusive
                gamma_1 = (alpha_2 - beta_2) / alpha_2
                if torch.rand(1).item() <= abs(gamma_1):
                    break
            return gamma_1, [2, beta_1, beta_2, 1], [1, alpha_1 - beta_1, alpha_2 - beta_2, -1], None # Return \{( gamma_2, \frac{1}{\beta!} \partial^{\beta} \circ f^{(1)}, \frac{1}{(alpha_2-beta_2)!} \partial^{\alpha-\beta} )\}
    elif form == 2:
        m = torch.randint(1, 5, (1,)).item() # uniform random integer in [1, 4] inclusive
        beta_1 = torch.randint(0, alpha_1 + 1, (1,)).item() # uniform random integer in [0, alpha_1] inclusive
        beta_2 = torch.randint(0, alpha_2 + 1, (1,)).item() # uniform random integer in [0, alpha_2] inclusive
        if m==1:
            return 1, [2, beta_1, beta_2, j+2], [3, alpha_1 - beta_1, alpha_2 - beta_2, -1], None # Return \{(1, \frac{1}{\beta!} \partial^{\beta} \circ f^{(j+2)}, \frac{1}{(\alpha-\beta)!} \partial^{\alpha-\beta} ((\partial_t(\cdot))^2) )\}
        elif m==2:
            return 1, [2, beta_1, beta_2, j+1], [2, alpha_1 - beta_1, alpha_2 - beta_2, 0], None # Return \{(1, \frac{1}{\beta!} \partial^{\beta} \circ f^{(j+1)}, \frac{1}{(\alpha-\beta)!} \partial^{\alpha-\beta} f )\}
        elif m==3:
            i = 1 # coordinate in concern will be z_1
            while True: # accept-reject sampling
                beta_1 = torch.randint(0, alpha_1 + 1, (1,)).item() # uniform random integer in [0, alpha_1] inclusive
                beta_2 = torch.randint(0, alpha_2 + 1, (1,)).item() # uniform random integer in [0, alpha_2] inclusive
                L = ( 4 * (alpha_1 - beta_1 + 1) * (beta_1 + 1) ) / ((2+alpha_1)**2)
                if torch.rand(1).item() <= abs(L):
                    break
            return -a**2 * (beta_1+1) * (alpha_1-beta_1+1), [2, beta_1+1, beta_2, j+1], [1, alpha_1-beta_1+1, alpha_2-beta_2, -1], i # Return \{( -a^2 * (beta_1+1) * (alpha_1-beta_1+1), \frac{1}{(\beta+e_i)!} \partial^{\beta+e_i} \circ f^{(j+1)}, \frac{1}{(\alpha-\beta+e_i)!} \partial^{\alpha-\beta+e_i} )\}
        elif m==4:
            i = 2 # coordinate in concern will be z_2
            while True: # accept-reject sampling
                beta_1 = torch.randint(0, alpha_1 + 1, (1,)).item() # uniform random integer in [0, alpha_1] inclusive
                beta_2 = torch.randint(0, alpha_2 + 1, (1,)).item() # uniform random integer in [0, alpha_2] inclusive
                L = ( 4 * (alpha_2 - beta_2 + 1) * (beta_2 + 1) ) / ((2+alpha_2)**2)
                if torch.rand(1).item() <= abs(L):
                    break
            return -a**2 * (beta_2+1) * (alpha_2-beta_2+1), [2, beta_1, beta_2+1, j+1], [1, alpha_1-beta_1, alpha_2-beta_2+1, -1], i # Return \{( -a^2 * (beta_2+1) * (alpha_2-beta_2+1), \frac{1}{(\beta+e_i)!} \partial^{\beta+e_i} \circ f^{(j+1)}, \frac{1}{(\alpha-\beta+e_i)!} \partial^{\alpha-\beta+e_i} )\}
    elif form == 3:
        m = torch.randint(1, 5, (1,)).item() # uniform random integer in [1, 4] inclusive
        beta_1 = torch.randint(0, alpha_1 + 1, (1,)).item() # uniform random integer in [0, alpha_1] inclusive
        beta_2 = torch.randint(0, alpha_2 + 1, (1,)).item() # uniform random integer in [0, alpha_2] inclusive
        if m==1:
            return 2, [5, beta_1, beta_2, -1], [5, alpha_1 - beta_1, alpha_2 - beta_2, -1], None # Return \{(2, \frac{1}{\beta!} \partial^{\beta} \circ \partial_{tt}, \frac{1}{(\alpha-\beta)!} \partial^{\alpha-\beta} \partial_{tt} )\}
        elif m==2:
            return 2, [3, beta_1, beta_2, -1], [2, alpha_1 - beta_1, alpha_2 - beta_2, 1], None # Return \{(2, \frac{1}{\beta!} \partial^{\beta} \circ ((\partial_t(\cdot))^2), \frac{1}{(\alpha-\beta)!} \partial^{\alpha-\beta} f^{(1)} )\}
        elif m==3:
            i = 1 # coordinate in concern will be z_1
            while True: # accept-reject sampling
                beta_1 = torch.randint(0, alpha_1 + 1, (1,)).item() # uniform random integer in [0, alpha_1] inclusive
                beta_2 = torch.randint(0, alpha_2 + 1, (1,)).item() # uniform random integer in [0, alpha_2] inclusive
                L = ( 4 * (alpha_1 - beta_1 + 1) * (beta_1 + 1) ) / ((2+alpha_1)**2)
                if torch.rand(1).item() <= abs(L):
                    break
            return -2*a**2 * (beta_1+1) * (alpha_1-beta_1+1), [4, beta_1+1, beta_2, -1], [4, alpha_1-beta_1+1, alpha_2-beta_2, -1], i # Return \{( -2*a^2 * (beta_1+1) * (alpha_1-beta_1+1), \frac{1}{(\beta+e_i)!} \partial^{\beta+e_i} \circ \partial_t, \frac{1}{(\alpha-\beta+e_i)!} \partial^{\alpha-\beta+e_i} \partial_t )\}
        elif m==4:
            i = 2 # coordinate in concern will be z_2
            while True: # accept-reject sampling
                beta_1 = torch.randint(0, alpha_1 + 1, (1,)).item() # uniform random integer in [0, alpha_1] inclusive
                beta_2 = torch.randint(0, alpha_2 + 1, (1,)).item() # uniform random integer in [0, alpha_2] inclusive
                L = ( 4 * (alpha_2 - beta_2 + 1) * (beta_2 + 1) ) / ((2+alpha_2)**2)
                if torch.rand(1).item() <= abs(L):
                    break
            return -2*a**2 * (beta_2+1) * (alpha_2-beta_2+1), [4, beta_1, beta_2+1, -1], [4, alpha_1-beta_1, alpha_2-beta_2+1, -1], i # Return \{( -2*a^2 * (beta_2+1) * (alpha_2-beta_2+1), \frac{1}{(\beta+e_i)!} \partial^{\beta+e_i} \circ \partial_t, \frac{1}{(\alpha-\beta+e_i)!} \partial^{\alpha-\beta+e_i} \partial_t )\}
    elif form == 4:
        beta_1 = torch.randint(0, alpha_1 + 1, (1,)).item() # uniform random integer in [0, alpha_1] inclusive
        beta_2 = torch.randint(0, alpha_2 + 1, (1,)).item() # uniform random integer in [0, alpha_2] inclusive
        return 1, [2, beta_1, beta_2, 1], [4, alpha_1 - beta_1, alpha_2 - beta_2, -1], None # Return \{(1, \frac{1}{\beta!} \partial^{\beta} \circ f^{(1)}, \frac{1}{(\alpha-\beta)!} \partial^{\alpha-\beta} \partial_t )\}
    elif form == 5:
        beta_1 = torch.randint(0, alpha_1 + 1, (1,)).item() # uniform random integer in [0, alpha_1] inclusive
        beta_2 = torch.randint(0, alpha_2 + 1, (1,)).item() # uniform random integer in [0, alpha_2] inclusive
        m = torch.randint(1, 3, (1,)).item() # uniform random integer in [1, 2] inclusive
        if m==1:
            return 1, [2, beta_1, beta_2, 2], [3, alpha_1 - beta_1, alpha_2 - beta_2, -1], None # Return \{(1, \frac{1}{\beta!} \partial^{\beta} \circ f^{(2)}, \frac{1}{(\alpha-\beta)!} \partial^{\alpha-\beta} ((\partial_t(\cdot))^2) )\}
        elif m==2:
            return 1, [2, beta_1, beta_2, 1], [5, alpha_1 - beta_1, alpha_2 - beta_2, -1], None # Return \{(1, \frac{1}{\beta!} \partial^{\beta} \circ f^{(1)}, \frac{1}{(\alpha-\beta)!} \partial^{\alpha-\beta} \partial_{tt} )\}

def branching2D(code, phi, psi, f, z1, z2, t, a, lambda_):
    tau = torch.distributions.Exponential(lambda_).sample().item() # sample waiting time tau from exponential distribution with rate lambda
    uniform = torch.rand(1).item()  # sample U uniformly from [0, 1]
    theta = torch.rand(1).item() * 2 * torch.pi # sample theta uniformly from [0, 2*pi]
    if tau >= t:
        r = t * math.sqrt(1-(1-uniform)**2) # radius of the disk from which we sample the spatial point
        y_1 = a * r * math.cos(theta) # spatial point z_1 = a * r * cos(theta)
        y_2 = a * r * math.sin(theta) # spatial point z_2 = a * r * sin(theta)
        i_1 = tilde_phi(code, phi, psi, f, z1+y_1, z2+y_2) # compute \tilde{\phi}(code, phi, psi, f, z1+y_1, z2+y_2)
        i_2 = y_1 * gradient_tilde_phi('z1', code, phi, psi, f, z1+y_1, z2+y_2) + \
              y_2 * gradient_tilde_phi('z2', code, phi, psi, f, z1+y_1, z2+y_2) # compute y_1 * \nabla_{z_1} \tilde{\phi}(code, phi, psi, f, z1+y_1, z2+y_2) + y_2 * \nabla_{z_2} \tilde{\phi}(code, phi, psi, f, z1+y_1, z2+y_2)
        i_3 = t * tilde_psi(code, phi, psi, f, z1+y_1, z2+y_2, a) # compute \tilde{\psi}(code, phi, psi, f, z1+y_1, z2+y_2, a)
        return math.exp(lambda_ * t) * (i_1 + i_2 + i_3)
    else:
        gamma_1, new_code_1, new_code_2, i = QC(code, a) # compute the next codes and coefficient using the code algebra
        # Ensure gamma_1 is a tensor on the correct device
        if isinstance(gamma_1, torch.Tensor):
            gamma_1 = gamma_1.clone().to(dtype=torch.complex64)
        else:
            gamma_1 = torch.tensor(gamma_1, dtype=torch.complex64, device=z1.device)
        H = gamma_1 * tau * math.exp(lambda_ * tau) / lambda_ # compute the coefficient H for the next codes
        alpha_1 = code[1] # alpha_1 of current code
        alpha_2 = code[2] # alpha_2 of current code
        if code[0]==1: # if current code is of form \partial^{\alpha}
            H *= (1 + alpha_1)* (1 + alpha_2) / (2 * torch.abs(gamma_1)) # multiply by (1 + alpha_1)* (1 + alpha_2) / (2 * abs(gamma_1)) to account for the change in the order of differentiation and the coefficient from the code algebra
        elif code[0]==2: # if current code is of form \partial^{\alpha} \circ f^{(j)}
            H *= 4 * (1 + alpha_1)* (1 + alpha_2)
            if i==1: # if the coordinate in concern is z_1
                H *= ( torch.abs(a)**2 * (2 + alpha_1) * (3 + alpha_1) ) / (6 * torch.abs(gamma_1))
            if i==2: # if the coordinate in concern is z_2
                H *= ( torch.abs(a)**2 * (2 + alpha_2) * (3 + alpha_2) ) / (6 * torch.abs(gamma_1))
        elif code[0]==3: # if current code is of form \partial^{\alpha} \circ ((\partial_t(\cdot))^2)
            H *= 8 * (1 + alpha_1)* (1 + alpha_2)
            if i==1: # if the coordinate in concern is z_1
                H *= ( torch.abs(a)**2 * (2 + alpha_1) * (3 + alpha_1) ) / (6 * torch.abs(gamma_1))
            if i==2: # if the coordinate in concern is z_2
                H *= ( torch.abs(a)**2 * (2 + alpha_2) * (3 + alpha_2) ) / (6 * torch.abs(gamma_1))
        elif code[0]==4 or code[0]==5: # if current code is of form \partial^{\alpha} \circ \partial_t or \partial^{\alpha} \circ \partial_{tt}
            H *= (1 + alpha_1)* (1 + alpha_2)
        r = tau * math.sqrt(1-(1-uniform)**2) # radius of the disk from which we sample the spatial point
        y_1 = a * r * math.cos(theta) # spatial point z_1 = a * r * cos(theta)
        y_2 = a * r * math.sin(theta) # spatial point z_2 = a * r * sin(theta)
        H *= branching2D(new_code_1, phi, psi, f, z1+y_1, z2+y_2, t-tau, a, lambda_) # compute the contribution from the next code
        if new_code_2 is not None: # if there is a second next code
            H *= branching2D(new_code_2, phi, psi, f, z1+y_1, z2+y_2, t-tau, a, lambda_) # compute the contribution from the second next code
        return H
            
def monte_carlo_simulation(phi, psi, f, z1, z2, t, a, lambda_, num_samples=1000):
    # Help me implement a multi-threading version of this function to speed up the simulation
    from concurrent.futures import ThreadPoolExecutor
    code = [0, 0, 0, -1] # start with the code representing the identity operator Id
    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(branching2D, code, phi, psi, f, z1, z2, t, a, lambda_) for _ in range(num_samples)]
        results = [future.result() for future in futures]
    # Detach results to avoid gradient warnings when creating a tensor
    results_detached = [r.detach() if isinstance(r, torch.Tensor) else r for r in results]
    return torch.mean(torch.tensor(results_detached))

if __name__ == "__main__":
    import os
    import csv
    import time
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Print device information
    if torch.cuda.is_available():
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA device count: {torch.cuda.device_count()}")
        print(f"Current CUDA device: {torch.cuda.current_device()}")
    else:
        print("CUDA not available; using CPU")
    print(f"Device: {device}\n")

    w = 0.5 + 0.0j # example complex number representing the wave speed
    phi = lambda x1, x2: 4 * torch.arctan(torch.exp((4*w/3)*(x1+x2))) # example initial condition phi(x) = 4 * arctan(exp(2x/3))
    psi = lambda x1, x2: (8*w/3)*(torch.exp((4*w/3)*(x1+x2)) / (1 + torch.exp((8*w/3)*(x1+x2)))) # example initial condition psi(x) = (4/3)*(exp(2x/3) / (1 + exp(4x/3)))
    f = lambda u: -(4*(w**2)/3)*torch.sin(u) # example nonlinearity f(u) = -(4*w^2/3)*sin(u)
    z1 = torch.tensor(1.0 + 0.0j, requires_grad=True, device=device) # complex number with requires_grad=True to enable differentiation, representing the initial spatial point z1
    z2 = torch.tensor(1.0 + 0.0j, requires_grad=True, device=device) # complex number with requires_grad=True to enable differentiation, representing the initial spatial point z2
    a = torch.tensor((1.0/math.sqrt(2)) + 0.0j, dtype=torch.complex64, device=device) # complex number representing the spatial scaling factor
    lambda_ = 1.0 # real number, rate parameter for the exponential distribution governing the waiting times in the branching process
    t_values = torch.arange(0, 1.1, 0.1) # list of t values from 0 to 1 with step 0.1
    real_results = []
    imag_results = []
    num_samples = 100000 # number of Monte Carlo samples to use for each t
    
    # Create directory if it does not exist
    os.makedirs("real_d2_results", exist_ok=True)
    output_file = "real_d2_results/monte_carlo.csv"

    # Initialize output file with zero placeholders (2 rows, len(t_values) columns)
    num_t_values = len(t_values)
    with open(output_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([0.0] * num_t_values)
        writer.writerow([0.0] * num_t_values)
    
    for idx, t in enumerate(t_values):
        start_time = time.perf_counter()
        result = monte_carlo_simulation(phi, psi, f, z1, z2, t.item(), a, lambda_, num_samples)
        elapsed_time = time.perf_counter() - start_time
        real_results.append(result.real.item())
        imag_results.append(result.imag.item())
        print(
            f"t={t.item():.1f}, Real part: {result.real.item():.6f}, "
            f"Imaginary part: {result.imag.item():.6f}, Time taken: {elapsed_time:.3f}s"
        )
        
        # Write results incrementally by replacing the placeholder at current index
        with open(output_file, mode='r', newline='') as file:
            reader = csv.reader(file)
            rows = list(reader)

        rows[0][idx] = str(result.real.item())
        rows[1][idx] = str(result.imag.item())

        with open(output_file, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerows(rows)
        
        print(f"Progress saved to {output_file}")
    
    print(f"\nAll results saved to {output_file}")
    