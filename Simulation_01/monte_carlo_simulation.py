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

# code is of the form [int, int, int]
# [1, \alpha_1, -1] means \frac{1}{\alpha_1!} \partial_{z_1}^{\alpha_1}
# [2, \alpha_1, j] means \frac{1}{\alpha_1!} \partial_{z_1}^{\alpha_1} f^{(j)}
# [3, \alpha_1, -1] means \frac{1}{\alpha_1!} \partial_{z_1}^{\alpha_1} ((\partial_t(\cdot))^2)
# [4, \alpha_1, -1] means \frac{1}{\alpha_1!} \partial_{z_1}^{\alpha_1} \partial_t
# [5, \alpha_1, -1] means \frac{1}{\alpha_1!} \partial_{z_1}^{\alpha_1} \partial_{tt}

def tilde_phi(code, phi, psi, f, z1):
    form = code[0]
    alpha_1 = code[1]
    j = code[2]
    if form == 0:
        return phi(z1)
    elif form == 1:
        return 1. / (torch_factorial_int(alpha_1)) * \
            mixed_partial_orders(g = phi, inputs = (z1,), orders = [(0, alpha_1)])
    elif form == 2:
        return 1. / (torch_factorial_int(alpha_1)) * \
            mixed_partial_orders(g = lambda x_in: nth_derivative_scalar(f, phi(x_in), j), 
                                 inputs = (z1,), 
                                 orders = [(0, alpha_1)])
    elif form == 3:
        return 1. / (torch_factorial_int(alpha_1)) * \
            mixed_partial_orders(g = lambda x_in: psi(x_in)**2,
                                 inputs = (z1,), 
                                 orders = [(0, alpha_1)])
    elif form == 4:
        return 1. / (torch_factorial_int(alpha_1)) * \
            mixed_partial_orders(g = lambda x_in: psi(x_in),
                                 inputs = (z1,), 
                                 orders = [(0, alpha_1)])
    elif form == 5:
        return 1. / (torch_factorial_int(alpha_1)) * \
            (
                mixed_partial_orders(g = phi, inputs=(z1,), orders=[(0, alpha_1 + 2)]) + \
                mixed_partial_orders(g = lambda x_in: f(phi(x_in)), inputs=(z1,), orders=[(0, alpha_1)])
            )
    
def tilde_psi(code, phi, psi, f, z1, a):
    form = code[0]
    alpha_1 = code[1]
    j = code[2]
    if form == 0:
        return psi(z1)
    elif form == 1:
        return 1. / (torch_factorial_int(alpha_1)) * \
            mixed_partial_orders(g = psi, inputs = (z1,), orders = [(0, alpha_1)])
    elif form == 2:
        return 1. / (torch_factorial_int(alpha_1)) * \
            mixed_partial_orders(g = lambda x_in: psi(x_in) * nth_derivative_scalar(f, phi(x_in), j+1), 
                                 inputs = (z1,), 
                                 orders = [(0, alpha_1)])
    elif form == 3:
        return 2. / (torch_factorial_int(alpha_1)) * \
            mixed_partial_orders(g = lambda x_in: a**2 * psi(x_in) * (mixed_partial_orders(g=phi,inputs=(x_in,), orders=[(0, 2)]))
                                     + psi(x_in) * f(phi(x_in)),
                                 inputs = (z1,),
                                 orders = [(0, alpha_1)])
    elif form == 4:
        return 1. / (torch_factorial_int(alpha_1)) * \
            (
                mixed_partial_orders(g = phi, inputs=(z1,), orders=[(0, alpha_1 + 2)]) + \
                mixed_partial_orders(g = lambda x_in: f(phi(x_in)), inputs=(z1,), orders=[(0, alpha_1)])
            )
    elif form == 5:
        return 1. / (torch_factorial_int(alpha_1)) * \
            (
                mixed_partial_orders(g = psi, inputs=(z1,), orders=[(0, alpha_1 + 2)]) + \
                mixed_partial_orders(g = lambda x_in: psi(x_in) * nth_derivative_scalar(f, phi(x_in), 1), 
                                     inputs=(z1,), 
                                     orders=[(0, alpha_1)])
            )

def QC(code, a):
    """Probabilistically generate the next code based on the code algebra"""
    form = code[0]
    alpha_1 = code[1]
    j = code[2]
    if form == 0:
            return 1, [2, 0, 0], None, None # Return \{(1, f, \emptyset)\}
    elif form == 1:
        while True: # accept-reject sampling
            beta_1 = torch.randint(0, alpha_1, (1,)).item() # uniform random integer in [0, alpha_1-1] inclusive
            gamma_1 = (alpha_1 - beta_1) / alpha_1
            if torch.rand(1).item() <= abs(gamma_1):
                break
        return gamma_1, [2, beta_1, 1], [1, alpha_1 - beta_1, -1], None # Return \{( gamma_1, \frac{1}{\beta!} \partial^{\beta} \circ f^{(1)}, \frac{1}{(alpha_1-beta_1)!} \partial^{\alpha-\beta} )\}
    elif form == 2:
        m = torch.randint(1, 4, (1,)).item() # uniform random integer in [1, 3] inclusive
        beta_1 = torch.randint(0, alpha_1 + 1, (1,)).item() # uniform random integer in [0, alpha_1] inclusive
        if m==1:
            return 1, [2, beta_1, j+2], [3, alpha_1 - beta_1, -1], None # Return \{(1, \frac{1}{\beta!} \partial^{\beta} \circ f^{(j+2)}, \frac{1}{(\alpha-\beta)!} \partial^{\alpha-\beta} ((\partial_t(\cdot))^2) )\}
        elif m==2:
            return 1, [2, beta_1, j+1], [2, alpha_1 - beta_1, 0], None # Return \{(1, \frac{1}{\beta!} \partial^{\beta} \circ f^{(j+1)}, \frac{1}{(\alpha-\beta)!} \partial^{\alpha-\beta} f )\}
        elif m==3:
            i = 1 # coordinate in concern will be z_1
            while True: # accept-reject sampling
                beta_1 = torch.randint(0, alpha_1 + 1, (1,)).item() # uniform random integer in [0, alpha_1] inclusive
                L = ( 4 * (alpha_1 - beta_1 + 1) * (beta_1 + 1) ) / ((2+alpha_1)**2)
                if torch.rand(1).item() <= abs(L):
                    break
            return -a**2 * (beta_1+1) * (alpha_1-beta_1+1), [2, beta_1+1, j+1], [1, alpha_1-beta_1+1, -1], i # Return \{( -a^2 * (beta_1+1) * (alpha_1-beta_1+1), \frac{1}{(\beta+e_i)!} \partial^{\beta+e_i} \circ f^{(j+1)}, \frac{1}{(\alpha-\beta+e_i)!} \partial^{\alpha-\beta+e_i} )\}
    elif form == 3:
        m = torch.randint(1, 4, (1,)).item() # uniform random integer in [1, 3] inclusive
        beta_1 = torch.randint(0, alpha_1 + 1, (1,)).item() # uniform random integer in [0, alpha_1] inclusive
        if m==1:
            return 2, [5, beta_1, -1], [5, alpha_1 - beta_1, -1], None # Return \{(2, \frac{1}{\beta!} \partial^{\beta} \circ \partial_{tt}, \frac{1}{(\alpha-\beta)!} \partial^{\alpha-\beta} \partial_{tt} )\}
        elif m==2:
            return 2, [3, beta_1, -1], [2, alpha_1 - beta_1, 1], None # Return \{(2, \frac{1}{\beta!} \partial^{\beta} \circ ((\partial_t(\cdot))^2), \frac{1}{(\alpha-\beta)!} \partial^{\alpha-\beta} f^{(1)} )\}
        elif m==3:
            i = 1 # coordinate in concern will be z_1
            while True: # accept-reject sampling
                beta_1 = torch.randint(0, alpha_1 + 1, (1,)).item() # uniform random integer in [0, alpha_1] inclusive
                L = ( 4 * (alpha_1 - beta_1 + 1) * (beta_1 + 1) ) / ((2+alpha_1)**2)
                if torch.rand(1).item() <= abs(L):
                    break
            return -2*a**2 * (beta_1+1) * (alpha_1-beta_1+1), [4, beta_1+1, -1], [4, alpha_1-beta_1+1, -1], i # Return \{( -2*a^2 * (beta_1+1) * (alpha_1-beta_1+1), \frac{1}{(\beta+e_i)!} \partial^{\beta+e_i} \circ \partial_t, \frac{1}{(\alpha-\beta+e_i)!} \partial^{\alpha-\beta+e_i} \partial_t )\}
    elif form == 4:
        beta_1 = torch.randint(0, alpha_1 + 1, (1,)).item() # uniform random integer in [0, alpha_1] inclusive
        return 1, [2, beta_1, 1], [4, alpha_1 - beta_1, -1], None # Return \{(1, \frac{1}{\beta!} \partial^{\beta} \circ f^{(1)}, \frac{1}{(\alpha-\beta)!} \partial^{\alpha-\beta} \partial_t )\}
    elif form == 5:
        beta_1 = torch.randint(0, alpha_1 + 1, (1,)).item() # uniform random integer in [0, alpha_1] inclusive
        m = torch.randint(1, 3, (1,)).item() # uniform random integer in [1, 2] inclusive
        if m==1:
            return 1, [2, beta_1, 2], [3, alpha_1 - beta_1, -1], None # Return \{(1, \frac{1}{\beta!} \partial^{\beta} \circ f^{(2)}, \frac{1}{(\alpha-\beta)!} \partial^{\alpha-\beta} ((\partial_t(\cdot))^2) )\}
        elif m==2:
            return 1, [2, beta_1, 1], [5, alpha_1 - beta_1, -1], None # Return \{(1, \frac{1}{\beta!} \partial^{\beta} \circ f^{(1)}, \frac{1}{(\alpha-\beta)!} \partial^{\alpha-\beta} \partial_{tt} )\}

def branching1D(code, phi, psi, f, z, t, a, lambda_):
    tau = torch.distributions.Exponential(lambda_).sample().item() # sample waiting time tau from exponential distribution with rate lambda    
    uniform = torch.rand(1).item()  # sample U uniformly from [0, 1]
    if tau >= t:
        i_1 = (1./2.)*tilde_phi(code, phi, psi, f, z+a*t) # compute \frac{1}{2}\tilde{\phi}(code, phi, psi, f, z+a*t)
        i_2 = (1./2.)*tilde_phi(code, phi, psi, f, z-a*t) # compute \frac{1}{2}\tilde{\phi}(code, phi, psi, f, z-a*t)
        i_3 = t * tilde_psi(code, phi, psi, f, z+a*t*(2*uniform-1), a) # compute \tilde{\psi}(code, phi, psi, f, z1+at(2u-1), a)
        return math.exp(lambda_ * t) * (i_1 + i_2 + i_3)
    else:
        gamma_1, new_code_1, new_code_2, i = QC(code, a) # compute the next codes and coefficient using the code algebra
        # Ensure gamma_1 is a tensor on the correct device
        if isinstance(gamma_1, torch.Tensor):
            gamma_1 = gamma_1.clone().to(dtype=torch.complex64)
        else:
            gamma_1 = torch.tensor(gamma_1, dtype=torch.complex64, device=z.device)
        H = gamma_1 * tau * math.exp(lambda_ * tau) / lambda_ # compute the coefficient H for the next codes
        alpha = code[1] # alpha_1 of current code
        if code[0]==1: # if current code is of form \partial^{\alpha}
            H *= (1 + alpha) / (2 * torch.abs(gamma_1)) # multiply by (1+alpha) and divide by abs(gamma_1) to account for the change in the order of derivative and the coefficient from QC
        elif code[0]==2: # if current code is of form \partial^{\alpha} \circ f^{(j)}
            H *= 3 * (1 + alpha)
            if i==1: # if the coordinate in concern is z_1
                H *= ( torch.abs(a)**2 * (2 + alpha) * (3 + alpha) ) / (6 * torch.abs(gamma_1))
        elif code[0]==3: # if current code is of form \partial^{\alpha} \circ ((\partial_t(\cdot))^2)
            H *= 6 * (1 + alpha)
            if i==1: # if the coordinate in concern is z_1
                H *= ( torch.abs(a)**2 * (2 + alpha) * (3 + alpha) ) / (6 * torch.abs(gamma_1))
        elif code[0]==4 or code[0]==5: # if current code is of form \partial^{\alpha} \circ \partial_t or \partial^{\alpha} \circ \partial_{tt}
            H *= (1 + alpha)
        H *= branching1D(new_code_1, phi, psi, f, z+a*tau*(2*uniform-1), t-tau, a, lambda_) # compute the contribution from the next code
        if new_code_2 is not None: # if there is a second next code
            H *= branching1D(new_code_2, phi, psi, f, z+a*tau*(2*uniform-1), t-tau, a, lambda_) # compute the contribution from the second next code
        return H
            
def monte_carlo_simulation(phi, psi, f, z, t, a, lambda_, num_samples=1000):
    # Help me implement a multi-threading version of this function to speed up the simulation
    from concurrent.futures import ThreadPoolExecutor
    code = [0, 0, 0, -1] # start with the code representing the identity operator Id
    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(branching1D, code, phi, psi, f, z, t, a, lambda_) for _ in range(num_samples)]
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

    phi = lambda x: 4 * torch.arctan(torch.exp(2*x/3)) # example initial condition phi(x) = 4 * arctan(exp(2x/3))
    psi = lambda x: (4/3)*(torch.exp(2*x/3) / (1 + torch.exp(4*x/3))) # example initial condition psi(x) = (4/3)*(exp(2x/3) / (1 + exp(4x/3)))
    f = lambda u: -(1/3)*torch.sin(u) # example nonlinearity f(u) = -(1/3)*sin(u)
    z = torch.tensor(1.0 + 0.0j, requires_grad=True, device=device) # complex number with requires_grad=True to enable differentiation, representing the initial spatial point z
    a = torch.tensor(1.0 + 0.0j, dtype=torch.complex64, device=device) # complex number representing the spatial scaling factor
    lambda_ = 1.0 # real number, rate parameter for the exponential distribution governing the waiting times in the branching process
    t_values = torch.arange(0, 1.1, 0.1) # list of t values from 0 to 1 with step 0.1
    real_results = []
    imag_results = []
    num_samples = 100000 # number of Monte Carlo samples to use for each t
    
    # Create directory if it does not exist
    os.makedirs("results", exist_ok=True)
    output_file = "results/monte_carlo.csv"

    # Initialize output file with zero placeholders (2 rows, len(t_values) columns)
    num_t_values = len(t_values)
    with open(output_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([0.0] * num_t_values)
        writer.writerow([0.0] * num_t_values)
    
    for idx, t in enumerate(t_values):
        start_time = time.perf_counter()
        result = monte_carlo_simulation(phi, psi, f, z, t.item(), a, lambda_, num_samples)
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
    