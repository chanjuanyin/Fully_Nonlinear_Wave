import torch
import os
import csv

def u_analytical(z1, z2, z3, t):
    """
    Analytical solution: u(z1, z2, z3, t) = 4 * arctan(exp(2/3 * (z1 + z2 + z3 + t/2)))
    
    Args:
        z1: complex tensor, spatial coordinate
        z2: complex tensor, spatial coordinate
        z3: complex tensor, spatial coordinate
        t: real number, time
    
    Returns:
        complex tensor: value of u at (z1, z2, z3, t)
    """
    return 4 * torch.arctan(torch.exp((2.0/3.0) * (z1 + z2 + z3 + t/2.0)))

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initial spatial point (complex)
    z1 = torch.tensor(1.0 + 0.0j, dtype=torch.complex64, device=device)
    z2 = torch.tensor(1.0 + 0.0j, dtype=torch.complex64, device=device)
    z3 = torch.tensor(1.0 + 0.0j, dtype=torch.complex64, device=device)
    
    # Time values from 0 to 1 with step 0.01
    t_values = torch.arange(0, 1.01, 0.01)
    
    real_results = []
    imag_results = []
    
    # Compute analytical solution for each time value
    for t in t_values:
        result = u_analytical(z1, z2, z3, t.item())
        real_results.append(result.real.item())
        imag_results.append(result.imag.item())
        # print(f"t={t.item():.2f}, Real part: {result.real.item():.6f}, Imaginary part: {result.imag.item():.6f}")
    
    # Create directory if it does not exist
    os.makedirs("results", exist_ok=True)
    
    # Save results to CSV file
    with open("results/analytic.csv", mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(real_results)  # write real parts in the first row
        writer.writerow(imag_results)  # write imaginary parts in the second row
    
    print("\nResults saved to results/analytic.csv")
