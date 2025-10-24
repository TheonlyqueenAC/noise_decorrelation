import numpy as np
import matplotlib.pyplot as plt

# Constants
hbar = 1.0  # Reduced Planck's constant
m = 1.0  # Effective mass
L = 10.0  # Axial length of microtubule
R_inner = 7.0  # Inner radius of microtubule
R_outer = 12.5  # Outer radius of microtubule
N_r = 100  # Radial grid points
N_z = 100  # Axial grid points
dr = (R_outer - R_inner) / N_r  # Radial step size
dz = L / N_z  # Axial step size
dt = 0.001  # Time step
time_steps = 300  # Total simulation steps

# Cytokine parameters
V_0 = 5.0  # Peak cytokine potential
Gamma_0 = 0.05  # Baseline decoherence rate
alpha_c = 0.1  # Scaling factor for cytokine-induced decoherence
D_c = 0.1  # Cytokine diffusion coefficient
kappa_c = 0.01  # Cytokine clearance rate

# Spatial grids
r = np.linspace(R_inner, R_outer, N_r)
z = np.linspace(0, L, N_z)
R, Z = np.meshgrid(r, z)

# Initial cytokine concentration field
C = np.exp(-((Z - L / 2) ** 2) / (2 * (L / 10) ** 2)) * np.exp(-((R - R_outer) ** 2) / (2 * (R_outer - R_inner) ** 2))

# Initialize wavefunction
Psi = np.exp(-0.5 * ((Z - L / 2) / (L / 10)) ** 2) * (R_outer - R)
Psi /= np.sqrt(np.sum(np.abs(Psi) ** 2))  # Normalize

# Initialize cytokine-dependent potential
V_cytokine = V_0 * C


# Evolve cytokine concentration
def evolve_cytokines(C, dr, dz, dt, D_c, kappa_c):
    laplacian_r = (np.roll(C, -1, axis=0) - 2 * C + np.roll(C, 1, axis=0)) / dr ** 2
    laplacian_z = (np.roll(C, -1, axis=1) - 2 * C + np.roll(C, 1, axis=1)) / dz ** 2
    return C + dt * (D_c * (laplacian_r + laplacian_z) - kappa_c * C)


# Evolve wavefunction with decoherence
def evolve_wavefunction(Psi, V, Gamma, dr, dz, dt):
    laplacian_r = (np.roll(Psi, -1, axis=0) - 2 * Psi + np.roll(Psi, 1, axis=0)) / dr ** 2
    laplacian_z = (np.roll(Psi, -1, axis=1) - 2 * Psi + np.roll(Psi, 1, axis=1)) / dz ** 2
    Psi_new = Psi - (1j * hbar * dt / (2 * m)) * (laplacian_r + laplacian_z + V * Psi) - Gamma * Psi * dt
    return Psi_new


# Time evolution loop with fixes
for step in range(time_steps):
    # Update cytokine concentration
    C = evolve_cytokines(C, dr, dz, dt, D_c, kappa_c)
    C = np.clip(C, 0, 1)  # Prevent unbounded growth

    # Update cytokine-dependent potential and decoherence
    V_cytokine = V_0 * C
    Gamma_cytokine = Gamma_0 * (1 + alpha_c * C)

    # Evolve wavefunction
    Psi = evolve_wavefunction(Psi, V_cytokine, Gamma_cytokine, dr, dz, dt)
    Psi = np.clip(Psi, -1e10, 1e10)  # Prevent overflow

    # Normalize wavefunction
    norm_factor = np.sqrt(np.sum(np.abs(Psi)**2 * r[:, None]) * dr * dz)
    Psi /= norm_factor

    # Store probability density
Psi_list = []  # Initialize list to store probability densities
Psi_list.append(np.abs(Psi)**2)
# Prevent overflow during computation
Psi = np.clip(Psi, -1e10, 1e10)  # Adjust thresholds as necessary
# Clamp cytokine concentration to biologically realistic values
C = np.clip(C, 0, 1)  # Example: max concentration is normalized to 1

# Visualization with adjusted limits
plt.figure(figsize=(10, 6))
plt.contourf(Z, R, Psi_list[-1], levels=50, cmap='viridis', vmin=0, vmax=np.max(Psi_list[-1]) * 0.1)
plt.colorbar(label='|Ψ|^2')
plt.title('Final Wavefunction Probability Density |Ψ|^2')
plt.xlabel('Axial Position (z)')
plt.ylabel('Radial Position (r)')
plt.show()

