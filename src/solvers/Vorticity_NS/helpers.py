import sys
import jax.numpy as jnp
import numpy as np
import jax



def get_energy(omega_hat, mesh):
    psi_hat = omega_hat / mesh.k2

    # energy spectrum
    E_k = 0.5 * (jnp.abs(psi_hat)**2 )
    total_energy = jnp.sum(E_k) * (mesh.L**2 / mesh.N**4) 
    return total_energy

def get_enstrophy(omega_hat, mesh):
    enstrophy_spectrum = 0.5 * jnp.abs(omega_hat)**2
    total_enstrophy = jnp.sum(enstrophy_spectrum) * (mesh.L**2 / mesh.N**4) 
    return total_enstrophy

def get_palinstrophy(omega_hat, mesh):
    omega_x_hat = 1j * mesh.kx * omega_hat
    omega_y_hat = 1j * mesh.ky * omega_hat
    palinstrophy_spectrum = 0.5 * (jnp.abs(omega_x_hat)**2 + jnp.abs(omega_y_hat)**2)
    total_palinstrophy = jnp.sum(palinstrophy_spectrum) * (mesh.L)**2 / (mesh.X.shape[0]**2)
    return total_palinstrophy



def gaussian_vortex(mesh, x0, y0, r0, omega0):
    r2 = (mesh.X - x0)**2 + (mesh.Y - y0)**2
    return omega0 * jnp.exp(-r2 / (2 * r0**2))

def Taylor_green_vortex(mesh, E0=2.0):
    omega =  2 * jnp.sin(mesh.X) * jnp.sin(mesh.Y)
    omega_hat = jnp.fft.fft2(omega)

    E = get_energy(omega_hat, mesh)
    omega_hat = omega_hat *  jnp.sqrt(E0 / E)
    return omega_hat


def dipole_vortex(mesh, E0=2.0):
    omega_e = 300.
    r0 = jnp.pi / 8
    x1, y1 = mesh.L / 2, mesh.L / 2 - 0.1 * mesh.L / 2
    x2, y2 = mesh.L / 2, mesh.L / 2 + 0.1 * mesh.L / 2

    r1 = (mesh.X - x1)**2 + (mesh.Y - y1)**2
    r2 = (mesh.X - x2)**2 + (mesh.Y - y2)**2

    omega1 = -omega_e * (1. - (r1 / r0)**2) * jnp.exp(- (r1 / r0)**2)
    omega2 = omega_e * (1. - (r2 / r0)**2) * jnp.exp(- (r2 / r0)**2)

    omega = omega1 + omega2
    omega_hat = jnp.fft.fft2(omega)

    E = get_energy(omega_hat, mesh)
    omega_hat = omega_hat *  jnp.sqrt(E0 / E)
    return omega_hat

def gaussian_noise(mesh, key, E0=2.0):
    noise_real = jax.random.normal(key, (mesh.N, mesh.N))
    key, subkey = jax.random.split(key)
    noise_imag = jax.random.normal(subkey, (mesh.N, mesh.N))

    omega_hat = (noise_real + 1j * noise_imag) 
    omega_hat =  omega_hat.at[0,0].set(0.0)

    omega = jnp.real(jnp.fft.ifft2(omega_hat))
    omega_hat = jnp.fft.fft2(omega)

    E = get_energy(omega_hat, mesh)
    omega_hat = omega_hat *  jnp.sqrt(E0 / E)  # normalize energy to E0

    return omega_hat  


# r2 = (mesh.X - jnp.pi)**2 + (mesh.Y - jnp.pi)**2
# omega0  = jnp.exp(-r2 / (2 * 0.2**2))
# omega_hat = jnp.fft.fft2(omega0)
 
# psi_hat = - omega_hat / mesh.k2
# psi_hat = psi_hat.at[0,0].set(0.0)
# # velocity in Fourier
# u_hat =  1j * mesh.ky * psi_hat 
# v_hat = -1j * mesh.kx * psi_hat 
# E_uv = jnp.abs(u_hat)**2 + jnp.abs(v_hat)**2
# E_uv = jnp.sum(E_uv) * (L**2 / N**4) 
# print("E_uv:", E_uv)

# E_omega = jnp.abs(omega_hat )**2  / mesh.k2
# E_omega = E_omega.at[0,0].set(0.0)
# E_omega = jnp.sum(E_omega) * (L**2 / N**4)
# print("E_omega:", E_omega)


# # Solve for streamfunction: ψ_hat = -ω_hat / k²
# psi_hat = -omega_hat / mesh.k2
# psi_hat = psi_hat.at[0,0].set(0.0)
# # Compute velocities: u = ∂ψ/∂y, v = -∂ψ/∂x
# u_hat = 1j * mesh.ky * psi_hat
# v_hat = -1j * mesh.kx * psi_hat
# # Transform to physical space
# u = jnp.real(jnp.fft.ifft2(u_hat))
# v = jnp.real(jnp.fft.ifft2(v_hat))
# energy = jnp.sum(u**2 + v**2) * (L**2 / N**2)
# print("energy:", energy)