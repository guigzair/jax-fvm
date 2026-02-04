import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import mesh as mesh_vort
import helpers as helpers
size = 14
params = {
    'text.usetex': True,
    'font.family': 'serif',
    'font.serif': 'cm',  # Computer Modern font
	'legend.fontsize':size,
    'axes.labelsize' : size,
	'axes.titlesize' : size +2,
    'xtick.labelsize' : size+1,
    'ytick.labelsize' : size+1
}
plt.rcParams.update(params)

# -----------------------------
# Parameters
# -----------------------------
N = 512
L = 2 * jnp.pi
nu = 1e-2

# -----------------------------
# RHS
# -----------------------------

def get_velocities(omega_hat, mesh):
    psi_hat = + omega_hat / mesh.k2
    psi_hat = psi_hat.at[0,0].set(0.0)

    # velocity in Fourier
    u_hat =  1j * mesh.ky * psi_hat
    v_hat = -1j * mesh.kx * psi_hat

    # back to real
    u = jnp.real(jnp.fft.ifft2(u_hat))
    v = jnp.real(jnp.fft.ifft2(v_hat))

    return u, v

# @jax.jit
def rhs(omega_hat, mesh):
    omega_hat = omega_hat.at[0,0].set(0.)
    # stream function
    u, v = get_velocities(omega_hat * mesh.dealias, mesh)

    # vorticity gradients
    omega_x = jnp.real(jnp.fft.ifft2(1j * mesh.kx * omega_hat))
    omega_y = jnp.real(jnp.fft.ifft2(1j * mesh.ky * omega_hat))

    # nonlinear term
    adv = u * omega_x + v * omega_y
    adv_hat = jnp.fft.fft2(adv) * mesh.dealias

    # diffusion
    diff = nu * mesh.k2 * omega_hat

    # hyperviscosity
    # alpha = 3.75e-4 * jnp.where(mesh.k2 <= 2, 0.0, 1.)
    # psi_hat = + omega_hat / mesh.k2
    # psi_hat = psi_hat.at[0,0].set(0.0)
    # u_hat = alpha * 1j * mesh.ky * psi_hat

    return - adv_hat - diff 


@jax.jit(static_argnums=(1,))
def step(omega_hat, mesh, dt):
    k1 = rhs(omega_hat, mesh)
    omega_hat = omega_hat + dt * k1 

    Energy = helpers.get_energy(omega_hat, mesh)
    Enstrophy = helpers.get_enstrophy(omega_hat, mesh)
    Palinstrophy = helpers.get_palinstrophy(omega_hat, mesh)
    return omega_hat, (Energy, Enstrophy, Palinstrophy)



if __name__ == "__main__":
    # -----------------------------
    # Mesh
    # -----------------------------
    mesh = mesh_vort.Mesh()
    mesh.mesh_generator(N=N, L=L)

    # omega_hat = helpers.gaussian_noise(mesh, jax.random.PRNGKey(0), E0=2.0)
    # omega_hat = helpers.Taylor_green_vortex(mesh)
    omega_hat = helpers.dipole_vortex(mesh, E0=2.0)


    u, v = get_velocities(omega_hat, mesh)
    dt = 0.03 * (L / N) / jnp.max(jnp.sqrt(u**2 + v**2))
    T = 1.
    N_t = int(T / dt)
    print(f"dt = {dt:.3e}, nsteps = {N_t}")
    E = []
    Eta = []
    P = []
    for n in range(N_t):
        omega_hat, (Energy, Enstrophy, Palinstrophy) = step(omega_hat, mesh, dt)
        if n%50 == 0:
            omega_hat = omega_hat * mesh.dealias
        if n % 100 == 0:
            print(f"Step {n}/{N_t}")
        E.append(Energy)
        Eta.append(Enstrophy)
        P.append(Palinstrophy)
            

    omega = jnp.real(jnp.fft.ifft2(omega_hat))
    u, v = get_velocities(omega_hat, mesh)

    mesh.plot_field(omega)
    mesh.plot_field(u, clb_title=r"$u$")
    mesh.plot_field(v, clb_title=r"$v$")

    fig, ax = plt.subplots()
    ax.plot(jnp.arange(N_t) * dt, E, label=r"Energy")
    # ax.plot(jnp.arange(N_t) * dt, E[0] * jnp.exp(-4 * nu * jnp.arange(N_t) * dt), label=r"Energy")
    ax.set_xlabel("Time")
    ax.legend()


