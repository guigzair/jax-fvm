import jax.numpy as jnp
import jax
import sys

sys.path.append('../../../..')
from jax_fvm.src.mesh.mesh import Mesh # pyright: ignore[reportMissingImports]
import jax_fvm.src.mesh.plot as plot # pyright: ignore[reportMissingImports]
import jax_fvm.src.Cases.Test_Cases as Test_Cases # pyright: ignore[reportMissingImports]
import jax_fvm.src.mesh.Mesh_cases as Mesh_cases # pyright: ignore[reportMissingImports]
import time
import jax_fvm.src.solvers.helper as helper # pyright: ignore[reportMissingImports]
import jax_fvm.src.solvers.Euler.Euler as Euler # pyright: ignore[reportMissingImports]
import matplotlib.pyplot as plt
# jax.config.update('jax_enable_x64', True)
# jax.config.update("jax_debug_nans", True)
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

"""
Finite Volume Method for 2D incompressible Navier-Stokes equations.

Chorin projection (fractional-step) method, same shape as the Euler solver:
    1. predictor  : advance momentum with convection + viscosity (no pressure)
    2. projection : solve the pressure Poisson equation  lap(p) = (rho/dt) div(V*)
    3. corrector  : V = V* - (dt/rho) grad(p)   (makes V divergence-free)

State is the primitive velocity  V = [u, v]  at cell centers plus a scalar
pressure p at cell centers (constant density rho, kinematic viscosity nu = mu/rho).
"""
###########################################################################################################
##############################                Fluxes                 ######################################
###########################################################################################################

def getConvectiveFlux(V_L, V_R, normals, **kwargs):
	# Incompressible momentum advection with Rusanov (local Lax-Friedrichs) dissipation.
	# V_L, V_R : (N_cells, 3, 2) reconstructed face states ; normals : (N_cells, 3, 2)
	nx = normals[...,0]
	ny = normals[...,1]

	un_L = V_L[...,0] * nx + V_L[...,1] * ny  # face-normal velocity (N_cells, 3)
	un_R = V_R[...,0] * nx + V_R[...,1] * ny

	# Central momentum flux  F = V (V . n)
	central = 0.5 * (V_L * un_L[...,None] + V_R * un_R[...,None])  # (N_cells, 3, 2)

	# Upwind dissipation : advection wave speed is |V . n| (no acoustic term)
	C_max = jnp.maximum(jnp.abs(un_L), jnp.abs(un_R))  # (N_cells, 3)
	diss = 0.5 * C_max[...,None] * (V_R - V_L)

	return central - diss  # (N_cells, 3, 2)

def getViscousFlux(grad_V, mesh, **kwargs):
	# Newtonian viscous stress for a divergence-free flow (drop the -2/3 div(u) term).
	# grad_V : (N_cells, 2, 2) cell gradient, grad_V[cell, var, spatial]
	mu = kwargs.get('mu', 1e-2)

	# Face-averaged gradients (left cell / neighbor cell)
	grad_face = 0.5 * (grad_V[:, None, :, :] + grad_V[mesh.neighbors])  # (N_cells, 3, 2, 2)
	du_dx = grad_face[..., 0, 0]
	du_dy = grad_face[..., 0, 1]
	dv_dx = grad_face[..., 1, 0]
	dv_dy = grad_face[..., 1, 1]

	tau_xx = mu * 2.0 * du_dx
	tau_yy = mu * 2.0 * dv_dy
	tau_xy = mu * (du_dy + dv_dx)

	nx = mesh.normals[..., 0]
	ny = mesh.normals[..., 1]

	f_mom_x = tau_xx * nx + tau_xy * ny
	f_mom_y = tau_xy * nx + tau_yy * ny

	return jnp.stack([f_mom_x, f_mom_y], axis=-1)  # (N_cells, 3, 2)

###########################################################################################################
##############################          Differential operators          ###################################
###########################################################################################################

def divergence(F, mesh):
	# Green-Gauss cell divergence of a vector field F : (N_cells, 2) -> (N_cells,)
	# Periodic faces (marker 1) use the correct opposite cell through mesh.neighbors.
	F_L = jnp.repeat(F[..., None, :], 3, axis=-2)  # (N_cells, 3, 2)
	F_R = F[mesh.neighbors]                          # (N_cells, 3, 2)

	nx = mesh.normals[..., 0]
	ny = mesh.normals[..., 1]
	fn_face = 0.5 * ((F_L[..., 0] + F_R[..., 0]) * nx + (F_L[..., 1] + F_R[..., 1]) * ny)  # (N_cells, 3)

	surfaces = mesh.surface[mesh.face_connectivity]
	return jnp.sum(fn_face * surfaces, axis=-1) / mesh.area

def cell_gradient_scalar(p, mesh):
	# Least-squares cell-centered gradient of a scalar p : (N_cells,) -> (N_cells, 2)
	p_L = jnp.repeat(p[..., None, None], 3, axis=-2)  # (N_cells, 3, 1)
	p_R = p[mesh.neighbors][..., None]                 # (N_cells, 3, 1)
	p_R = apply_pressure_BC(p_R, p_L, mesh)
	grad_p = helper.getgradientLSQ(p_L, p_R, mesh)     # (N_cells, 1, 2)
	return grad_p[:, 0, :]                              # (N_cells, 2)

def laplacian(p, mesh):
	# Discrete Laplacian defined as  div(grad(p))  using EXACTLY the same divergence and
	# cell gradient as the RHS and the corrector -> makes the projection discretely exact.
	return divergence(cell_gradient_scalar(p, mesh), mesh)

###########################################################################################################
##############################          Boundary conditions             ###################################
###########################################################################################################

def apply_velocity_BC(V_R, V_L, mesh):
	# marker 1 (periodic) : handled by mesh.neighbors -> nothing to do.
	# marker 2 (no-slip wall) : ghost velocity = - interior velocity.
	is_wall = jnp.repeat((mesh.face_markers[mesh.face_connectivity] == 2)[..., None], 2, axis=-1)
	return jnp.where(is_wall, -V_L, V_R)

def apply_pressure_BC(p_R, p_L, mesh):
	# marker 2 (wall) : homogeneous Neumann  dp/dn = 0  -> ghost pressure = interior pressure.
	is_wall = (mesh.face_markers[mesh.face_connectivity] == 2)[..., None]
	return jnp.where(is_wall, p_L, p_R)

###########################################################################################################
############################           Fractional step                  ###################################
###########################################################################################################

@jax.jit(static_argnums=(1,))
def predictor(V, mesh, dt, **kwargs):
	# 1st order face states
	V_L = jnp.repeat(V[..., None, :], 3, axis=-2)  # (N_cells, 3, 2)
	V_R = V[mesh.neighbors]                          # (N_cells, 3, 2)
	V_R = apply_velocity_BC(V_R, V_L, mesh)

	# 2nd order MUSCL reconstruction with least-square gradient
	grad_V = helper.getgradientLSQ(V_L, V_R, mesh)  # (N_cells, 2, 2)
	V_L, V_R = Euler.MUSCL(V_L, V_R, grad_V, mesh)

	Flux_conv = getConvectiveFlux(V_L, V_R, mesh.normals, **kwargs)  # (N_cells, 3, 2)
	Flux_visc = getViscousFlux(grad_V, mesh, **kwargs)               # (N_cells, 3, 2)

	Flux = mesh.surface[mesh.face_connectivity][..., None] * (Flux_conv - Flux_visc)
	Res = jnp.sum(Flux, axis=-2) / mesh.area[..., None]  # (N_cells, 2)

	return V - dt * Res

@jax.jit(static_argnums=(1,))
def pressure_poisson(V_star, mesh, dt, **kwargs):
	rho = kwargs.get('rho', 1.0)

	b = (rho / dt) * divergence(V_star, mesh)  # (N_cells,)
	b = b - jnp.mean(b)  # remove the constant nullspace component (compatibility)

	Lp = lambda p: laplacian(p, mesh)
	p, _ = jax.scipy.sparse.linalg.cg(Lp, b, tol=1e-6, maxiter=500)

	return p - jnp.mean(p)  # pin the pressure mean (defined up to a constant)

@jax.jit(static_argnums=(2,))
def corrector(V_star, p, mesh, dt, **kwargs):
	rho = kwargs.get('rho', 1.0)
	grad_p = cell_gradient_scalar(p, mesh)  # (N_cells, 2)
	return V_star - (dt / rho) * grad_p

@jax.jit(static_argnums=(2,))
def time_step_projection(V, p, mesh, dt, **kwargs):
	V_star = predictor(V, mesh, dt, **kwargs)
	p_new = pressure_poisson(V_star, mesh, dt, **kwargs)
	V_new = corrector(V_star, p_new, mesh, dt, **kwargs)
	return V_new, p_new

###########################################################################################################
############################              Diagnostics                   ###################################
###########################################################################################################

def get_dt(V, mesh, CFL=0.5):
	speed = jnp.sqrt(V[..., 0]**2 + V[..., 1]**2)
	dx = mesh.area / jnp.sum(mesh.surface[mesh.face_connectivity], axis=-1)
	return jnp.min(dx / (speed + 1e-12)) * CFL

def get_velocity_gradient(V, mesh):
	V_L = jnp.repeat(V[..., None, :], 3, axis=-2)
	V_R = V[mesh.neighbors]
	V_R = apply_velocity_BC(V_R, V_L, mesh)
	return helper.getgradientLSQ(V_L, V_R, mesh)  # (N_cells, 2, 2), grad[cell, var, spatial]

def get_vorticity(V, mesh):
	grad_V = get_velocity_gradient(V, mesh)
	return grad_V[:, 1, 0] - grad_V[:, 0, 1]  # dv/dx - du/dy

def total_kinetic_energy(V, mesh):
	return jnp.sum(0.5 * (V[..., 0]**2 + V[..., 1]**2) * mesh.area)


if __name__ == "__main__":
	# Taylor-Green vortex on a periodic square.
	# NOTE: UniformMesh hardwires its interior points to the unit box, so keep Lx=Ly=1.
	# TGV analytic decay : KE(t) = KE(0) exp(-4 nu k^2 t) with wavenumber k = 2 pi / L.
	L = 1.0
	k = 2 * jnp.pi / L
	mesh = Mesh_cases.UniformMesh(Nx=64, Ny=64, Lx=L, Ly=L).build()

	rho = 1.0
	nu = 1e-2
	kwargs = {'rho': rho, 'nu': nu, 'mu': rho * nu, 'alpha': 1.0}

	Primitives = Test_Cases.TaylorGreenVortex(V_0=1., M=0.1).build(mesh)
	V = Primitives[..., 1:3]           # [u, v]
	p = jnp.zeros(mesh.area.shape[0])  # initial pressure

	mesh.plot_mesh()

	# Time loop
	t_final = 2.0
	CFL = 0.4
	dt = jnp.minimum(get_dt(V, mesh, CFL=CFL), helper.get_dt_viscous(mesh, CFL=CFL, nu=nu))
	N_t = int(t_final / dt) + 1
	print(f"dt = {dt:.3e}, nsteps = {N_t}")

	start_time = time.time()

	T_interval_snapshots = 20
	KE0 = total_kinetic_energy(V, mesh)
	times, KE_rec = [], []
	for n in range(N_t):
		V, p = time_step_projection(V, p, mesh, dt, **kwargs)
		if n % 100 == 0:
			div_max = jnp.max(jnp.abs(divergence(V, mesh)))
			print(f'It : {n} / {N_t} | max|div V| = {div_max:.2e}')
			if jnp.isnan(V).any():
				print(f'NaN detected at iteration {n}')
				break
		if n % T_interval_snapshots == 0:
			times.append(n * dt)
			KE_rec.append(total_kinetic_energy(V, mesh))

	print(f'Simulation time: {time.time() - start_time} seconds')

	# Plot solution
	mesh.plot_solution(V[..., 0], labels=r'$u$')
	mesh.plot_solution(V[..., 1], labels=r'$v$')
	mesh.plot_solution(p, labels=r'$p$')

	# vorticity
	vorticity = get_vorticity(V, mesh)
	mesh.plot_solution(vorticity, labels=r'$\omega$')

	# kinetic-energy decay vs analytic  KE(0) exp(-4 nu t)
	times = jnp.array(times)
	KE_rec = jnp.array(KE_rec)
	fig, ax = plt.subplots()
	ax.plot(times, KE_rec, 'o-', label=r'FVM')
	ax.plot(times, KE0 * jnp.exp(-4 * nu * k**2 * times), '--', label=r'$KE_0\,e^{-4\nu k^2 t}$')
	ax.set_xlabel(r't')
	ax.set_ylabel(r'$E$')
	ax.legend()
	ax.grid()
	plt.show()
