import jax.numpy as jnp
import jax
import sys
sys.path.append('../../../..')  
from FVM.src.mesh.mesh import Mesh # pyright: ignore[reportMissingImports]
import FVM.src.Cases.Test_Cases as Test_Cases # pyright: ignore[reportMissingImports]
import FVM.src.mesh.Mesh_cases as Mesh_cases # pyright: ignore[reportMissingImports]
import time
import FVM.src.solvers.Euler.helper as Euler_helper # pyright: ignore[reportMissingImports]

"""
Finite Volume Method for 2D Euler equations
With jax.jit compilation = ~ 100x faster for large data
"""
###########################################################################################################
##############################                Solver                 ######################################
###########################################################################################################

def venkatakrishnan(a, b, h = 0, K = 0):
	omega = (K * h)**3 
	L = (a**2 + 2 * a*b + omega) / (a**2 + 2*b**2 + a*b + omega + 1e-09)
	return L

def getlimiting(W_L, W_R, grad, mesh):
	W_m  = jnp.min(jnp.concatenate([W_L, W_R], axis = -2), axis = -2)
	W_M  = jnp.max(jnp.concatenate([W_L, W_R], axis = -2), axis = -2)

	mid_point_faces = jnp.mean(mesh.points[mesh.faces[mesh.face_connectivity]], axis = -2) # (N_cells,3,2)
	delta_x = mid_point_faces - mesh.barycenter[...,None,:]  # (N_cells, 3, 2)
	Delta = jnp.einsum('ijl,ikl->ijk', delta_x, grad)  # (N_cells, 3, N_vars)

	phi = jnp.ones_like(Delta)
	phi = jnp.where(Delta > 1e-8,
					venkatakrishnan(W_M[...,None,:] - W_L, Delta),
					phi)
	phi = jnp.where(Delta < -1e-8,
					venkatakrishnan(W_m[...,None,:] - W_L, Delta),
					phi)
	phi = jnp.min(phi, axis = -2)  # (N_cells, N_vars)
	return phi

def MUSCL(W_L, W_R, grad, mesh):
	phi = getlimiting(W_L, W_R, grad, mesh)  # (N_cells, N_vars)

	mid_point_faces = jnp.mean(mesh.points[mesh.faces[mesh.face_connectivity]], axis = -2) # (N_cells,3,2)
	
	delta_x = mid_point_faces - mesh.barycenter[...,None,:]  # (N_cells, 3, 2)
	Delta = jnp.einsum('ijl,ikl->ijk', delta_x, grad)  # (N_cells, 3, N_vars)

	delta_x_neigh = mid_point_faces - mesh.barycenter[mesh.neighbors]  # (N_cells, 3, 2)
	delta_x_neigh = jnp.where(jnp.repeat((mesh.face_markers[mesh.face_connectivity] > 0)[...,None], 2, axis=-1), -delta_x, delta_x_neigh) # Boundary faces: reverse the direction
	Delta_neigh = jnp.einsum('ijl,ijkl->ijk', delta_x_neigh, grad[mesh.neighbors])


	W_L_MUSCL = W_L + phi[...,None,:] * Delta  # (N_cells, 3, N_vars)
	W_R_MUSCL = W_R + phi[mesh.neighbors] * Delta_neigh  # (N_cells, 3, N_vars)

	return W_L_MUSCL, W_R_MUSCL

def getFlux(W_L, W_R, normals, surfaces, gamma = 1.4, alpha = 1.):
	# i did not put mesh as input in order to vmap this function
	# Get the cell state for each edge
	rho_L = W_L[...,0]
	mom_x_L = W_L[...,1]
	mom_y_L = W_L[...,2]
	E_L = W_L[...,3]
	u_L = mom_x_L / rho_L
	v_L = mom_y_L / rho_L
	P_L = (gamma - 1) * (E_L - 0.5 * rho_L * (u_L**2 + v_L**2))

	# Get the corresponding neighbors state
	rho_R = W_R[...,0]
	mom_x_R = W_R[...,1]
	mom_y_R = W_R[...,2]
	E_R = W_R[...,3]
	u_R = mom_x_R / rho_R
	v_R = mom_y_R / rho_R
	P_R = (gamma - 1) * (E_R - 0.5 * rho_R * (u_R**2 + v_R**2))

	# Get corresponding normals
	nx = normals[...,0]
	ny = normals[...,1]

    # Maximum wavelenghts
	C_L = jnp.sqrt(jnp.abs(gamma*P_L/rho_L))  + jnp.abs(u_L * nx + v_L * ny)
	C_R = jnp.sqrt(jnp.abs(gamma*P_R/rho_R))  + jnp.abs(u_R * nx + v_R * ny)
	C_max = jnp.maximum(C_R, C_L)

	# Energy
	en_L = P_L/(gamma-1) + 0.5 * rho_L * (u_L**2 + v_L**2)
	en_R = P_R/(gamma-1) + 0.5 * rho_R * (u_R**2 + v_R**2)

	# Flux
	flux_rho_L = rho_L * (u_L * nx + v_L * ny)
	flux_ru_L = rho_L * u_L* ( u_L * nx + v_L * ny) + P_L * nx
	flux_rv_L = rho_L * v_L * (u_L * nx + v_L * ny) + P_L * ny
	flux_E_L = (en_L + P_L) * (u_L * nx + v_L * ny)

	flux_rho_R = rho_R * (u_R * nx + v_R * ny)
	flux_ru_R = rho_R * u_R * ( u_R * nx + v_R * ny) + P_R * nx
	flux_rv_R = rho_R * v_R * (u_R * nx + v_R * ny) + P_R * ny
	flux_E_R = (en_R + P_R) * (u_R * nx + v_R * ny)

	# Total flux
	# local_mach = jnp.maximum(jnp.abs(u_L)/jnp.maximum(C_L, 1e-09), jnp.abs(u_R)/jnp.maximum(C_R, 1e-09))
	# alpha = jnp.sin(jnp.pi * local_mach /2)  # andea s damping 2024
	# alpha = jnp.where(local_mach >= 1, 1.0, alpha)
	# alpha = 0.1

	flux_rho = (flux_rho_L + flux_rho_R)/2 - alpha * C_max * 0.5 * (rho_R - rho_L)
	flux_ru = (flux_ru_L + flux_ru_R)/2 - alpha * C_max * 0.5 * (rho_R * u_R - rho_L * u_L)
	flux_rv = (flux_rv_L + flux_rv_R)/2 - alpha * C_max * 0.5 * (rho_R * v_R - rho_L * v_L)
	flux_E = (flux_E_L + flux_E_R)/2 - alpha * C_max * 0.5 * (en_R - en_L)

	Flux = jnp.stack([surfaces * flux_rho, 
						surfaces * flux_ru, 
						surfaces * flux_rv, 
						surfaces * flux_E], axis = -1)

	Flux = jnp.sum(Flux, axis = -2)
	return Flux


###########################################################################################################
############################           Time integration                 ###################################
###########################################################################################################

@jax.jit(static_argnums=(1,))
def get_dt(W, mesh, CFL = 0.5):
	Primitives = Euler_helper.getPrimitive(W)
	c = jnp.sqrt(1.4 * Primitives[...,3] / Primitives[...,0])
	lambda_max = c[...,None] + jnp.abs(jnp.sum(jnp.repeat(W[...,None,:], 3, axis=-2)[...,1:3] * mesh.normals, axis = -1))
	dt_unstr = mesh.area / jnp.sum(lambda_max * mesh.surface[mesh.face_connectivity], axis = -1)
	return jnp.min(dt_unstr) * CFL

@jax.jit(static_argnums=(1,))
def time_step(W, mesh, dt, **kwargs):
	# 1st order
	W_L = jnp.repeat(W[...,None,:], 3, axis=-2)
	W_R = W[mesh.neighbors]

	# 2nd order - MUSCL with least-square gradient
	W_R = Euler_helper.BC_state(W_R, W_L, mesh)
	grad = Euler_helper.getgradientLSQ(W_L, W_R, mesh)

	W_L, W_R = MUSCL(W_L, W_R, grad, mesh)
	W_R = Euler_helper.BC_state(W_R, W_L, mesh)

	Flux = getFlux(W_L, W_R, mesh.normals, mesh.surface[mesh.face_connectivity], 
				gamma = 1.4, alpha = kwargs.get('alpha', 1.)) 
		
	W = W - dt / mesh.area[...,None] * (Flux) 
	return W

@jax.jit(static_argnums=(1,))
def residual(W, mesh, **kwargs):
	# 1st order
	W_L = jnp.repeat(W[...,None,:], 3, axis=-2)
	W_R = W[mesh.neighbors]

	# 2nd order - MUSCL with least-square gradient
	W_R = Euler_helper.BC_state(W_R, W_L, mesh)
	grad = Euler_helper.getgradientLSQ(W_L, W_R, mesh)


	W_L, W_R = MUSCL(W_L, W_R, grad, mesh)
	W_R = Euler_helper.BC_state(W_R, W_L, mesh)

	Flux = getFlux(W_L, W_R, mesh.normals, mesh.surface[mesh.face_connectivity], gamma = 1.4, alpha = kwargs.get('alpha', 1.)) 
	# Flux = jax.vmap(getFlux, 
	# 			in_axes=(0,0,0,0,None,None))(W_L, W_R, mesh.normals, mesh.surface[mesh.face_connectivity], 1.4, kwargs.get('alpha', 1.))
	
	return Flux / mesh.area[...,None] 

@jax.jit(static_argnums=(1,))
def time_step_RK2(W, mesh, dt, **kwargs):
	F1 = residual(W, mesh, **kwargs)
	W1 = W - dt/2 * F1
	F2 = residual(W1, mesh, **kwargs)
	W = W - dt * (F1 + F2)
	return W

@jax.jit(static_argnums=(1,))
def time_step_RK4(W, mesh, dt, **kwargs):
	F1 = residual(W, mesh, **kwargs)
	W1 = W - dt/2 * F1
	F2 = residual(W1, mesh, **kwargs)
	W2 = W - dt/2 * F2
	F3 = residual(W2, mesh, **kwargs)
	W3 = W - dt * F3
	F4 = residual(W3, mesh, **kwargs)

	W = W - dt/6 * (F1 + 2*F2 + 2*F3 + F4)
	return W


@jax.jit(static_argnums=(1,))
def times_step_Newton(W, mesh, dt, **kwargs):
	# linear newton iterations = 1
	Fval = dt * residual(W, mesh, **kwargs) 
	# Implicit form of the residual, solving Jv = v + dt * Jv = -Fval
	def Jv(v):
		_, jvp = jax.jvp(lambda x: residual(x, mesh, **kwargs),
					(W,),
					(v,))
		return v + dt * jvp
	
	# no preconditioner, just solve the linear system with gmres
	delta, _ = jax.scipy.sparse.linalg.gmres(Jv, -Fval, maxiter=20)
	W = W + delta
	return W

@jax.jit(static_argnums=(1,))
def SDIKR2(W, mesh, dt, **kwargs):
	x = 1 - 1/jnp.sqrt(2) # singly diagonally implicit RK

	# first step
	W1 = times_step_Newton(W, mesh, dt * x, **kwargs)
	F1 = residual(W1, mesh, **kwargs)

	# second step
	Fval = W1 - W + dt * (x * residual(W1, mesh, **kwargs) + (1-2 * x) * F1)
	def Jv(v):
		_, jvp = jax.jvp(lambda x: residual(x, mesh, **kwargs),
					(W1,),
					(v,))
		return v + dt * x * jvp
	
	# no preconditioner, just solve the linear system with gmres
	delta, _ = jax.scipy.sparse.linalg.gmres(Jv, -Fval, maxiter=20)
	W2 = W1 + delta
	F2 = residual(W2, mesh, **kwargs)

	W = W - 0.5 * dt * (F1 + F2) 
	return W



if __name__ == "__main__":
	mesh = Mesh_cases.TestDipoleVortex().build(h = 5e-5, L = 1.)

	# Initial condition
	Primitives, mesh = Test_Cases.TestDipoleVortex2(R = 0.1, omega = 300, mach = 0.01).build(mesh)
	W = Euler_helper.getConserved(Primitives)
	mesh.plot_mesh()

	# Time loop
	t_final = 0.2 #/ jnp.mean(Primitives[...,1]) # to get real time
	CFL = 15
	dt = get_dt(W, mesh, CFL = CFL)
	N_t = int(t_final / dt) + 1

	start_time = time.time()

	# E = []
	# Enstrophy = []
	t = 0
	n = 0
	for n in range(1000):
		# W = time_step_RK2(W, mesh, dt, alpha = 1.)
		# W = times_step_Newton(W, mesh, dt, alpha = 1.)
		W = SDIKR2(W, mesh, dt, alpha = 1.)
		if n % 100 == 0:
			# print(f'time: {t:.4f} / {t_final:.3f} seconds')
			print(f'It : {n} / {N_t}')
		# if n % 1000 == 0:
		# 	Prim = Euler_helper.getPrimitive(W)
		# 	energy = 0.5 * jnp.sum((Prim[...,1]**2 + Prim[...,2]**2) * mesh.area)
		# 	E.append(energy)
		# 	Enstrophy.append(jnp.sum(vorticity**2* mesh.area) )

	print(f'Simulation time: {time.time() - start_time} seconds')

	# Plot solution
	Primitives = Euler_helper.getPrimitive(W)
	mesh.plot_solution(Primitives[...,0], labels = r'$\rho$')
	mesh.plot_solution(Primitives[...,1], labels = r'$u$')
	mesh.plot_solution(Primitives[...,2], labels = r'$v$')
	mesh.plot_solution(Primitives[...,3], labels = r'$P$')

	# vorticity
	vorticity = Euler_helper.get_vorticity_from_field(W, mesh)
	mesh.plot_solution(vorticity, labels = r'$\omega$')
