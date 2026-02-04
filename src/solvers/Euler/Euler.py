import jax.numpy as jnp
import jax
import sys
sys.path.append('../../../..')  
from FVM.src.mesh.mesh import Mesh
import FVM.src.Cases.Test_Cases as Test_Cases
import FVM.src.mesh.Mesh_cases as Mesh_cases
import time
import FVM.src.solvers.Euler.helper as Euler_helper

"""
Finite Volume Method for 2D Euler equations
With jax.jit compilation = ~ 100x faster for large data
"""

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

def BC_outflow(W_R, W_L, mesh, bc_type = 4):
	W_R = jnp.where(jnp.repeat((mesh.face_markers[mesh.face_connectivity] == bc_type)[...,None], 4, axis=-1), W_L, W_R)
	return W_R	

def BC_inflow(W, mesh, bc_type = 3, value = jnp.array([1.0, 1.0, 1.0, 1.0])):
	W = jnp.where(jnp.repeat((mesh.face_markers[mesh.face_connectivity] == bc_type)[...,None], 4, axis=-1), value, W)
	return W	

def BC_subsonic_inlet(W_R, W_L, mesh, bc_type = 5):
	Prim_L = Euler_helper.getPrimitive(W_L)
	Prim_b = Prim_L.at[...,:3].set(mesh.inlet_subsonic[...,:3])
	
	rho = Prim_b[...,0]
	u = Prim_b[...,1]
	v = Prim_b[...,2]
	P = Prim_b[...,3]
	Mass  = rho
	Mom_x = rho * u 
	Mom_y = rho * v 
	Energy = P/(1.4-1) + rho*(u**2 + v**2)
	W_b = jnp.stack([Mass, Mom_x, Mom_y, Energy], axis = -1)

	W_R = jnp.where(jnp.repeat((mesh.face_markers[mesh.face_connectivity] == bc_type)[...,None], 4, axis=-1), W_b, W_R)
	return W_R


def BC_slipwall(W_R, W_L, mesh, bc_type = 2, value = jnp.array([0., 0., 0., 0.])):
	# value is a background flow to subtract
	Prim_L = Euler_helper.getPrimitive(W_L)
	vn = (Prim_L[...,1] - value[1]) * mesh.normals[...,0] + (Prim_L[...,2] - value[2]) * mesh.normals[...,1]
	vb = (Prim_L[...,1:3] - value[1:3]) - 2 * vn[...,None] * mesh.normals
	Prim_b = Prim_L.at[...,1:3].set(vb + value[1:3])
	W_b = Euler_helper.getConserved(Prim_b)
	W_R = jnp.where(jnp.repeat((mesh.face_markers[mesh.face_connectivity] == bc_type)[...,None], 4, axis=-1), W_b, W_R)
	return W_R	

def BC_state(W_R, W_L, mesh, **kwargs):
	# value = kwargs.get('value', jnp.array([1.0, 1.0, 1.0, 1.0]))
	W_R = BC_slipwall(W_R, W_L, mesh, bc_type=2)  # (slip-wall)
	W_R = BC_inflow(W_R, mesh, bc_type=3, value = Euler_helper.getConserved(mesh.inlet_subsonic))  # (supersonic inlet)
	W_R = BC_outflow(W_R, W_L, mesh, bc_type=4)  # (free outflow)
	W_R = BC_subsonic_inlet(W_R, W_L, mesh, bc_type=5)  # (subsonic inlet)
	return W_R

def getFlux(W_L, W_R, mesh, gamma = 1.4, alpha = 1.):
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
	nx = mesh.normals[...,0]
	ny = mesh.normals[...,1]

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

	surfaces = mesh.surface[mesh.face_connectivity]
	Flux = jnp.stack([surfaces * flux_rho, 
						surfaces * flux_ru, 
						surfaces * flux_rv, 
						surfaces * flux_E], axis = -1)

	Flux = jnp.sum(Flux, axis = -2)
	return Flux


@jax.jit(static_argnums=(1,))
def time_step(W, mesh, dt, alpha = 1.):
	# 1st order
	W_L = jnp.repeat(W[...,None,:], 3, axis=-2)
	W_R = W[mesh.neighbors]

	# 2nd order - MUSCL with least-square gradient
	W_R = BC_state(W_R, W_L, mesh)
	grad = Euler_helper.getgradientLSQ(W_L, W_R, mesh)

	# # Vorticity
	# grad_Prim = Euler_helper.getgradientLSQ(Euler_helper.getPrimitive(W_L), Euler_helper.getPrimitive(W_R), mesh)
	# vorticity = Euler_helper.get_vorticity(grad_Prim)	

	W_L, W_R = MUSCL(W_L, W_R, grad, mesh)
	W_R = BC_state(W_R, W_L, mesh)

	Flux = getFlux(W_L, W_R, mesh, gamma = 1.4, alpha = alpha)
		
	W = W - dt / mesh.area[...,None] * (Flux) 
	return W

def residual(W, W_old, mesh, dt):
	# 1st order
	W_L = jnp.repeat(W[...,None,:], 3, axis=-2)
	W_R = W[mesh.neighbors]

	# 2nd order - MUSCL with least-square gradient
	W_R = BC_state(W_R, W_L, mesh)
	grad = Euler_helper.getgradientLSQ(W_L, W_R, mesh)


	W_L, W_R = MUSCL(W_L, W_R, grad, mesh)
	W_R = BC_state(W_R, W_L, mesh)

	Flux = getFlux(W_L, W_R, mesh, gamma = 1.4)
		
	Res = W - W_old + dt / mesh.area[...,None] * (Flux) 
	return Res

@jax.jit(static_argnums=(1,))
def times_step_Newton(W_old, mesh, dt):
	W = W_old
	for _ in range(3):
		Fval = residual(W, W_old, mesh, dt)
		def Jv(v):
			_, jvp = jax.jvp(lambda x: residual(x, W_old, mesh, dt),
						(W,),
						(v,))
			return jvp
		delta, _ = jax.scipy.sparse.linalg.gmres(Jv, -Fval, maxiter=20)
		W = W + delta
	return W



if __name__ == "__main__":
	# little test case: Forward facing step
	mesh = Mesh()
	# mesh = Mesh_cases.Forward_Step().build(h = 2.5e-4)
	mesh = Mesh_cases.TestDipoleVortex().build(h = 5e-5, L = 1.)
	# mesh = Mesh_cases.TestMovingVortex().build(h = 2e-4, L = 1.)
	# mesh.mesh_generator(maxV=2e-4, marker_boundary=1)


	# Initial condition
	# Primitives = Test_Cases.ForwardFacingStep().build(mesh)
	# Primitives, mesh = Test_Cases.TestDipoleVortex(R = 0.05, beta = 1/5, mach = 0.01).build(mesh)
	Primitives, mesh = Test_Cases.TestDipoleVortex2(R = 0.1, omega = 300, mach = 0.01).build(mesh)
	# Primitives = Test_Cases.TestMovingVortex(R = 0.05, beta = 1/50, mach = 0.05).build(mesh)
	# Primitives = Test_Cases.TestMovingAdvection().build(mesh)
	
	W = Euler_helper.getConserved(Primitives)

	mesh.plot_mesh()

	# Time loop
	t_final = 0.15 #/ jnp.mean(Primitives[...,1]) # to get real time
	CFL = 0.05
	dx_min = jnp.min(jnp.sqrt(mesh.area))
	c = jnp.sqrt(1.4 * Primitives[...,3] / Primitives[...,0])
	dt = CFL * dx_min / jnp.max(jnp.linalg.norm(Primitives[...,1:3], axis = -1) + c)
	N_t = int(t_final / dt) + 1

	start_time = time.time()

	E = []
	Enstrophy = []
	for n in range(N_t):
		W = time_step(W, mesh, dt, alpha = 0.1)
		# W = times_step_Newton(W, mesh, dt)
		if n % 100 == 0:
			print(f'time: {n} / {N_t}')

	print(f'Simulation time: {time.time() - start_time} seconds')

	# Plot solution
	Primitives = Euler_helper.getPrimitive(W)
	mesh.plot_solution(Primitives[...,0], labels = r'$\rho$')
	mesh.plot_solution(Primitives[...,1], labels = r'$u$')
	mesh.plot_solution(Primitives[...,2], labels = r'$v$')
	mesh.plot_solution(Primitives[...,3], labels = r'$P$')

	# vorticity
	# mesh.plot_solution(vorticity, labels = r'$\omega$')

