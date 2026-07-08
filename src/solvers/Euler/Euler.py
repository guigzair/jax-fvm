import jax.numpy as jnp
import jax
import sys

sys.path.append('../../../..')  
from jax_fvm.src.config import Case
from jax_fvm.src.mesh.mesh import Mesh 
import jax_fvm.src.solvers.Time_Integration as TimeIntegration
import time
import jax_fvm.src.solvers.helper as helper 
import matplotlib.pyplot as plt
# jax.config.update('jax_enable_x64', True)
jax.config.update("jax_debug_nans", True)
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


###########################################################################################################
##############################                Solver                 ######################################
###########################################################################################################

def venkatakrishnan(a, b, h = 0, K = 0):
	omega = (K * h)**3 
	L = (a**2 + 2 * a*b + omega[...,None,None]) / (a**2 + 2*b**2 + a*b + omega[...,None,None] + 1e-7)
	return L

def minmod(a, b, **kwargs):
	c = a / b
	return jnp.minimum(1., c)

def getlimiting(W_L, W_R, grad, mesh, limiting_fct = venkatakrishnan):
	W_m  = jnp.min(jnp.concatenate([W_L, W_R], axis = -2), axis = -2) # (N_cells, N_vars)
	W_M  = jnp.max(jnp.concatenate([W_L, W_R], axis = -2), axis = -2) # (N_cells, N_vars)

	mid_point_faces = jnp.mean(mesh.points[mesh.faces[mesh.face_connectivity]], axis = -2) # (N_cells,3,2)
	delta_x = mid_point_faces - mesh.barycenter[...,None,:]  # (N_cells, 3, 2)
	Delta = jnp.einsum('ijl,ikl->ijk', delta_x, grad)  # (N_cells, 3, N_vars)

	phi = jnp.ones_like(Delta)

	eps = jnp.asarray(1e-12 if W_L.dtype == jnp.float64 else 1e-7, dtype=W_L.dtype)
	phi = jnp.where(Delta > eps,
					limiting_fct(W_M[...,None,:] - W_L, Delta, h = jnp.sqrt(mesh.area), K =0.),
					phi)
	phi = jnp.where(Delta < -eps,
					limiting_fct(W_m[...,None,:] - W_L, Delta, h = jnp.sqrt(mesh.area), K = 0.),
					phi)
	
	# phi = jnp.where(mesh.face_markers[mesh.face_connectivity][...,None] > 1,
	# 					0.,
	# 					phi)  # set phi to 0 if tri is on the boundary (no limiter for boundary faces)
	phi = jnp.min(phi, axis = -2)  # (N_cells, N_vars)
	return phi

def MUSCL(W_L, W_R, grad, mesh):

	phi = getlimiting(W_L, W_R, grad, mesh)  # (N_cells, N_vars)
	# phi = jnp.ones_like(phi) # --- IGNORE --- for 2nd order without limiter

	mid_point_faces = jnp.mean(mesh.points[mesh.faces[mesh.face_connectivity]], axis = -2) # (N_cells,3,2)
	
	delta_x = mid_point_faces - mesh.barycenter[...,None,:]  # (N_cells, 3, 2)
	Delta = jnp.einsum('ijl,ikl->ijk', delta_x, grad)  # (N_cells, 3, N_vars)

	mid_point_faces_neigh = jnp.mean(mesh.points[mesh.faces[mesh.face_connectivity_opposite[mesh.face_connectivity]]], axis = -2)
	delta_x_neigh = mid_point_faces_neigh - mesh.barycenter[mesh.neighbors] 
	
	delta_x_neigh = jnp.where(jnp.repeat((mesh.face_markers[mesh.face_connectivity] > 1)[...,None], 2, axis=-1), 
						   		-delta_x, delta_x_neigh) 
	
	Delta_neigh = jnp.einsum('ijl,ijkl->ijk', delta_x_neigh, grad[mesh.neighbors])

	phi_L = jnp.where(mesh.face_markers[mesh.face_connectivity][...,None] > 1,
						0.,
						jnp.repeat(phi[...,None,:], 3, axis=-2))  # (N_cells, 3, N_vars) 
	# phi_L = phi[...,None,:]
	W_L_MUSCL = W_L + phi_L * Delta  # (N_cells, 3, N_vars)

	phi_R = jnp.where(mesh.face_markers[mesh.face_connectivity][...,None] > 1,
							0.,
							phi[mesh.neighbors])  # (N_cells, 3, N_vars) 
	W_R_MUSCL = W_R + phi_R * Delta_neigh  # (N_cells, 3, N_vars)

	return W_L_MUSCL, W_R_MUSCL

def Roe(W_L, W_R, normals, **kwargs):
	gamma = kwargs.get('gamma', 1.4)
	Roe_avg = helper.get_Roe_averaged_state(W_L, W_R, gamma = gamma)
	rho_roe = Roe_avg[...,0]
	u_roe = Roe_avg[...,1] 
	v_roe = Roe_avg[...,2]
	H_roe = Roe_avg[...,3]
	a_roe = jnp.sqrt(jnp.abs((gamma-1) * (H_roe - 0.5 * (u_roe**2 + v_roe**2))))

	R = jnp.zeros(W_L.shape + (4,))  # (N_cells, 3, N_vars, N_vars)
	R = R.at[...,0,0].set(1.)
	R = R.at[...,1,0].set(u_roe - a_roe * normals[...,0])
	R = R.at[...,2,0].set(v_roe - a_roe * normals[...,1])
	R = R.at[...,3,0].set(H_roe - a_roe * (u_roe * normals[...,0] + v_roe * normals[...,1]))

	R = R.at[...,0,1].set(1.)
	R = R.at[...,1,1].set(u_roe)
	R = R.at[...,2,1].set(v_roe)
	R = R.at[...,3,1].set(0.5 * (u_roe**2 + v_roe**2))	

	R = R.at[...,0,2].set(0.)
	R = R.at[...,1,2].set(-normals[...,1])
	R = R.at[...,2,2].set(normals[...,0])
	R = R.at[...,3,2].set( - u_roe * normals[...,1] + v_roe * normals[...,0])	

	R = R.at[...,0,3].set(1.)
	R = R.at[...,1,3].set(u_roe + a_roe * normals[...,0])
	R = R.at[...,2,3].set(v_roe + a_roe * normals[...,1])
	R = R.at[...,3,3].set(H_roe + a_roe * (u_roe * normals[...,0] + v_roe * normals[...,1]))	

	Lambda = jnp.zeros(W_L.shape[:-1] + (4,))  # (N_cells, 3, N_vars)
	Lambda = Lambda.at[...,0].set(u_roe * normals[...,0] + v_roe * normals[...,1] - a_roe)
	Lambda = Lambda.at[...,1].set(u_roe * normals[...,0] + v_roe * normals[...,1])
	Lambda = Lambda.at[...,2].set(u_roe * normals[...,0] + v_roe * normals[...,1])
	Lambda = Lambda.at[...,3].set(u_roe * normals[...,0] + v_roe * normals[...,1] + a_roe)

	Eta_R = helper.getEntropyVariables(W_R, gamma = gamma)
	Eta_L = helper.getEntropyVariables(W_L, gamma = gamma)
	dv = Eta_R - Eta_L

	scaling = jnp.zeros_like(Lambda)
	scaling = scaling.at[...,0].set(rho_roe / (2 * gamma))   # acoustic L
	scaling = scaling.at[...,1].set((gamma-1) * rho_roe / gamma)  # entropy
	scaling = scaling.at[...,2].set( rho_roe / gamma)                      # shear
	scaling = scaling.at[...,3].set(rho_roe / (2 * gamma))    # acoustic R
	
	R_inv = jnp.transpose(R, axes = (0,1,3,2))  
	# R_inv = jnp.linalg.inv(R)
	RH = jnp.einsum('...ij,...j->...i', R_inv, dv)  
	DRH = jnp.einsum('...i,...i->...i',jnp.abs(Lambda  * scaling), RH)  
	RDRH = jnp.einsum('...ij,...j->...i', R, DRH)

	return 0.5 * RDRH

def Rusanov(W_L, W_R, normals, **kwargs):
	gamma = kwargs.get('gamma', 1.4)
	M = kwargs.get('M', 1.)

	# Get the cell state for each edge
	Prim_L = helper.getPrimitive(W_L, gamma = gamma, M = M)
	rho_L = Prim_L[...,0]
	u_L = Prim_L[...,1]
	v_L = Prim_L[...,2]
	P_L = Prim_L[...,3] 

	# Get the corresponding neighbors state
	Prim_R = helper.getPrimitive(W_R, gamma = gamma, M = M)
	rho_R = Prim_R[...,0]
	u_R = Prim_R[...,1]
	v_R = Prim_R[...,2]
	P_R = Prim_R[...,3] 

	# Get corresponding normals
	nx = normals[...,0]
	ny = normals[...,1]

	# Maximum wavelenghts
	C_L = jnp.sqrt(jnp.abs(gamma*P_L/rho_L)) / M + jnp.abs(u_L * nx + v_L * ny)
	C_R = jnp.sqrt(jnp.abs(gamma*P_R/rho_R)) / M + jnp.abs(u_R * nx + v_R * ny)
	C_max = jnp.maximum(C_R, C_L)

	Eta_L = helper.getEntropyVariables(W_L, gamma = gamma)
	Eta_R = helper.getEntropyVariables(W_R, gamma = gamma)
	Eta_bar = 0.5 * (Eta_L + Eta_R)
	dv = jax.jvp(lambda x: helper.getConserved_from_Entropy(x, gamma = gamma), (Eta_bar,), (Eta_R - Eta_L,))[1]

	return C_max[...,None] * 0.5 * dv # (N_cells, 3, N_vars)

def getFlux(W_L, W_R, normals, surfaces, **kwargs):
	gamma = kwargs.get('gamma', 1.4)
	M = kwargs.get('M', 1.)
	# i did not put mesh as input in order to vmap this function
	# Get the cell state for each edge
	Prim_L = helper.getPrimitive(W_L, gamma = gamma, M = M)
	rho_L = Prim_L[...,0]
	u_L = Prim_L[...,1]
	v_L = Prim_L[...,2]
	P_L = Prim_L[...,3]

	# Get the corresponding neighbors state
	Prim_R = helper.getPrimitive(W_R, gamma = gamma, M = M)
	rho_R = Prim_R[...,0]
	u_R = Prim_R[...,1]
	v_R = Prim_R[...,2]
	P_R = Prim_R[...,3]

	# Get corresponding normals
	nx = normals[...,0]
	ny = normals[...,1]

	# Energy
	en_L = W_L[...,3] 
	en_R = W_R[...,3]

	# Flux
	flux_rho_L = rho_L * (u_L * nx + v_L * ny)
	flux_ru_L = rho_L * u_L* ( u_L * nx + v_L * ny) + 1/M**2 * P_L * nx
	flux_rv_L = rho_L * v_L * (u_L * nx + v_L * ny) + 1/M**2 * P_L * ny
	flux_E_L = (en_L + 1/M**2 * P_L) * (u_L * nx + v_L * ny)

	flux_rho_R = rho_R * (u_R * nx + v_R * ny)
	flux_ru_R = rho_R * u_R * ( u_R * nx + v_R * ny) + 1/M**2 * P_R * nx
	flux_rv_R = rho_R * v_R * (u_R * nx + v_R * ny) + 1/M**2 * P_R * ny
	flux_E_R = (en_R + 1/M**2 * P_R) * (u_R * nx + v_R * ny)

	Flux = jnp.stack([0.5 * (flux_rho_L + flux_rho_R), 
						0.5 * (flux_ru_L + flux_ru_R), 
						0.5 * (flux_rv_L + flux_rv_R), 
						0.5 * (flux_E_L + flux_E_R)], axis = -1) 
	# Total flux
	alpha = kwargs.get('alpha', 1.)
	scaling = jnp.array([alpha, alpha, alpha, alpha])

	Flux = Flux - scaling * Rusanov(W_L, W_R, normals, **kwargs)
	# Flux = Flux - scaling * Roe(W_L, W_R, normals, **kwargs)

	Flux = surfaces[...,None] * Flux

	Flux = jnp.sum(Flux, axis = -2)
	return Flux

def log_mean(aL, aR):
    xi = aL / aR
    f  = (xi - 1.0) / (xi + 1.0)
    u  = f * f
    F = jnp.where(u < 1e-3, 
                    1.0 + u/3.0 + u**2/5.0 + u**3/7.0,  # Taylor series, avoids log cancellation
                    jnp.log(xi) / (2.0 * f)
				)
    return (aL + aR) / (2.0 * F)

def getFlux_Tadmor(W_L, W_R, normals, surfaces, **kwargs):
	gamma = kwargs.get('gamma', 1.4)
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

	Z_L = helper.get_IsmailRoe_variables(W_L, gamma = gamma)
	Z_R = helper.get_IsmailRoe_variables(W_R, gamma = gamma)

	Eta_L = helper.getEntropyVariables(W_L, gamma = gamma)
	Eta_R = helper.getEntropyVariables(W_R, gamma = gamma)

	Z_bar = 0.5 * (Z_L + Z_R)

	# Get corresponding normals
	nx = normals[...,0]
	ny = normals[...,1]

	# Maximum wavelenghts
	C_L = jnp.sqrt(jnp.abs(gamma*P_L/rho_L)) + jnp.abs(u_L * nx + v_L * ny)
	C_R = jnp.sqrt(jnp.abs(gamma*P_R/rho_R)) + jnp.abs(u_R * nx + v_R * ny)
	C_max = jnp.maximum(C_R, C_L)

	Z1_ln = log_mean(Z_L[...,0], Z_R[...,0])
	Z4_ln = log_mean(Z_L[...,3], Z_R[...,3])

	rho_hat = Z_bar[...,0] * Z4_ln
	u_hat = Z_bar[...,1] / Z_bar[...,0]
	v_hat = Z_bar[...,2] / Z_bar[...,0]
	P1_hat = Z_bar[...,3] / Z_bar[...,0]
	P2_hat = (gamma + 1) / (2*gamma) * Z4_ln / Z1_ln +  (gamma - 1) / (2*gamma) * Z_bar[...,3] / Z_bar[...,0]
	a_hat = jnp.sqrt(gamma * P2_hat / rho_hat)
	H_hat = a_hat**2 / (gamma-1) + (u_hat**2 + v_hat**2) / 2


	F1 = rho_hat * (u_hat * nx + v_hat * ny)
	F2 = rho_hat * u_hat * (u_hat * nx + v_hat * ny) + P1_hat * nx
	F3 = rho_hat * v_hat * (u_hat * nx + v_hat * ny) + P1_hat * ny
	F4 = rho_hat * (u_hat * nx + v_hat * ny) * H_hat


	Flux = jnp.stack([F1, F2, F3, F4], axis = -1)

	# Total flux
	alpha = kwargs.get('alpha', 1.)
	scaling = jnp.array([alpha, alpha, alpha, alpha])
	Flux = Flux - scaling * Rusanov(W_L, W_R, normals, **kwargs)
	# Flux = Flux - alpha * Roe(W_L, W_R, normals, **kwargs) 

	# Flux = surfaces[...,None] * Flux

	# Flux = jnp.sum(Flux, axis = -2)
	return Flux


###########################################################################################################
############################           Time integration                 ###################################
###########################################################################################################

@jax.jit(static_argnums=(1,))
def residual(W, mesh, **kwargs):
	# 1st order
	W_L = jnp.repeat(W[...,None,:], 3, axis=-2)
	W_R = W[mesh.neighbors]

	# 2nd order - MUSCL with least-square gradient
	W_R = helper.BC_state(W_R, W_L, mesh, **kwargs)

	W_L = helper.getPrimitive(W_L, gamma = kwargs.get('gamma', 1.4), M = kwargs.get('M', 1.))
	W_R = helper.getPrimitive(W_R, gamma = kwargs.get('gamma', 1.4), M = kwargs.get('M', 1.))
	grad = helper.getgradientLSQ(W_L, W_R, mesh)
	W_L, W_R = MUSCL(W_L, W_R, grad, mesh)
	W_L = helper.getConserved(W_L, gamma = kwargs.get('gamma', 1.4), M = kwargs.get('M', 1.))
	W_R = helper.getConserved(W_R, gamma = kwargs.get('gamma', 1.4), M = kwargs.get('M', 1.))

	Flux = getFlux_Tadmor(W_L, W_R, mesh.normals, mesh.surface[mesh.face_connectivity], **kwargs) 
	
	Flux = mesh.surface[mesh.face_connectivity][...,None] * Flux
	Flux = jnp.sum(Flux, axis = -2)
	return Flux / mesh.area[...,None] 






if __name__ == "__main__":
	cfg_path = "../../Cases/config/Euler/corotating_merge.yaml"
	# cfg_path = "../../Cases/config/KelvinHelmholtz.yaml"
	# cfg_path = "../../Cases/config/DipoleVortex.yaml"
	case = Case.build(cfg_path, MeshClass=Mesh)

	# mesh = Mesh()
	# mesh.mesh_generator(maxV = 1e-3, marker_boundary = 1, bounds = [-1., 1., -1., 1.])
	# case.mesh = mesh
	# import jax_fvm.src.Cases.Test_Cases as Test_Cases
	# case.primitives = Test_Cases.advected_sinus().build(case.mesh)

	kwargs = case.config.kwargs

	W = helper.getConserved(case.primitives, gamma = kwargs['gamma'], M = kwargs.get('M', 1.))
	Primitives = helper.getPrimitive(W, gamma = kwargs['gamma'], M = kwargs.get('M', 1.))
	case.mesh.plot_solution(Primitives[...,1], labels = r'Initial condition $u$')

	case.mesh.plot_slice(Primitives[...,1], labels = r'$\rho$')

	t_final = case.config.time.t_end
	dt = helper.get_dt(W, case.mesh, CFL = case.config.time.cfl)
	N_t = int(t_final / dt) + 1

	start_time = time.time()

	T_interval_snapshots = 1000
	Snapshots = jnp.zeros((int(N_t/T_interval_snapshots), *W.shape))
	for n in range(N_t):
		W = TimeIntegration.time_step_RK2(W, dt, case.mesh, residual, **kwargs)
		if n % 1000 == 0:
			print(f'It : {n} / {N_t}')
			if jnp.isnan(W).any():
				print(f'NaN detected at iteration {n}')
				break
		if n % T_interval_snapshots == 0:
			Snapshots = Snapshots.at[int(n/T_interval_snapshots)].set(W)

	print(f'Simulation time: {time.time() - start_time} seconds')
	# Plot solution
	Primitives = helper.getPrimitive(W, gamma = kwargs['gamma'], M = kwargs.get('M', 1.))
	case.mesh.plot_solution(Primitives[...,0], labels = r'$\rho$')
	case.mesh.plot_solution(Primitives[...,1], labels = r'$u$')

	# vorticity
	vorticity = helper.get_vorticity_from_field(W, case.mesh)
	case.mesh.plot_solution(vorticity, labels = r'$\omega$')

	# entropy plot
	Entropy_rec = jnp.zeros(Snapshots.shape[0])
	for i in range(Snapshots.shape[0]):
		Entropy_rec = Entropy_rec.at[i].set(helper.get_total_entropy(Snapshots[i], case.mesh))
	plt.plot(Entropy_rec)

	Total_entropy = jax.lax.map(lambda x: helper.get_total_entropy(x, case.mesh), Snapshots)
	fig, ax = plt.subplots()
	ax.plot(jnp.arange(Snapshots.shape[0]) * T_interval_snapshots * dt, Total_entropy)
	ax.set_xlabel(r't')
	ax.set_ylabel(r'$\mathcal{S}$')
	ax.grid()

	# enstrophy plot
	Total_enstrophy = jax.lax.map(lambda x: helper.get_total_enstrophy(x, case.mesh), Snapshots)
	fig, ax = plt.subplots()
	ax.plot(jnp.arange(Snapshots.shape[0]) * T_interval_snapshots * dt, Total_enstrophy)
	ax.set_xlabel(r't')
	ax.set_ylabel(r'$\xi$')
	ax.grid()

	# energy plot
	Total_energy = jax.lax.map(lambda x: helper.get_total_kinetic_energy(x, case.mesh), Snapshots)
	fig, ax = plt.subplots()
	ax.plot(jnp.arange(Snapshots.shape[0]) * T_interval_snapshots * dt, Total_energy)
	ax.set_xlabel(r't')
	ax.set_ylabel(r'$E$')
	ax.grid()

	# vorticity_snap = jax.lax.map(lambda x: helper.get_vorticity_from_field(x, case.mesh), Snapshots)
	# case.mesh.animate_field(vorticity_snap[::2], path = "vorticity.gif", interval = 200)
