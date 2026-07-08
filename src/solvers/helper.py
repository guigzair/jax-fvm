from webbrowser import get

import jax.numpy as jnp
import jax
import sys

def get_dt(W, mesh, CFL = 0.5):
	Primitives = getPrimitive(W)
	c = jnp.sqrt(1.4 * Primitives[...,3] / Primitives[...,0])
	lambda_max = c[...,None] + jnp.abs(jnp.sum(jnp.repeat(W[...,None,:], 3, axis=-2)[...,1:3] * mesh.normals, axis = -1))
	dt_unstr = mesh.area / jnp.sum(lambda_max * mesh.surface[mesh.face_connectivity], axis = -1)
	return jnp.min(dt_unstr) * CFL

def get_dt_viscous(mesh, CFL = 0.5, nu= 1e-5):
	dx_i = mesh.area / jnp.sum(mesh.surface[mesh.face_connectivity], axis = -1)
	dt = jnp.min(CFL * dx_i**2 / nu)
	return dt

def getConserved(Primitives, gamma = 1.4, M = 1):
	rho = Primitives[...,0]
	u = Primitives[...,1]
	v = Primitives[...,2]
	P = Primitives[...,3]
	Mass  = rho
	Mom_x = rho * u 
	Mom_y = rho * v 
	Energy = P/((gamma-1) * M**2) + 0.5*rho*(u**2 + v**2)
	W = jnp.stack([Mass, Mom_x, Mom_y, Energy], axis = -1)
	return W

def getPrimitive(W, gamma = 1.4, M = 1):
	rho = W[...,0]
	Mom_x = W[...,1]
	Mom_y = W[...,2]
	Energy = W[...,3]
	u = Mom_x / rho 
	v = Mom_y / rho 
	P = (Energy - 0.5*rho * (u**2 + v**2)) * M**2 * (gamma-1)
	Primitives = jnp.stack([rho, u, v, P], axis = -1)
	return Primitives

def get_specific_entropy(W, gamma = 1.4):
    rho = W[...,0]
    u = W[...,1] / rho
    v = W[...,2] / rho
    E = W[...,3]
    P = (E - 0.5*rho * (u**2 + v**2)) * (gamma-1)
    s = jnp.log(P / rho**gamma)
    return s

def getEntropyVariables(W, **kwargs):
	gamma = kwargs.get('gamma', 1.4)
	rho = W[...,0]
	u = W[...,1] / rho
	v = W[...,2] / rho
	E = W[...,3]
	P = (E - 0.5*rho * (u**2 + v**2)) * (gamma-1)
	s = get_specific_entropy(W, gamma = gamma)
	V1 = (gamma - s) / (gamma - 1) - 0.5 * rho * (u**2 + v**2) / P
	V2 = W[...,1] / P
	V3 = W[...,2] / P
	V4 = - rho / P
	V = jnp.stack([V1, V2, V3, V4], axis=-1)
	return V

def getConserved_from_Entropy(ETA, **kwargs):
	gamma = kwargs.get('gamma', 1.4)
	eta1 = ETA[...,0]
	eta2 = ETA[...,1]
	eta3 = ETA[...,2]
	eta4 = ETA[...,3]

	u = - eta2 / eta4
	v = - eta3 / eta4
	s = gamma - (gamma - 1) * (eta1 - 0.5 * (u**2 + v**2) * eta4)
	rho = (- eta4)**(1/(1-gamma)) * jnp.exp(s/(1-gamma))
	P = -rho / eta4
	E = P/(gamma-1) + 0.5*rho*(u**2 + v**2)
	W = jnp.stack([rho, rho * u, rho * v, E], axis=-1)
	return W

def get_IsmailRoe_variables(W, gamma = 1.4):
	rho = W[...,0]
	u = W[...,1] / rho
	v = W[...,2] / rho
	E = W[...,3]
	P = (E - 0.5*rho * (u**2 + v**2)) * (gamma-1)
	V1 = jnp.ones_like(u)
	V2 = u
	V3 = v
	V4 = P
	V = jnp.sqrt(rho/P)[...,None] * jnp.stack([V1, V2, V3, V4], axis=-1)
	return V

def get_Roe_averaged_state(W_L, W_R, gamma = 1.4):
	rho_L = W_L[...,0]
	rho_L = jnp.where(rho_L <= 0, 1e-6, rho_L)  # Avoid division by zero or negative density
	u_L = W_L[...,1] / rho_L
	v_L = W_L[...,2] / rho_L
	E_L = W_L[...,3]
	P_L = (E_L - 0.5*rho_L * (u_L**2 + v_L**2)) * (gamma-1)
	P_L = jnp.where(P_L <= 0, 1e-6, P_L)  # Avoid negative pressure

	rho_R = W_R[...,0]
	rho_R = jnp.where(rho_R <= 0, 1e-6, rho_R)  # Avoid division by zero or negative density
	u_R = W_R[...,1] / rho_R
	v_R = W_R[...,2] / rho_R
	E_R = W_R[...,3]
	P_R = (E_R - 0.5*rho_R * (u_R**2 + v_R**2)) * (gamma-1)
	P_R = jnp.where(P_R <= 0, 1e-6, P_R)  # Avoid negative pressure

	sqrt_rho_L = jnp.sqrt(rho_L)
	sqrt_rho_R = jnp.sqrt(rho_R)
	rho_avg = sqrt_rho_L * sqrt_rho_R
	u_avg = (sqrt_rho_L * u_L + sqrt_rho_R * u_R) / (sqrt_rho_L + sqrt_rho_R)
	v_avg = (sqrt_rho_L * v_L + sqrt_rho_R * v_R) / (sqrt_rho_L + sqrt_rho_R)
	H_L = (E_L + P_L) / rho_L
	H_R = (E_R + P_R) / rho_R
	H_avg = (sqrt_rho_L * H_L + sqrt_rho_R * H_R) / (sqrt_rho_L + sqrt_rho_R)
	Roe = jnp.stack([rho_avg, u_avg, v_avg, H_avg], axis=-1)
	return Roe



###########################################################################################################
##############################                Gradient                 ######################################
###########################################################################################################


def getgradientLSQ(W_L, W_R, mesh):
	# Periodic shift: for periodic faces (marker == 1) the neighbor is a real cell on the
	# opposite edge, so its barycenter must be translated by the domain period to sit next
	# to the current cell. The shift is midpoint(face) - midpoint(opposite face); it is
	# exactly zero for interior faces (face_connectivity_opposite is identity there).
	face_mid = jnp.mean(mesh.points[mesh.faces[mesh.face_connectivity]], axis = -2)  # (N_cells, 3, 2)
	face_mid_opp = jnp.mean(mesh.points[mesh.faces[mesh.face_connectivity_opposite[mesh.face_connectivity]]], axis = -2)  # (N_cells, 3, 2)
	shift = face_mid - face_mid_opp  # (N_cells, 3, 2), nonzero only on periodic faces
	# print("shift", shift.shape, shift)

	Delta_x = mesh.barycenter[mesh.neighbors] + shift - mesh.barycenter[...,None,:]  # (N_cells, 3, 2)


	# for different boundary conditions
	replace = jnp.mean(mesh.points[mesh.faces[mesh.face_connectivity]], axis = -2)
	replace = 2 * (replace - mesh.barycenter[...,None,:]) # trick in case the face is on the boundary = use face midpoint instead of neighbor cell center
	
	Delta_x = jnp.where(jnp.repeat((mesh.face_markers[mesh.face_connectivity] > 1)[...,None], 2, axis=-1), replace, Delta_x)


	Delta_w = W_R - W_L
	weights = 1 / jnp.linalg.norm(Delta_x, axis = -1)**2  # (N_cells, 3)
	# weights = jnp.ones_like(weights)  # uniform weights --- IGNORE ---

	A = jnp.einsum('ijk,ijl->ikl', weights[...,None] * Delta_x, Delta_x)  # (N_cells, 2, 2)

	b = jnp.einsum('ijk,ijl->ikl',  weights[...,None] * Delta_w, Delta_x)  # (N_cells, 2, N_vars)

	grad = jax.vmap(jax.vmap(jnp.linalg.solve))(jnp.repeat(A[:,None,...], b.shape[-2], axis=-3), b)  # (N_cells, 2, N_vars)
	return grad


def gradient_GG(W_L, W_R, mesh):
	surfaces = mesh.surface[mesh.face_connectivity]  # (N_cells, 3)
	grad = jnp.sum(0.5 * (W_R + W_L)[...,None] * mesh.normals[...,None,:] * surfaces[...,None,None], axis=-3) / mesh.area[...,None,None]  # (N_cells, N_vars, 2)
	return grad

def cell_gradient(Prim, mesh):
	# Least-square cell-centred gradient of a per-cell field, using the same
	# periodic-aware face stencil as the solver. Returns (N_cells, N_vars, 2).
	W_L = jnp.repeat(Prim[...,None,:], 3, axis=-2)  # (N_cells, 3, N_vars)
	W_R = Prim[mesh.neighbors]                      # (N_cells, 3, N_vars)
	return getgradientLSQ(W_L, W_R, mesh)

###########################################################################################################
##########################        Equilibrium (balanced) IC             ###################################
###########################################################################################################

def equilibrium_pressure_poisson(Primitives, mesh, gamma = 1.4, rho_inf = 1.0,
								  n_iter = 3, cg_tol = 1e-10, cg_maxiter = 2000):
	"""
	Given a (fixed) velocity field, rebuild rho and p so that the initial state is
	in mechanical equilibrium, grad(p) = -rho (u.grad)u, killing the spurious
	acoustic transient that a mismatched IC would otherwise radiate.

	Takes the divergence of the momentum balance and solves the finite-volume
	Poisson problem   div(grad p) = div( -rho (u.grad)u )   on the mesh, with the
	natural Neumann flux grad(p).n = f.n on physical (marker>1) boundaries and full
	periodic coupling on marker==1 faces. Density is then recovered from the
	isentropic law p = K rho^gamma and the whole thing is Picard-iterated a few
	times because f depends on rho (the velocity field itself is held fixed).
	"""
	Prim = Primitives
	u = Prim[...,1]
	v = Prim[...,2]
	p_inf = jnp.mean(Prim[...,3])            # uniform far-field pressure
	K = p_inf / rho_inf**gamma               # isentropic constant p = K rho^gamma

	# --- convective acceleration a = (u.grad)u  (velocity is fixed -> compute once)
	grad = cell_gradient(Prim, mesh)         # (N_cells, N_vars, 2)
	gu = grad[:, 1, :]                        # grad u
	gv = grad[:, 2, :]                        # grad v
	a = jnp.stack([u * gu[:, 0] + v * gu[:, 1],
				   u * gv[:, 0] + v * gv[:, 1]], axis=-1)  # (N_cells, 2)

	# --- face geometry / transmissibilities (periodic shift as in getgradientLSQ)
	nb = mesh.neighbors                                              # (N_cells, 3)
	n_f = mesh.normals                                              # (N_cells, 3, 2) unit outward
	S_f = mesh.surface[mesh.face_connectivity]                      # (N_cells, 3)
	face_mid = jnp.mean(mesh.points[mesh.faces[mesh.face_connectivity]], axis=-2)
	face_mid_opp = jnp.mean(mesh.points[mesh.faces[mesh.face_connectivity_opposite[mesh.face_connectivity]]], axis=-2)
	shift = face_mid - face_mid_opp
	Delta_x = mesh.barycenter[nb] + shift - mesh.barycenter[...,None,:]  # (N_cells, 3, 2)
	d2 = jnp.sum(Delta_x**2, axis=-1)                                    # (N_cells, 3)
	w_f = S_f * jnp.sum(Delta_x * n_f, axis=-1) / (d2 + 1e-30)           # two-point transmissibility
	interior = mesh.face_markers[mesh.face_connectivity] <= 1            # periodic(1) & interior(0) coupled; wall(>1) = Neumann
	w_f = jnp.where(interior, w_f, 0.0)

	def matvec(p):
		# SPD operator: -(integral of Laplacian) + pin of the constant nullspace.
		lap = jnp.sum(w_f * (p[nb] - p[...,None]), axis=-1)  # sum_f w_f (p_j - p_i) ~ int Lap dV
		return -lap + jnp.mean(p)

	def build_rhs(rho):
		f = -rho[...,None] * a                                          # forcing -rho (u.grad)u
		f_face = jnp.where(interior[...,None], 0.5 * (f[...,None,:] + f[nb]), f[...,None,:])
		div = jnp.sum(jnp.sum(f_face * n_f, axis=-1) * S_f, axis=-1)     # int div(f) dV per cell
		div = div - jnp.mean(div)                                       # compatibility (zero mean)
		return -div                                                    # matches -lap sign in matvec

	rho = rho_inf * jnp.ones_like(u)
	p = Prim[...,3]
	for _ in range(n_iter):
		rhs = build_rhs(rho)
		p, _ = jax.scipy.sparse.linalg.cg(matvec, rhs, x0=p - jnp.mean(p),
										  tol=cg_tol, maxiter=cg_maxiter)
		p = p - jnp.mean(p) + p_inf
		rho = rho_inf * (p / p_inf)**(1.0 / gamma)

	Prim = Prim.at[...,0].set(rho)
	Prim = Prim.at[...,3].set(p)
	return Prim

###########################################################################################################
##############################                  BC                   ######################################
###########################################################################################################

def BC_outflow(W_R, W_L, mesh, bc_type = 4):
	W_R = jnp.where(jnp.repeat((mesh.face_markers[mesh.face_connectivity] == bc_type)[...,None], 4, axis=-1), W_L, W_R)
	return W_R	

def BC_inflow(W, mesh, bc_type = 3, value = jnp.array([1.0, 1.0, 1.0, 1.0])):
	W = jnp.where(jnp.repeat((mesh.face_markers[mesh.face_connectivity] == bc_type)[...,None], 4, axis=-1), value, W)
	return W	

def BC_subsonic_inlet(W_R, W_L, mesh, bc_type = 5):
	Prim_L = getPrimitive(W_L)
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
	Prim_L = getPrimitive(W_L)
	vn = (Prim_L[...,1] - value[1]) * mesh.normals[...,0] + (Prim_L[...,2] - value[2]) * mesh.normals[...,1]
	vb = (Prim_L[...,1:3] - value[1:3]) - 2 * vn[...,None] * mesh.normals
	Prim_b = Prim_L.at[...,1:3].set(vb + value[1:3])
	W_b = getConserved(Prim_b)
	W_R = jnp.where(jnp.repeat((mesh.face_markers[mesh.face_connectivity] == bc_type)[...,None], 4, axis=-1), W_b, W_R)
	return W_R	

def BC_noslip_wall(W_R, W_L, mesh, bc_type = 2):
	Prim_L = getPrimitive(W_L)
	vn = (Prim_L[...,1] * mesh.normals[...,0] + Prim_L[...,2] * mesh.normals[...,1])
	vt = (- Prim_L[...,1] * mesh.normals[...,1] + Prim_L[...,2] * mesh.normals[...,0])

	vb = Prim_L[...,1:3] - 2 * vn[...,None] * mesh.normals - 2 * vt[...,None] * jnp.stack([-mesh.normals[...,1], mesh.normals[...,0]], axis=-1)

	Prim_b = Prim_L.at[...,1:3].set(vb)
	W_b = getConserved(Prim_b)
	W_R = jnp.where(jnp.repeat((mesh.face_markers[mesh.face_connectivity] == bc_type)[...,None], 4, axis=-1), W_b, W_R)
	return W_R	

def BC_state(W_R, W_L, mesh, **kwargs):
	W_R = jax.lax.cond(kwargs.get('flag_NS', False),
						lambda x, y: BC_noslip_wall(x, y, mesh, bc_type=2),
						lambda x, y: BC_slipwall(x, y, mesh, bc_type=2),
						W_R, W_L)
	W_R = BC_inflow(W_R, mesh, bc_type=3, value = kwargs.get('value', jnp.array([1.0, 1.0, 1.0, 1.0])))  # (supersonic inlet)
	W_R = BC_outflow(W_R, W_L, mesh, bc_type=4)  # (free outflow)
	W_R = BC_subsonic_inlet(W_R, W_L, mesh, bc_type=5)  # (subsonic inlet)
	return W_R

###########################################################################################################
##########################               other functions                   ################################
###########################################################################################################

def get_temperature(Primitives, R = 287):
	rho = Primitives[...,0]
	P = Primitives[...,3]
	T = P / (rho * R)
	return T

def get_mach_number(Primitives, gamma = 1.4):
	u = Primitives[...,1]
	v = Primitives[...,2]
	P = Primitives[...,3]
	rho = Primitives[...,0]
	c = jnp.sqrt(gamma * P / rho)
	M = jnp.sqrt(u**2 + v**2) / c
	return M

def get_total_entropy(W, mesh, gamma = 1.4):
	eta = get_specific_entropy(W, gamma = gamma)
	total_entropy = - jnp.sum(W[...,0] * eta * mesh.area / (gamma - 1), axis = -1)
	return total_entropy

def get_kinetic_energy(Primitives):
    u = Primitives[...,1]
    v = Primitives[...,2]
    return  0.5 * (u**2 + v**2)

def get_total_kinetic_energy(W, mesh):
	Primitives = getPrimitive(W)
	kinetic_energy = get_kinetic_energy(Primitives)
	total_kinetic_energy = jnp.sum(kinetic_energy * mesh.area, axis = -1)
	return total_kinetic_energy

def get_vorticity(grad):
	# take as input the gradient of primitives field
    du_dy = grad[:,1,1]
    dv_dx = grad[:,2,0]
    omega = dv_dx - du_dy
    return omega

def get_vorticity_from_field(W, mesh, **kwargs):
	W_L = jnp.repeat(W[...,None,:], 3, axis=-2)
	W_R = W[mesh.neighbors]
	W_R = BC_state(W_R, W_L, mesh, flag_NS=kwargs.get('flag_NS', False))
	grad = getgradientLSQ(getPrimitive(W_L), getPrimitive(W_R), mesh)

	vort = get_vorticity(grad)
	return vort

def get_total_enstrophy(W, mesh, **kwargs):
	vorticity = get_vorticity_from_field(W, mesh, **kwargs)
	total_enstrophy = 0.5 * jnp.sum(vorticity**2 * mesh.area, axis = -1)
	return total_enstrophy

def get_palinstrophy(grad, mesh):
    # take as input the gradient of primitives field
    du_dy = grad[:,1,1]
    dv_dx = grad[:,2,0]
    omega = dv_dx - du_dy
    omega_L = jnp.repeat(omega[...,None,:], 3, axis=-2)
    omega_R = omega[mesh.neighbors]
    omega_R = jnp.where(jnp.repeat((mesh.face_markers[mesh.face_connectivity] > 0)[...,None], 1, axis=-1), 0., omega_R) # Boundary faces: reverse the direction
    grad_omega = getgradientLSQ(omega_L, omega_R, mesh)  # (N_cells, 2, 1)
    palin = jnp.linalg.norm(grad_omega, axis = -1)**2  # (N_cells, 1)
    return palin


if __name__ == "__main__":
	sys.path.append('../../..')  
	from jax_fvm.src.config import Case
	from jax_fvm.src.mesh.mesh import Mesh 
	mesh = Mesh()
	# mesh.mesh_generator(maxV = 5.0e-3, marker_boundary = 1, bounds = [-2., 2., -2., 2.])

	case = Case.build("../Cases/config/Euler/corotating_merge.yaml", MeshClass=Mesh)
	W_L = jnp.repeat(case.primitives[...,None,:], 3, axis=-2)
	W_R = case.primitives[case.mesh.neighbors]
	grad = getgradientLSQ(W_L, W_R, case.mesh)

	case.mesh.plot_solution(grad[...,0,0], labels = "grad_rho")