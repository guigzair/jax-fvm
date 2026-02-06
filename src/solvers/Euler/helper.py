import jax.numpy as jnp
import jax
import sys


def getConserved(Primitives, gamma = 1.4):
	rho = Primitives[...,0]
	u = Primitives[...,1]
	v = Primitives[...,2]
	P = Primitives[...,3]
	Mass  = rho
	Mom_x = rho * u 
	Mom_y = rho * v 
	Energy = P/(gamma-1) + 0.5*rho*(u**2 + v**2)
	W = jnp.stack([Mass, Mom_x, Mom_y, Energy], axis = -1)
	return W

def getPrimitive(W, gamma = 1.4):
	rho = jnp.clip(W[...,0], a_min = 1e-5)
	Mom_x = W[...,1]
	Mom_y = W[...,2]
	Energy = W[...,3]
	u = Mom_x / rho 
	v = Mom_y / rho 
	P = (Energy - 0.5*rho * (u**2 + v**2)) * (gamma-1)
	P = jnp.clip(P, a_min = 1e-4)
	Primitives = jnp.stack([rho, u, v, P], axis = -1)
	return Primitives

def getgradientLSQ(W_L, W_R, mesh):
	Delta_x = mesh.barycenter[mesh.neighbors] - mesh.barycenter[...,None,:]  # (N_cells, 3, 2)
	replace = jnp.mean(mesh.points[mesh.faces[mesh.face_connectivity]], axis = -2)
	replace = 2 * (replace - mesh.barycenter[...,None,:]) # trick in case the face is on the boundary = use face midpoint instead of neighbor cell center
	Delta_x = jnp.where(jnp.repeat((mesh.face_markers[mesh.face_connectivity] > 0)[...,None], 2, axis=-1), replace, Delta_x)

	Delta_w = W_R - W_L

	A = jnp.einsum('ijk,ijl->ikl', Delta_x, Delta_x)  # (N_cells, 2, 2)
	b = jnp.einsum('ijk,ijl->ikl', Delta_w, Delta_x)  # (N_cells, 2, N_vars)

	grad = jax.vmap(jax.vmap(jnp.linalg.solve))(jnp.repeat(A[:,None,...], b.shape[-2], axis=-3), b)  # (N_cells, 2, N_vars)
	return grad

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

def BC_state(W_R, W_L, mesh, **kwargs):
	# value = kwargs.get('value', jnp.array([1.0, 1.0, 1.0, 1.0]))
	W_R = BC_slipwall(W_R, W_L, mesh, bc_type=2)  # (slip-wall)
	W_R = BC_inflow(W_R, mesh, bc_type=3, value = getConserved(mesh.inlet_subsonic))  # (supersonic inlet)
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


def get_kinetic_energy(Primitives):
    u = Primitives[...,1]
    v = Primitives[...,2]
    return  0.5 * (u**2 + v**2)

def get_vorticity(grad):
	# take as input the gradient of primitives field
    du_dy = grad[:,1,1]
    dv_dx = grad[:,2,0]
    omega = dv_dx - du_dy
    return omega

def get_vorticity_from_field(W, mesh):
	W_L = jnp.repeat(W[...,None,:], 3, axis=-2)
	W_R = W[mesh.neighbors]
	W_R = BC_state(W_R, W_L, mesh)
	grad = getgradientLSQ(getPrimitive(W_L), getPrimitive(W_R), mesh)

	vort = get_vorticity(grad)
	return vort



def get_enstrophy(grad):
	# take as input the gradient of primitives field
    omega = get_vorticity(grad)
    return 0.5 * omega**2

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


