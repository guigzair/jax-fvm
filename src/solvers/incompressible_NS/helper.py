import jax.numpy as jnp
import jax

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


