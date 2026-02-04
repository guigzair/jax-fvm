import jax.numpy as jnp
import jax
import sys

sys.path.append('../../../..')  
from FVM.src.mesh.mesh import Mesh
import FVM.src.Cases.Test_Cases as Test_Cases
import FVM.src.mesh.Mesh_cases as Mesh_cases
import time
import helper as NS_helper

"""
Finite Volume Method for 2D incompressible NS equations
With jax.jit compilation = ~ 100x faster for large data
"""


def getFlux_convective(W_L, W_R, mesh, gamma = 1.4):
	
	return None

def getFlux_diffusive(U_L, U_R):
    grad_U = NS_helper.getgradientLSQ(U_L, U_R, mesh)

    grad_U_L = jnp.repeat(grad_U[...,None,:,:], 3, axis=-3)
    grad_U_R = grad_U[mesh.neighbors]

    grad_ij = 0.5 * (grad_U_L + grad_U_R)

    Delta_x = mesh.barycenter[mesh.neighbors] - mesh.barycenter[...,None,:]  # (N_cells, 3, 2)
    replace = jnp.mean(mesh.points[mesh.faces[mesh.face_connectivity]], axis = -2)
    replace = 2 * (replace - mesh.barycenter[...,None,:]) # trick in case the face is on the boundary = use face midpoint instead of neighbor cell center
    Delta_x = jnp.where(jnp.repeat((mesh.face_markers[mesh.face_connectivity] > 0)[...,None], 2, axis=-1), replace, Delta_x)
    Delta = jnp.linalg.norm(Delta_x, axis = -1)
    normed_Delta_x = Delta_x / Delta[...,None]
    grad_ij += ((U_L - U_R)/Delta[...,None] - jnp.einsum('ijkl,ijl->ijk',grad_ij , normed_Delta_x))[...,None] * normed_Delta_x[...,None,:]

    # Get corresponding normals
    nx = mesh.normals[...,0]
    ny = mesh.normals[...,1]
	
    flux_diffusive = grad_ij[...,0] * nx[...,None] + grad_ij[...,1] * ny[...,None]
	
    return jnp.sum(flux_diffusive, axis = -2)  # (N_cells, N_var)



@jax.jit(static_argnums=(1,))
def time_step(W, mesh, dt, **kwargs):

	W = W - dt / mesh.area[...,None] 
	return W


if __name__ == "__main__":
	# little test case: Forward facing step
	# mesh = Mesh_cases.Forward_Step().build(h = 2.5e-4)
	mesh = Mesh()
	mesh.mesh_generator(maxV=2e-4, marker_boundary=1)


	# Initial condition
	U = Test_Cases.TaylorGreenVortex().build(mesh)
	

	mesh.plot_mesh()

	# Time loop
	t_final = 4.
	CFL = 0.1
	dx_min = jnp.min(jnp.sqrt(mesh.area))
	dt = CFL * dx_min 
	N_t = int(t_final / dt) + 1

	start_time = time.time()
	for n in range(0):
		U = time_step(U, mesh, dt)
		if n % 100 == 0:
			print(f'time: {n} / {N_t}')
	print(f'Simulation time: {time.time() - start_time} seconds')

	# Plot solution
	mesh.plot_solution(U[...,0], labels = r'$u$')
	mesh.plot_solution(U[...,1], labels = r'$v$')