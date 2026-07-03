from dataclasses import dataclass

import jax
import jax.numpy as jnp

@jax.jit(static_argnums=(2,3))
def time_step_RK2(W, dt, mesh, residual, **kwargs):
	F1 = residual(W, mesh, **kwargs)
	W1 = W - dt * F1
	F2 = residual(W1, mesh, **kwargs)
	W = W - dt * F2
	W = 0.5 * (W + W1)  # Heun's method
	return W

@jax.jit(static_argnums=(2,3))
def time_step_RK4(W, dt, mesh, residual, **kwargs):
	F1 = residual(W, mesh, **kwargs)
	W1 = W - dt/2 * F1
	F2 = residual(W1, mesh, **kwargs)
	W2 = W - dt/2 * F2
	F3 = residual(W2, mesh, **kwargs)
	W3 = W - dt * F3
	F4 = residual(W3, mesh, **kwargs)

	W = W - dt/6 * (F1 + 2*F2 + 2*F3 + F4)
	return W

@jax.jit(static_argnums=(2,3))
def time_step_Newton(W, dt, mesh, residual, **kwargs):
	W_old = W
	for _ in range(3):
		Fval = W - W_old + dt * residual(W, mesh, **kwargs)
		def Jv(v):
			_, jvp = jax.jvp(lambda x: residual(x, mesh, **kwargs),
						(W,),
						(v,))
			return v + dt * jvp
		delta, _ = jax.scipy.sparse.linalg.gmres(Jv, -Fval, tol=5e-3, maxiter=2, restart = 25)
		W = W + delta * 0.6  # under-relaxation to ensure convergence
	return W


@jax.jit(static_argnums=(2,3))
def SDIRK2(W, dt, mesh, residual, **kwargs):
	x = 1 - 1/jnp.sqrt(2) # singly diagonally implicit RK
	n_newton = 4
	# first step
	W1 = W
	for _ in range(n_newton):
		Fval = W1 - W + dt * residual(W1, mesh, **kwargs)
		def Jv(v):
			_, jvp = jax.jvp(lambda x: residual(x, mesh, **kwargs),
						(W1,),
						(v,))
			return v + dt * jvp
		delta, _ = jax.scipy.sparse.linalg.gmres(Jv, -Fval, tol=5e-3, maxiter=3, restart = 25)
		W1 = W1 + delta * 0.6  # under-relaxation to ensure convergence
	F1 = residual(W1, mesh, **kwargs)

	# second step
	W2 = W1
	for _ in range(n_newton):
		Fval = W2 - W + dt * (x * residual(W2, mesh, **kwargs) + (1-2 * x) * F1)
		def Jv(v):
			_, jvp = jax.jvp(lambda x: residual(x, mesh, **kwargs),
						(W2,),
						(v,))
			return v + dt * x * jvp
		delta, _ = jax.scipy.sparse.linalg.gmres(Jv, -Fval, tol=5e-3, maxiter=3, restart = 25)
		W2 = W2 + delta * 0.6  # under-relaxation to ensure convergence
	F2 = residual(W2, mesh, **kwargs)

	W = W - 0.5 * dt * (F1 + F2) 
	return W

