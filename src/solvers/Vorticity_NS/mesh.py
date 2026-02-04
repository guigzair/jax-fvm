import sys
import jax.numpy as jnp
import numpy as np
import jax
import matplotlib.pyplot as plt
import matplotlib.animation as animation
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



class Mesh:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

    def mesh_generator(self, N = 128, L = 2 * jnp.pi):
        self.L = L
        self.N = N
        x = jnp.linspace(0, L, N, endpoint=False)
        y = jnp.linspace(0, L, N, endpoint=False)
        self.X, self.Y = jnp.meshgrid(x, y, indexing="ij")

        # -----------------------------
        # Fourier wavenumbers
        # -----------------------------
        k = jnp.fft.fftfreq(N, L/N) * 2 * jnp.pi
        self.kx, self.ky = jnp.meshgrid(k, k, indexing="ij")
        self.k2 = self.kx**2 + self.ky**2
        self.k2 = self.k2.at[0,0].set(1.0)


        # -----------------------------
        # Dealias mask (2/3 rule)
        # -----------------------------
        kmax = jnp.max(jnp.abs(k))
        self.dealias = (jnp.abs(self.kx) < 2/3*kmax) & (jnp.abs(self.ky) < 2/3*kmax)

    def plot_field(self, field, clb_title=r"$\omega$"):
        fig, ax = plt.subplots()
        c = ax.pcolormesh(self.X, self.Y, field, shading='auto', cmap='viridis')
        clb = fig.colorbar(c)
        clb.ax.set_title(clb_title)

    def animate_field(self, field_sequence, interval=100, clb_title=r"$\omega$"):
        fig, ax = plt.subplots(figsize=(6,5))

        im = ax.imshow(field_sequence[0]/jnp.max(jnp.abs(field_sequence[0])),
                    origin="lower",
                    extent=[0, self.L, 0, self.L])
        ax.set_title("2D Navier-Stokes Vorticity")
        plt.colorbar(im, ax=ax)

        def update(frame):
            im.set_data(field_sequence[frame]/jnp.max(jnp.abs(field_sequence[frame])))
            return [im]

        ani = animation.FuncAnimation(fig, update, frames=len(field_sequence), interval=interval)

        ani.save('vorticity_evolution.gif', writer='imagemagick')