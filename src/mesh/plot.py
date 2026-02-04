import matplotlib.pyplot as plt
from matplotlib import tri as mtri
import numpy as np
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
import jax.numpy as jnp

from matplotlib.pyplot import cm
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

color = cm.rainbow(np.linspace(0, 1, 10))


def plot_mesh(mesh):
    triang = mtri.Triangulation(mesh.points[:, 0], mesh.points[:, 1], mesh.tris)

    fig, ax = plt.subplots(dpi = 500, figsize=(6,6))
    ax.triplot(triang, color='black', lw=0.5)
    for bc_marker in jnp.unique(mesh.face_markers)[jnp.where(jnp.unique(mesh.face_markers) > 0)[0]]:
        ids = jnp.where(mesh.face_markers == bc_marker)[0]
        for id in ids:
            ax.plot(mesh.points[mesh.faces[id]][...,0], mesh.points[mesh.faces[id]][...,1], 
                    c=color[bc_marker], lw=1.5)
    ax.set_aspect('equal')
    ax.set_xlabel(r'$x$')
    ax.set_ylabel(r'$y$')
    ax.set_title('Mesh')
    

def plot_solution(mesh, field_data, labels = r'$\rho$'):
    xmin = mesh.points[:,0].min()
    xmax = mesh.points[:,0].max()
    ymin = mesh.points[:,1].min()
    ymax = mesh.points[:,1].max()

    triang = mtri.Triangulation(mesh.points[:, 0], mesh.points[:, 1], mesh.tris)

    fig, ax = plt.subplots(dpi = 500)
    ax.set_aspect('equal')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    tpc = ax.tripcolor(triang, facecolors = field_data)
    ax.set_xlabel(r'$x$')
    ax.set_ylabel(r'$y$')
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    clb = fig.colorbar(tpc, cax = cax)
    clb.ax.set_title(labels)
    plt.tight_layout()


def animate_solution(mesh, field_sequence, interval=200):
    triang = mtri.Triangulation(mesh.points[:, 0], mesh.points[:, 1], mesh.tris)

    xmin = mesh.points[:,0].min()
    xmax = mesh.points[:,0].max()
    ymin = mesh.points[:,1].min()
    ymax = mesh.points[:,1].max()

    fig, ax = plt.subplots()
    ax.set_aspect('equal')
    ax.set_xlabel(r'$x$')
    ax.set_ylabel(r'$y$')
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    def update(frame):
        ax.tripcolor(triang, facecolors = field_sequence[frame])

    ani = animation.FuncAnimation(fig, update, frames=len(field_sequence), interval=interval)

    ani.save('vorticity_evolution.gif', writer='imagemagick')

def plot(y, labels = r'$E_k$'):
    fig, ax = plt.subplots()
    ax.plot(y)
    ax.grid()
    ax.set_xlabel(r'$t$')
    ax.set_ylabel(labels)