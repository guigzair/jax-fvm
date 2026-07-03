import jax.numpy as jnp
import numpy as np
import jax
import sys

from scipy.datasets import face
sys.path.append('../../..')  
from jax_fvm.src.mesh.mesh import Mesh 
import meshpy.triangle as triangle
from scipy.spatial import Delaunay


# Forward facing step 
class Forward_Step:
    def build(self, h = 1e-3):
        mesh = Mesh()

        boundaries = [[0,0],[0.6,0],[0.6,0.2],[3,0.2], [3,1], [0,1]]
        markers = [2, 2, 2, 4, 2, 4]
        facets = mesh.round_trip_connect(0, len(boundaries) - 1)
        info = triangle.MeshInfo()
        info.set_points(boundaries)
        info.set_holes([(0.5, 0)])
        info.set_facets(facets, facet_markers=markers)

        mesh.mesh_generator(info, maxV=h)

        return mesh


class TestDipoleVortex():
    def __init__(self, maxV = 8e-5, L = 1.0):
        self.h = maxV
        self.L = L
    def build(self):
        mesh = Mesh()
        Lx = self.L / 2
        Ly = self.L

        N_maille_x = int(np.floor(Lx * np.sqrt(1/self.h)))
        N_maille_y = int(np.floor(Ly * np.sqrt(1/self.h)))
        # N_maille = int(np.floor(L * np.sqrt(1/h)))
        boundaries = np.array([[x, 0] for x in np.linspace(0,Lx,N_maille_x)][:-1])
        markers  = [1] * (N_maille_x - 1)
        boundaries = np.concatenate([boundaries, np.array([[Lx, y] for y in np.linspace(0,Ly,N_maille_y)][:-1])])
        markers.extend([2] * (N_maille_y - 1))
        boundaries = np.concatenate([boundaries, np.array([[x, Ly] for x in np.linspace(Lx,0,N_maille_x)][:-1])])
        markers.extend([1] * (N_maille_x - 1))
        boundaries = np.concatenate([boundaries, np.array([[ 0, y] for y in np.linspace(Ly,0,N_maille_y)][:-1])])
        markers.extend([2] * (N_maille_y - 1))

        info = triangle.MeshInfo()
        info.set_points(boundaries)
        info.set_facets(mesh.round_trip_connect(0, len(boundaries)-1), facet_markers=markers)

        mesh.mesh_generator(info = info, maxV=self.h)

        return mesh
    

class UniformMesh():
    def __init__(self, Nx=10, Ny=10, Lx=1.0, Ly=1.0):
        self.Nx = Nx
        self.Ny = Ny
        self.Lx = Lx
        self.Ly = Ly

    def build(self):
        mesh = Mesh()

        Lx = self.Lx
        Ly = self.Ly
        Nx = self.Nx
        Ny = self.Ny
        dx = Lx / Nx
        dy = Ly / Ny

        # Generate point cloud
        x = np.linspace(0, Lx, Nx)
        y = np.linspace(0, Ly, Ny)  
        X, Y = np.meshgrid(x, y)
        points = np.c_[X.ravel(), Y.ravel()]

        # to have squares divided in 4 triangles, we need to add points at the center of each square
        x = np.linspace(0 + dx/2 , 1. - dx/2, Nx-1)
        y = np.linspace(0 + dy/2 , 1. - dy/2, Ny-1) 
        X, Y = np.meshgrid(x, y)
        points = np.concatenate([points, np.c_[X.ravel(), Y.ravel()]])

        # Generate triangulation
        tri = Delaunay(points)

        mesh_tris = jnp.array(tri.simplices, dtype=jnp.int32)
        mesh_points = jnp.array(tri.points, dtype=jnp.float32)
        mesh_tris_neighbors = jnp.array(tri.neighbors, dtype=jnp.int32)

        # Define boundary facets and markers
        face_1 = jnp.stack((tri.simplices[:, 0], tri.simplices[:, 1]), axis=-1)
        face_2 = jnp.stack((tri.simplices[:, 1], tri.simplices[:, 2]), axis=-1)
        face_3 = jnp.stack((tri.simplices[:, 2], tri.simplices[:, 0]), axis=-1)
        faces = jnp.concatenate((face_1, face_2, face_3), axis=0)
        mesh_tris_neighbors = jnp.roll(jnp.array(mesh_tris_neighbors), 1, axis=-1)
        face_markers = jnp.where(mesh_tris_neighbors == -1, 1, 0)
        face_markers = jnp.concatenate((face_markers[:, 0], face_markers[:, 1], face_markers[:, 2]), axis=0)

        faces = jnp.sort(faces, axis=-1)
        faces, indices = jnp.unique(faces, axis=0, return_index=True)
        faces_markers = face_markers[indices]

        mesh.mesh_generator_from_points(mesh_points, mesh_tris, mesh_tris_neighbors, faces, faces_markers)
        return mesh


if __name__ == "__main__":
    mesh = UniformMesh(Nx=15, Ny=15, Lx=1.0, Ly=1.0).build()
    mesh.plot_mesh()



