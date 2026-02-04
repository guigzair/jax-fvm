import jax.numpy as jnp
import numpy as np
import jax
import sys
sys.path.append('../../..')  
from FVM.src.mesh.mesh import Mesh 
import meshpy.triangle as triangle


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
    def build(self, h = 1e-3, L = 1.):
        mesh = Mesh()
        Lx = L / 2
        Ly = L 

        N_maille_x = int(np.floor(Lx * np.sqrt(1/h)))
        N_maille_y = int(np.floor(Ly * np.sqrt(1/h)))
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

        mesh.mesh_generator(info = info, maxV=h)

        return mesh
    

class TestMovingVortex():
    def build(self, h = 1e-3, L = 0.1):
        mesh = Mesh()
        L = L

        N_maille = int(np.floor(L * np.sqrt(1/h)))
        boundaries = np.array([[x, 0] for x in np.linspace(0,L,N_maille)][:-1])
        markers  = [1] * (N_maille - 1)
        boundaries = np.concatenate([boundaries, np.array([[L, y] for y in np.linspace(0,L,N_maille)][:-1])])
        markers.extend([1] * (N_maille - 1))
        boundaries = np.concatenate([boundaries, np.array([[x, L] for x in np.linspace(L,0,N_maille)][:-1])])
        markers.extend([1] * (N_maille - 1))
        boundaries = np.concatenate([boundaries, np.array([[ 0, y] for y in np.linspace(L,0,N_maille)][:-1])])
        markers.extend([1] * (N_maille-1))

        info = triangle.MeshInfo()
        info.set_points(boundaries)
        info.set_facets(mesh.round_trip_connect(0, len(boundaries)-1), facet_markers=markers)

        mesh.mesh_generator(info = info, maxV=h)
        return mesh
    
    

