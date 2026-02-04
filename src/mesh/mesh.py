import sys
import meshio
import meshpy.triangle as triangle
import jax.numpy as jnp
import numpy as np
sys.path.append('../../..')  
import FVM.src.mesh.plot as plot
import jax


class Mesh:
    """ Class Mesh to handle the mesh generation and storage of mesh data """
    """
     Attributes:
        points: array of shape (N_points, 2) containing the coordinates of the points
        tris: array of shape (N_tris, 3) containing the indices of the points forming each triangle
        neighbors: array of shape (N_tris, 3) containing the indices of the neighboring triangles for each triangle
        faces: array of shape (N_faces, 2) containing the indices of the points forming each face
        face_markers: array of shape (N_faces,) containing the markers for each face
        field_data: array of shape (N_points,) containing the field data at each point
        barycenter: array of shape (N_tris, 2) containing the barycenter of each triangle
        area: array of shape (N_tris,) containing the area of each triangle
        surface: array of shape (N_faces,) containing the length of each face
        normals: array of shape (N_tris, 3, 2) containing the normals of each face of each triangle
        midedge: array of shape (N_faces, 2) containing the midpoints of each face
        
    access via mesh.points, mesh.tris, etc.

        BCs handled via face_markers and neighbors:
              - BC_marker = 1 : periodic BCs
              - BC_marker = 2 : wall BCs (no-slip)
              - BC_marker = 3 : inlet BCs (prescribed values)
              - BC_marker = 4 : outlet BCs (free flow)
              - BC_marker = 5 : subsonic inflow 

    """
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

    def round_trip_connect(self, start, end):
        result = []
        for i in range(start, end):
            result.append((i, i+1))
        result.append((end, start))
        return result

    def mesh_generator(self, info = None, maxV = 5e-3, Lx = 1., Ly = 1.,
					    min_angle = 30, marker_boundary = 1):
        # info is used to create the geometry before passing to create the mesh
        # In the case there is no info, it is only a periodic square
        if info == None:
            N_maille = int(np.floor(np.sqrt(1/maxV)))
            boundaries = np.array([[x, 0] for x in np.linspace(0,Lx,N_maille)][:-1])
            markers  = [marker_boundary ] * (N_maille - 1)
            boundaries = np.concatenate([boundaries, np.array([[Lx, y] for y in np.linspace(0,Ly,N_maille)][:-1])])
            markers.extend([marker_boundary] * (N_maille - 1))
            boundaries = np.concatenate([boundaries, np.array([[x, Ly] for x in np.linspace(Ly,0,N_maille)][:-1])])
            markers.extend([marker_boundary] * (N_maille - 1))
            boundaries = np.concatenate([boundaries, np.array([[0, y] for y in np.linspace(Lx,0,N_maille)][:-1])])
            markers.extend([marker_boundary] * (N_maille-1))

            info = triangle.MeshInfo()
            info.set_points(boundaries)
            info.set_facets(self.round_trip_connect(0, len(boundaries)-1), facet_markers=markers)


        # Create mesh
        mesh = triangle.build(info, max_volume=maxV, min_angle=min_angle, generate_faces=True)

        self.points = jnp.array(mesh.points)
        self.tris = jnp.array(mesh.elements)
        self.neighbors = jnp.roll(jnp.array(mesh.neighbors), 1, axis=-1)  # For neighbors stuff, each side with v0-V1 is next to neighbor 0
        self.faces = jnp.array(mesh.faces)
        self.face_markers = jnp.array(mesh.face_markers)

        self.field_data = jnp.zeros_like(self.points[:,0])

        self.getCenterTriangles()
        self.getArea()
        self.getSurface()
        self.getNormals()
        self.get_face_connectivity()
        self.set_periodic_BC() # For periodic BCs, to ensure that neighbors are correctly set, (-1 by default)

        # BC by default
        self.inlet_supersonic = jnp.array([1.0, 1.0, 0.0, 1.0])  # rho, u, v, P
        self.inlet_subsonic = jnp.array([1.0, 1., 0.0, 1.0])  # rho, u, v, P

###################################################################
###################     Utilities    ##############################
###################################################################

    def getCenterTriangles(self):
        self.barycenter = jnp.mean(self.points[self.tris], axis=-2)

    def getNormals(self):
        diff = jnp.roll(self.points[self.tris], -1, axis = -2) - self.points[self.tris]
        norm = jnp.linalg.norm(diff, axis=-1)
        normals = jnp.stack([diff[...,1], -diff[...,0]], -1)
        self.normals = normals / norm[..., None] # For neighbors stuff

    def getArea(self):
        points = self.points[self.tris]
        area = 0.5 * ((points[...,1,1] - points[...,0,1]) * (points[...,2,0] - points[...,0,0]) - 
                            (points[...,1,0] - points[...,0,0]) * (points[...,2,1] - points[...,0,1]))
        self.area = jnp.abs(area)
        del area, points

    def getSurface(self):
        self.surface = jnp.linalg.norm(self.points[self.faces[:,1]] - self.points[self.faces[:,0]], axis=1)

    def get_face_connectivity(self):
        """ For each triangle, find the indices of the faces that form its edges.
            The order of the faces corresponds to the order of the edges:
            Edge 0: between vertex 0 and 1
            Edge 1: between vertex 1 and 2
            Edge 2: between vertex 2 and 0"""
        N_f = self.faces.shape[0]
        
        # Create sorted faces for matching (since edge (i,j) == edge (j,i))
        sorted_faces = jnp.sort(self.faces, axis=1)
        
        # For each triangle, get the 3 edges in order
        # Edge 0: between vertex 0 and 1
        # Edge 1: between vertex 1 and 2
        # Edge 2: between vertex 2 and 0
        edge_0 = jnp.stack([self.tris[:, 0], self.tris[:, 1]], axis=1)
        edge_1 = jnp.stack([self.tris[:, 1], self.tris[:, 2]], axis=1)
        edge_2 = jnp.stack([self.tris[:, 2], self.tris[:, 0]], axis=1)
        
        # Sort edges for matching
        sorted_edge_0 = jnp.sort(edge_0, axis=1)
        sorted_edge_1 = jnp.sort(edge_1, axis=1)
        sorted_edge_2 = jnp.sort(edge_2, axis=1)
        
        def find_face_index(edge):
            """Find which face matches this edge"""
            # Compare edge with all faces
            matches = jnp.all(sorted_faces == edge[None, :], axis=1)
            # Get index of match, or -1 if not found
            idx = jnp.where(matches, jnp.arange(N_f), -1)
            return jnp.max(idx)  # Will be -1 if no match found
        
        # Vectorized version to find all face indices
        face_idx_0 = jax.vmap(find_face_index)(sorted_edge_0)
        face_idx_1 = jax.vmap(find_face_index)(sorted_edge_1)
        face_idx_2 = jax.vmap(find_face_index)(sorted_edge_2)
        
        # Stack into result
        self.face_connectivity = jnp.stack([face_idx_0, face_idx_1, face_idx_2], axis=1)

###################################################################
################     Boundary conditions    #######################
###################################################################
    
    def set_periodic_BC(self, tol=1e-6):
        # Domain bounds
        x_min, x_max = self.points[:, 0].min(), self.points[:, 0].max()
        y_min, y_max = self.points[:, 1].min(), self.points[:, 1].max()
        dx, dy = x_max - x_min, y_max - y_min

        # Boundary faces
        faces_mask = self.face_markers == 1
        boundary_face_ids = jnp.where(faces_mask)[0]

        diff = self.points[self.faces[boundary_face_ids]][:,1,:] - self.points[self.faces[boundary_face_ids]][:,0,:]

        # Classify boundaries
        is_left = (jnp.abs(diff[:, 0]) < tol) & (jnp.abs(self.points[self.faces[boundary_face_ids]][:, 0, 0] - x_min) < tol)
        is_right = (jnp.abs(diff[:, 0]) < tol) & (jnp.abs(self.points[self.faces[boundary_face_ids]][:, 0, 0] - x_max) < tol)
        is_bottom = (jnp.abs(diff[:, 1]) < tol) & (jnp.abs(self.points[self.faces[boundary_face_ids]][:, 0, 1] - y_min) < tol)
        is_top = (jnp.abs(diff[:, 1]) < tol) & (jnp.abs(self.points[self.faces[boundary_face_ids]][:, 0, 1] - y_max) < tol)

        # Compute shifts
        shifts = (is_left[:, None] * jnp.array([dx, 0.]) +  
                    is_right[:, None] * jnp.array([-dx, 0.]) +
                    is_bottom[:, None] * jnp.array([0., dy]) +
                    is_top[:, None] * jnp.array([0., -dy]))

        opposite_points = self.points[self.faces[boundary_face_ids]] + shifts[:, None, :]

        # Distance matrices
        dist = jnp.linalg.norm(
            self.points[self.faces[boundary_face_ids]][:, None, :, :] - opposite_points[None, :, :, :],
            axis=-1
        )  # (N_b, N_b, 2)

        dist_shifted = jnp.linalg.norm(
            self.points[self.faces[boundary_face_ids]][:, None, :, :] - jnp.roll(opposite_points[None, :, :, :], 1, axis=-2),
            axis=-1
        )

        # One point must match
        id_f_opposite = jnp.where((jnp.sum(dist == 0, axis = -1) == 2) | (jnp.sum(dist_shifted == 0, axis = -1) == 2))[1]

        # update neighbors
        for i, boundary_face_id in enumerate(boundary_face_ids):
            id_tri = jnp.where(boundary_face_id == self.face_connectivity)[0]
            id_opposite_tri = jnp.where(boundary_face_ids[id_f_opposite[i]] == self.face_connectivity)[0]
            self.neighbors = self.neighbors.at[id_tri,jnp.where(self.face_connectivity[id_tri] == boundary_face_id)[1]].set(id_opposite_tri)

###################################################################
###################     mesh vtk     ##############################
###################################################################

    def save_mesh(self, filename="./mesh.vtk"):
        cells = [("triangle", self.tris)]
        mesh = meshio.Mesh(
            self.points,
            cells,
        )
        mesh.write(filename)

    def load_mesh(self, filename="./mesh.vtk"):
        raise NotImplementedError("Loading mesh not implemented yet.")

###################################################################
###################     plotting     ##############################
###################################################################

    def plot_mesh(self):
        plot.plot_mesh(self)

    def plot_solution(self, field_data, *args, **kwargs):
        plot.plot_solution(self, field_data, *args, **kwargs)

    def animate_field(self, field_sequence, interval=100):
        plot.animate_solution(self, field_sequence, interval=interval)



# mesh = Mesh()
# mesh.mesh_generator(maxV=2e-4, marker_boundary=1)
# mesh.plot_mesh()


