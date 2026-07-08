import sys
import meshio
import meshpy.triangle as triangle
import jax.numpy as jnp
import numpy as np
sys.path.append('../../..')  
import jax_fvm.src.mesh.plot as plot
import jax

import matplotlib.pyplot as plt
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

    def mesh_generator(self, info = None, maxV = 5e-3, 
                       bounds = [-1.5, 1.5, -1.5, 1.5], min_angle = 30, marker_boundary = 1):
        # info is used to create the geometry before passing to create the mesh
        # In the case there is no info, it is only a periodic square
        if info == None:
            x_min, x_max, y_min, y_max = bounds
            Lx = x_max - x_min
            Ly = y_max - y_min
            m = max(Lx / Ly, Ly / Lx)
            N_maille_x = 2 * int(np.floor(np.sqrt(1/maxV)) * Lx / m )
            N_maille_y = 2 * int(np.floor(np.sqrt(1/maxV)) * Ly / m )

            boundaries = np.array([[x, y_min] for x in np.linspace(x_min, x_max, N_maille_x)][:-1])
            markers  = [marker_boundary ] * (N_maille_x - 1)
            boundaries = np.concatenate([boundaries, np.array([[x_max, y] for y in np.linspace(y_min, y_max, N_maille_y)][:-1])])
            markers.extend([marker_boundary] * (N_maille_y - 1))
            boundaries = np.concatenate([boundaries, np.array([[x, y_max] for x in np.linspace(x_max, x_min, N_maille_x)][:-1])])
            markers.extend([marker_boundary] * (N_maille_x - 1))
            boundaries = np.concatenate([boundaries, np.array([[x_min, y] for y in np.linspace(y_max, y_min, N_maille_y)][:-1])])
            markers.extend([marker_boundary] * (N_maille_y - 1))

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
        self.face_connectivity_opposite = self.face_connectivity.copy() 
        self.set_periodic_BC() # For periodic BCs, to ensure that neighbors are correctly set, (-1 by default)

        # BC by default
        self.inlet_supersonic = jnp.array([1.0, 1.0, 0.0, 1.0])  # rho, u, v, P
        self.inlet_subsonic = jnp.array([1.0, 1., 0.0, 1.0])  # rho, u, v, P

    def mesh_generator_from_points(self, mesh_points, mesh_tris, mesh_tris_neighbors, mesh_facets, mesh_face_markers):
        self.points = jnp.array(mesh_points)
        self.tris = jnp.array(mesh_tris)
        self.neighbors = jnp.array(mesh_tris_neighbors)  # For neighbors stuff, each side with v0-V1 is next to neighbor 0
        self.faces = jnp.array(mesh_facets)
        self.face_markers = jnp.array(mesh_face_markers)

        self.field_data = jnp.zeros_like(self.points[:,0])

        self.getCenterTriangles()
        self.getArea()
        self.getSurface()
        self.getNormals()
        self.get_face_connectivity()
        self.face_connectivity_opposite = self.face_connectivity.copy() 
        self.set_periodic_BC() 

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
        face_idx_0 = jax.vmap(jax.jit(find_face_index))(sorted_edge_0)
        face_idx_1 = jax.vmap(jax.jit(find_face_index))(sorted_edge_1)
        face_idx_2 = jax.vmap(jax.jit(find_face_index))(sorted_edge_2)
        
        # Stack into result
        self.face_connectivity = jnp.stack([face_idx_0, face_idx_1, face_idx_2], axis=1)

###################################################################
################     Boundary conditions    #######################
###################################################################
    
    def set_periodic_BC(self, tol=1e-5):
        if self.points.dtype == jnp.float64:
            tol = 1e-09
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

        # Distance matrices
        face_centers = jnp.mean(self.points[self.faces[boundary_face_ids]], axis=1)  # (N_b, 2)
        face_centers_shifted = face_centers + shifts  # (N_b, 2)
        dist = jnp.linalg.norm(
            face_centers[:, None, :] - face_centers_shifted[None, :, :],
            axis=-1
        )  # (N_b, N_b)
        # id_f_opposite = jnp.where(dist < tol)[1]
        if len(boundary_face_ids) > 0:
            id_f_opposite = jnp.argmin(dist, axis=1)
        else : 
            id_f_opposite = jnp.array([], dtype=int)

        # update neighbors
        for i, boundary_face_id in enumerate(boundary_face_ids):
            id_tri = jnp.where(boundary_face_id == self.face_connectivity)[0]
            id_opposite_tri = jnp.where(boundary_face_ids[id_f_opposite[i]] == self.face_connectivity)[0]
            self.neighbors = self.neighbors.at[id_tri,jnp.where(self.face_connectivity[id_tri] == boundary_face_id)[1]].set(id_opposite_tri)
        
        # knowing a face on edge, it get the corresponding face on the opposite edge
        self.face_connectivity_opposite = jnp.arange(self.faces.shape[0])  # (N_faces,) initialized to identity
        self.face_connectivity_opposite = self.face_connectivity_opposite.at[boundary_face_ids].set(boundary_face_ids[id_f_opposite])

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
    
    def plot_contour_solution(self, field_data, *args, **kwargs):
        plot.plot_contour_solution(self, field_data, *args, **kwargs)

    def animate_field(self, field_sequence, path = "animation.gif", interval=100):
        plot.animate_solution(self, field_sequence, path = path, interval=interval)

    def plot_slice(self, X, y = 0.5, n = 1000, labels = r'$\rho$'):
        out, slice = plot.getSliceinMesh(X, self, y = y, n = n)
        fig, ax = plt.subplots()
        ax.plot(slice[...,0], out)
        ax.grid()
        ax.set_xlabel(r'$x$')
        ax.set_ylabel(labels)

###################################################################
##################     Helper functions     #######################
###################################################################

    def index_nodes_to_faces(self):
        _, id_tris_to_faces, id_faces_to_tris = jnp.unique(self.face_connectivity.reshape(-1), return_index = True, return_inverse=True)
        return id_tris_to_faces, id_faces_to_tris
    
    def map_faces_to_left_right_triangles(self):
        """
        For each unique face interface, return the left and right triangle indices.
        Right is read from mesh.neighbors. Periodic face pairs (f, f_opp) are
        deduplicated via face_connectivity_opposite — each interface appears once.

        Returns:
            left_tri:  (N_interfaces,) int array
            right_tri: (N_interfaces,) int array
        """
        N_tris = self.face_connectivity.shape[0]
        N_faces = self.faces.shape[0]

        face_flat = np.array(self.face_connectivity).reshape(-1)  # (3*N_tris,)
        tri_flat = np.repeat(np.arange(N_tris), 3)
        neighbor_flat = np.array(self.neighbors).reshape(-1)       # (3*N_tris,)

        order = np.argsort(face_flat, kind='stable')
        face_sorted = face_flat[order]
        tri_sorted = tri_flat[order]
        neighbor_sorted = neighbor_flat[order]

        is_first = np.concatenate([[True], face_sorted[1:] != face_sorted[:-1]])

        left_tri = np.full(N_faces, -1, dtype=np.int32)
        right_tri = np.full(N_faces, -1, dtype=np.int32)
        left_tri[face_sorted[is_first]] = tri_sorted[is_first]
        right_tri[face_sorted[is_first]] = neighbor_sorted[is_first]

        return left_tri, right_tri



if __name__ == "__main__":
    mesh = Mesh()
    mesh.mesh_generator(maxV=3e-3, marker_boundary=1, bounds=[-2., 2., -1., 1.], min_angle=30)
    mesh.plot_mesh()



