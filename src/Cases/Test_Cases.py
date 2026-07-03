import jax.numpy as jnp
import numpy as np
import meshpy.triangle as triangle
from dataclasses import dataclass

# This file is used to implement all test cases present in lax paper
# SOLUTION OF TWO-DIMENSIONAL RIEMANN PROBLEMS OF GAS DYNAMICS BY POSITIVE SCHEMES, Lax, Liu, 1998
# And some more cases particular to the database

#########################################################
#                           #                           #
#                           #                           #
#                           #                           #
#                           #                           #
#               2           #               1           #
#                           #                           #
#                           #                           #
#                           #                           #
#                           #                           #
#                           #                           #      
#########################################################
#                           #                           #
#                           #                           #
#                           #                           #
#                           #                           #
#               3           #               4           #
#                           #                           #
#                           #                           #
#                           #                           #
#                           #                           #
#                           #                           #    
##########################################################

##########################################################

def round_trip_connect(start, end):
    result = []
    for i in range(start, end):
        result.append((i, i+1))
    result.append((end, start))
    return result


def createCylinderMesh(maxV = 8e-3):
    N_maille = int(np.floor(np.sqrt(1/maxV)))
    # N_maille = 10
    r = 0.25
    n = int(N_maille * 2 * np.pi * r)
    angle = np.linspace(0, n-1, n-1) * 2 * np.pi / n
    points = np.vstack((0.6 + r * np.cos(angle), r * np.sin(angle))).T.tolist()



    facets = round_trip_connect(0, len(points) - 1)
    markers  = [2] * len(points)

    outter_start = len(points)
    L = 1


    points.extend([[-1,-L],[L + 2,-L],[L + 2,L],[-1,L]])
    facets.extend(round_trip_connect(outter_start, len(points) - 1))
    markers.extend([2,3,2,3])

    # info on the mesh
    info = triangle.MeshInfo()
    info.set_points(points)
    info.set_holes([(0.6,0)])
    info.set_facets(facets, facet_markers=markers)
    return info

##########################################################

class TestDividing():
    def __init__(self, left, right):
        self.left = left
        self.right = right

    def build(self, mesh):
        N = len(mesh.area)
        Primitives = jnp.zeros((N, 4))
        for i in range(N):
            if mesh.barycenter[i,0] > 0.5:
                Primitives[i] = self.right
            elif mesh.barycenter[i,0] <= 0.5:
                Primitives[i] = self.left
        return Primitives

def DoubleMachReflection():
    left = jnp.array([8., 8.25, 0, 116.5])
    # left = jnp.array([1.4, 3., 0, 1.])
    right = jnp.array([1.4, 0., 0, 1.])
    return TestDividing(left, right)


class TestUniform():
    def __init__(self, state):
        self.state = state

    def build(self, mesh):
        N = len(mesh.area)
        Primitives = jnp.zeros((N, 4))
        Primitives = Primitives.at[...].set(self.state)
        return Primitives

def ForwardFacingStep():
    state = jnp.array([1.4, 3., 0, 1.])
    return TestUniform(state)

##########################################################

class TestDipoleVortex():
    def __init__(self, R = 0.05, beta = 1/50, mach = 0.05):
        self.R = R
        self.beta = beta
        self.mach = mach

    def build(self, mesh):
        N = len(mesh.area)
        L = jnp.max(mesh.points[...,0]) - jnp.min(mesh.points[...,0])

        T_inf = 300.
        P_inf = 1e5
        R = 287.15
        C_p = R * 1.4 / (1.4 - 1)
        T_inf = 300
        U_inf = self.mach * jnp.sqrt(1.4 * R * T_inf)  # assuming T_inf = 300K
        rho_inf = P_inf / (R * T_inf)
        U_0 = jnp.sqrt(1.4 * R * T_inf)

        pt_1 = jnp.mean(mesh.points, axis = 0) + jnp.array([0., -0.1*L])
        pt_2 = jnp.mean(mesh.points, axis = 0) + jnp.array([0.,  0.1*L])

        r1 = jnp.linalg.norm(mesh.barycenter - pt_1, axis = -1)
        r2 = jnp.linalg.norm(mesh.barycenter - pt_2, axis = -1)

        def velocity_field(x, y):
            v = U_inf * self.beta / self.R  *( (x-pt_1[0])  * jnp.exp(-(r1/self.R)**2) - (x-pt_2[0]) * jnp.exp(-(r2/self.R)**2))
            u = U_inf * (0.01 + self.beta / self.R  *( - (y-pt_1[1])  * jnp.exp(-(r1/self.R)**2) + (y-pt_2[1]) * jnp.exp(-(r2/self.R)**2)))
            return u, v
        
        def temperature_field():
            T = T_inf - (self.beta * U_inf ) ** 2 /(2 * C_p) * (jnp.exp(-(r1/self.R)**2) + jnp.exp(-(r2/self.R)**2))
            return T
        
        u, v = velocity_field(mesh.barycenter[:,0], mesh.barycenter[:,1])
        T = temperature_field()
        rho = rho_inf * (T/T_inf)**(1/(1.4-1))
        p = rho * R * T
        
        Primitives = jnp.zeros((N, 4))
        # Primitives = Primitives.at[...,0].set(rho / rho_inf)
        # Primitives = Primitives.at[...,1].set(u / U_0)
        # Primitives = Primitives.at[...,2].set(v / U_0)
        # Primitives = Primitives.at[...,3].set(p / (rho_inf * U_0**2))

        Primitives = Primitives.at[...,0].set(rho )
        Primitives = Primitives.at[...,1].set(u )
        Primitives = Primitives.at[...,2].set(v )
        Primitives = Primitives.at[...,3].set(p )


        # BCs
        mesh.inlet_subsonic = jnp.array([rho_inf, 0.01 * U_inf, 0.0, P_inf])  # rho, u, v, P
        return Primitives, mesh

class TestDipoleVortex2():
    def __init__(self, R = 0.05, omega = 300, mach = 0.01, rho_0 = 1.0):
        self.R = R
        self.omega = omega
        self.mach = mach
        self.rho_0 = rho_0

    def build(self, mesh):
        N = len(mesh.area)
        L = jnp.max(mesh.points[...,0]) - jnp.min(mesh.points[...,0])


        pt_1 = jnp.mean(mesh.points, axis = 0) + jnp.array([0., self.R])
        pt_2 = jnp.mean(mesh.points, axis = 0) + jnp.array([0.,-self.R])

        r1 = jnp.linalg.norm(mesh.barycenter - pt_1, axis = -1)
        r2 = jnp.linalg.norm(mesh.barycenter - pt_2, axis = -1)

        def velocity_field(x, y):
            u =  self.omega / 2  *( - (y-pt_1[1])  * jnp.exp(-(r1/self.R)**2) + (y-pt_2[1]) * jnp.exp(-(r2/self.R)**2))
            v = self.omega / 2  *( (x-pt_1[0])  * jnp.exp(-(r1/self.R)**2) - (x-pt_2[0]) * jnp.exp(-(r2/self.R)**2))
            return u, v
        
        def pressure_field():
            p = (self.omega * self.R / 4)**2 * (jnp.exp(-2*(r1/self.R)**2) + jnp.exp(-2*(r2/self.R)**2))
            return p
        
        u, v = velocity_field(mesh.barycenter[:,0], mesh.barycenter[:,1])
        p_0 = 1 / (1.4 * self.mach**2) #U_0 ** 2 * self.rho_0 / (1.4 * self.mach**2)
        p = p_0 - pressure_field()
        rho = self.rho_0 * (p / p_0)**(1/1.4)
        
        Primitives = jnp.zeros((N, 4))
        Primitives = Primitives.at[...,0].set(rho)
        Primitives = Primitives.at[...,1].set(u)
        Primitives = Primitives.at[...,2].set(v )
        Primitives = Primitives.at[...,3].set(p )
        return Primitives

class TestMovingVortex():
    def __init__(self, R = 0.005, beta = 1/50, mach = 0.05):
        self.R = R
        self.beta = beta
        self.mach = mach

    def build(self, mesh):
        T_inf = 300.
        P_inf = 1e5
        R = 287.15
        C_p = R * 1.4 / (1.4 - 1)
        U_inf = self.mach * jnp.sqrt(1.4 * R * T_inf)
        rho_inf = P_inf / (R * T_inf)
        N = len(mesh.area)
        U_0 = jnp.sqrt(1.4 * R * T_inf)

        pt_1 = jnp.mean(mesh.points, axis = 0)

        r1 = jnp.linalg.norm(mesh.barycenter - pt_1, axis = -1)

        def velocity_field(x, y):
            v = U_inf * (self.beta * (x-pt_1[0]) / self.R  * jnp.exp(-(r1/self.R)**2))
            u = U_inf * (1 - self.beta * (y-pt_1[1]) / self.R * jnp.exp(-(r1/self.R)**2))
            return u, v
        
        def temperature_field():
            T = T_inf - (self.beta * U_inf ) ** 2 /(2 * C_p) * jnp.exp(-(r1/self.R)**2)
            return T

        Primitives = jnp.zeros((N, 4))
        u, v = velocity_field(mesh.barycenter[:,0], mesh.barycenter[:,1])
        T = temperature_field()
        rho = rho_inf * (T/T_inf)**(1/(1.4-1))
        p = rho * R * T

        # Primitives = Primitives.at[...,0].set(rho / rho_inf)
        # Primitives = Primitives.at[...,1].set(u / U_0)
        # Primitives = Primitives.at[...,2].set(v / U_0)
        # Primitives = Primitives.at[...,3].set(p / (rho_inf * U_0**2))
        Primitives = Primitives.at[...,0].set(rho)
        Primitives = Primitives.at[...,1].set(u)
        Primitives = Primitives.at[...,2].set(v)
        Primitives = Primitives.at[...,3].set(p)

        return Primitives
    
class TestMovingAdvection():
    def __init__(self, R = 0.1):
        self.R = R

    def build(self, mesh):
        N = len(mesh.area)
        Primitives = jnp.zeros((N, 4))

        rho = 1.0 + 0.5 * jnp.exp(-((mesh.barycenter[:,0]-0.5)**2 + (mesh.barycenter[:,1]-0.5)**2)/self.R**2)
        Primitives = Primitives.at[...,0].set(rho)
        Primitives = Primitives.at[...,1].set(1.)
        Primitives = Primitives.at[...,2].set(0.)
        Primitives = Primitives.at[...,3].set(1.)
        return Primitives
    
class TaylorGreenVortex():
    def __init__(self, V_0 = 2., M = 0.1):
        self.V_0 = V_0
        self.M = M

    def build(self, mesh):
        N = len(mesh.area)
        L = jnp.max(mesh.points[...,0]) - jnp.min(mesh.points[...,0])

        def velocity_field(x, y):
            u =  - self.V_0 *  jnp.sin(2 * jnp.pi * x / L) * jnp.cos(2 * jnp.pi * y / L)
            v =  self.V_0 * jnp.cos(2 * jnp.pi * x / L) * jnp.sin(2 * jnp.pi * y / L)
            p_0 = self.V_0**2 / (1.4 * self.M**2) # assuming U_0 = 1
            p = p_0 + self.V_0**2 / 16 * (jnp.cos(4 * jnp.pi * x / L) + jnp.cos(4 * jnp.pi * y / L))
            rho = p/p_0
            return rho, u, v , p
        
        Primitives = jnp.zeros((N, 4))
        rho, u, v, p = velocity_field(mesh.barycenter[:,0], mesh.barycenter[:,1])
        Primitives = Primitives.at[...,0].set(rho)
        Primitives = Primitives.at[...,1].set(u)
        Primitives = Primitives.at[...,2].set(v)
        Primitives = Primitives.at[...,3].set(p)
        return Primitives
    
class KevinHelmotzInstability():
    def __init__(self, sigma = 1., alpha = 1.):
        self.sigma = sigma
        self.alpha = alpha

    def build(self, mesh):
        N = len(mesh.area)

        def velocity_field(x, y):
            # rho = jnp.where(jnp.logical_and(y > 0.25, y < 0.75), 2., 1.)
            # u = jnp.where(jnp.logical_and(y > 0.25, y < 0.75), 0.5, -0.5)
            # v = self.alpha * jnp.sin(4 * jnp.pi * y) * (jnp.exp(- (y - 0.25)**2/(self.sigma**2)) + jnp.exp(- (y - 0.75)**2/(self.sigma**2)))
            rho = 1 + 1/ (1 + jnp.exp(- (y - 0.25)/self.sigma**2)) - 1/ (1 + jnp.exp(- (y - 0.75)/self.sigma**2))
            u = 1/ (1 + jnp.exp(- (y - 0.25)/self.sigma**2)) - 1/ (1 + jnp.exp(- (y - 0.75)/self.sigma**2)) - 1/2
            v = self.alpha * jnp.sin(2 * jnp.pi * 2 * (x - 0.5)) * (jnp.exp(- 2 * (y - 0.25)**2/(self.sigma**2)) - jnp.exp(- 2 * (y - 0.75)**2/(self.sigma**2)))
            p = 2.5
            return rho, u, v , p
        
        Primitives = jnp.zeros((N, 4))
        rho, u, v, p = velocity_field(mesh.barycenter[:,0], mesh.barycenter[:,1])
        Primitives = Primitives.at[...,0].set(rho)
        Primitives = Primitives.at[...,1].set(u)
        Primitives = Primitives.at[...,2].set(v)
        Primitives = Primitives.at[...,3].set(p)
        return Primitives

# class MergingPair():
#     def __init__(self):
#         self.mach = 0.01
#         self.a = 1.0

#     def build(self, mesh):
#         r1 = jnp.array([-1.5, 0.])
#         r2 = jnp.array([1.5, 0.])

#         def velocity(x, y, centers=(r1, r2), Gamma=jnp.pi):
#             u = jnp.zeros_like(x); v = jnp.zeros_like(x)
#             for (xk, yk) in centers:
#                 dx, dy = x - xk, y - yk
#                 r2 = dx*dx + dy*dy
#                 # u_theta / r, finite at r=0
#                 f = (Gamma/(2*jnp.pi)) * (1 - jnp.exp(-r2/self.a**2)) / jnp.where(r2>0, r2, 1.0)
#                 f = jnp.where(r2>0, f, Gamma/(2*jnp.pi*self.a**2))  # r->0 limit
#                 u += -f*dy
#                 v +=  f*dx
#             return u, v
        
#         u, v = velocity(mesh.barycenter[:,0], mesh.barycenter[:,1])
#         p_0 = 1 / (1.4 * self.mach**2) #U_0 ** 2 * self.rho_0 / (1.4 * self.mach**2)
#         p = p_0 
#         rho = self.rho_0 * (p / p_0)**(1/1.4)
#         p = jnp.ones_like(u)
#         Primitives = jnp.zeros((len(mesh.area), 4))
#         Primitives = Primitives.at[...,0].set(rho)
#         Primitives = Primitives.at[...,1].set(u)
#         Primitives = Primitives.at[...,2].set(v)
#         Primitives = Primitives.at[...,3].set(p)
#         return Primitives


class CorotatingVortices():
    """
    Two co-rotating Gaussian vortices (Josserand & Rossi 2007) as initial
    conditions for the 2D compressible Euler equations.
 
    Primitives layout matches the template: columns (rho, u, v, p).
 
    Parameters
    ----------
    Gamma : circulation / strength of each vortex (beta).
    r0    : Gaussian core radius.
    D0    : separation between the two vortex centres.
    center: (cx, cy) box centre the pair is placed symmetrically about.
    gamma : ratio of specific heats (for the isentropic thermodynamics).
    rho_inf, p_inf : uniform background state.
    """
 
    def __init__(self, Gamma=1., r0=0.18, D0=1.,
                 center=(0., 0.), gamma=1.4, rho_inf=1., p_inf=1.):
        self.Gamma = Gamma
        self.r0 = r0
        self.D0 = D0
        self.center = center
        self.gamma = gamma
        self.rho_inf = rho_inf
        self.p_inf = p_inf
 
    def build(self, mesh):
        N = len(mesh.area)
        cx, cy = self.center
        # two cores placed symmetrically about the centre, separation D0 along x
        centres = ((cx - self.D0 / 2., cy),
                   (cx + self.D0 / 2., cy))
 
        def one_vortex_velocity(x, y, xc, yc):
            dx = x - xc
            dy = y - yc
            s2 = dx * dx + dy * dy
            s = jnp.sqrt(s2 + 1e-30)                       # regularised radius
            # Lamb-Oseen azimuthal velocity for a Gaussian vorticity core
            u_theta = self.Gamma / (2. * jnp.pi * s) * (1. - jnp.exp(-s2 / self.r0**2))
            # counter-clockwise unit azimuthal direction: (-dy, dx)/s
            return u_theta * (-dy / s), u_theta * (dx / s)
 
        def one_vortex_dT(x, y, xc, yc):
            dx = x - xc
            dy = y - yc
            s2 = dx * dx + dy * dy
            g = self.gamma
            beta = self.Gamma
            # isentropic-vortex temperature dip -> cyclostrophic balance
            return -(g - 1.) * beta**2 / (8. * g * jnp.pi**2) * jnp.exp(1. - s2 / self.r0**2)
 
        def velocity_field(x, y):
            g = self.gamma
            R = self.p_inf / self.rho_inf          # gives T_inf = 1
            T_inf = self.p_inf / (self.rho_inf * R)
 
            u = jnp.zeros_like(x)
            v = jnp.zeros_like(x)
            T = T_inf * jnp.ones_like(x)
            for (xc, yc) in centres:
                du, dv = one_vortex_velocity(x, y, xc, yc)
                u = u + du
                v = v + dv
                T = T + one_vortex_dT(x, y, xc, yc)
 
            rho = self.rho_inf * (T / T_inf) ** (1. / (g - 1.))
            p = (self.p_inf / self.rho_inf**g) * rho**g
            return rho, u, v, p
 
        Primitives = jnp.zeros((N, 4))
        rho, u, v, p = velocity_field(mesh.barycenter[:, 0], mesh.barycenter[:, 1])
        Primitives = Primitives.at[..., 0].set(rho)
        Primitives = Primitives.at[..., 1].set(u)
        Primitives = Primitives.at[..., 2].set(v)
        Primitives = Primitives.at[..., 3].set(p)
        return Primitives

##########################################################

class TestwFunctions():
    def __init__(self, f_rho, f_u, f_v, f_p):
        self.f_rho = f_rho
        self.f_u = f_u
        self.f_v = f_v
        self.f_p = f_p

    def build(self, mesh):
        Primitives = jnp.stack([self.f_rho(mesh.barycenter),
                              self.f_u(mesh.barycenter),
                              self.f_v(mesh.barycenter),
                              self.f_p(mesh.barycenter)
                              ], dim = -1)
        return Primitives

##########################################################

class TestCircleRiemman():
    def __init__(self, inside, upLeft, upRight, downLeft, downRight, center = [0.5, 0.5]):
        self.inside = inside
        self.upLeft = upLeft
        self.upRight = upRight
        self.downLeft = downLeft
        self.downRight = downRight
        self.center = center

    def build(self, mesh):
        N = len(mesh.area)
        Primitives = jnp.zeros((N, 4))
        for i in range(N):
            if jnp.norm(mesh.barycenter[i] - jnp.array(self.center), p = 2.) < 0.125:
                Primitives[i] = self.inside
            else:
                if mesh.barycenter[i,0] > 0.5 and mesh.barycenter[i,1] > 0.5:
                    Primitives[i] = self.upRight
                elif mesh.barycenter[i,0] <= 0.5 and mesh.barycenter[i,1] > 0.5:
                    Primitives[i] = self.upLeft
                elif mesh.barycenter[i,0] <= 0.5 and mesh.barycenter[i,1] <= 0.5:
                    Primitives[i] = self.downLeft
                elif mesh.barycenter[i,0] > 0.5 and mesh.barycenter[i,1] <= 0.5:
                    Primitives[i] = self.downRight
        return Primitives

class TestSquare():
    def __init__(self, inside, outside, center = [0.5, 0.5]):
        self.inside = inside
        self.outside = outside
        self.center = center

    def build(self, mesh):
        N = len(mesh.area)
        Primitives = jnp.zeros((N, 4))
        for i in range(N):
            if jnp.norm(mesh.barycenter[i] - jnp.array(self.center), p = 1.) < 0.2:
                Primitives[i] = self.inside
            else:
                Primitives[i] = self.outside
        return Primitives

##########################################################

class Test():
    def __init__(self, upLeft, upRight, downLeft, downRight):
        self.upLeft = upLeft
        self.upRight = upRight
        self.downLeft = downLeft
        self.downRight = downRight

    def build(self, mesh):
        N = len(mesh.area)
        Primitives = jnp.zeros((N, 4))
        Primitives = jnp.where(jnp.repeat(jnp.logical_and(mesh.barycenter[:,0] > 0.5, mesh.barycenter[:,1] > 0.5)[:,None], 4, axis=1), self.upRight * jnp.ones_like(Primitives), Primitives)
        Primitives = jnp.where(jnp.repeat(jnp.logical_and(mesh.barycenter[:,0] <= 0.5, mesh.barycenter[:,1] > 0.5)[:,None], 4, axis=1), self.upLeft * jnp.ones_like(Primitives), Primitives)
        Primitives = jnp.where(jnp.repeat(jnp.logical_and(mesh.barycenter[:,0] <= 0.5, mesh.barycenter[:,1] <= 0.5)[:,None], 4, axis=1), self.downLeft * jnp.ones_like(Primitives), Primitives)
        Primitives = jnp.where(jnp.repeat(jnp.logical_and(mesh.barycenter[:,0] > 0.5, mesh.barycenter[:,1] <= 0.5)[:,None], 4, axis=1), self.downRight * jnp.ones_like(Primitives), Primitives)
        return Primitives


def Test1():
    upLeft = jnp.array([0.5197, -0.7259, 0, 0.4])
    upRight = jnp.array([1., 0., 0., 1.])
    downLeft = jnp.array([0.1072, -1.4045, -0.7259, 0.0439])
    downRight = jnp.array([0.2579, 0., -1.4045, 0.15])
    return Test(upLeft, upRight, downLeft, downRight)

def Test2():
    upLeft = jnp.array([0.5197, -0.7259, 0, 0.4])
    upRight = jnp.array([1., 0., 0., 1.])
    downLeft = jnp.array([1., -0.7259, -0.7259, 1.])
    downRight = jnp.array([0.5197, 0, -0.7259, 0.4])
    return Test(upLeft, upRight, downLeft, downRight)

def Test3():
    upLeft = jnp.array([0.5323, 1.206, 0, 0.3])
    upRight = jnp.array([1.5, 0., 0., 1.5])
    downLeft = jnp.array([0.138, 1.206, 1.206, 0.029])
    downRight = jnp.array([0.5323, 0, 1.206, 0.3])
    return Test(upLeft, upRight, downLeft, downRight)

def Test4():
    upLeft = jnp.array([0.5065, 0.8939, 0, 0.35])
    upRight = jnp.array([1.1, 0., 0., 1.1])
    downLeft = jnp.array([1.1, 0.8939, 0.8939, 1.1])
    downRight = jnp.array([0.5065, 0, 0.8939, 0.35])
    return Test(upLeft, upRight, downLeft, downRight)

def Test5():
    upLeft = jnp.array([2., -0.75, 0.5, 1.])
    upRight = jnp.array([1., -0.75, -0.5, 1.])
    downLeft = jnp.array([1., 0.75, 0.5, 1])
    downRight = jnp.array([3, 0.75, -0.5, 1.])
    return Test(upLeft, upRight, downLeft, downRight)

def Test6():
    upLeft = jnp.array([2., 0.75, 0.5, 1.])
    upRight = jnp.array([1., 0.75, -0.5, 1.])
    downLeft = jnp.array([1., -0.75, 0.5, 1])
    downRight = jnp.array([3, -0.75, -0.5, 1.])
    return Test(upLeft, upRight, downLeft, downRight)

def Test7():
    upLeft = jnp.array([0.5197, -0.6297, 0.1, 0.4])
    upRight = jnp.array([1, 0.1, 0.1, 1.])
    downLeft = jnp.array([0.8, 0.1, 0.1, 0.4])
    downRight = jnp.array([0.5197, 0.1, -0.6259, 0.4])
    return Test(upLeft, upRight, downLeft, downRight)

def Test8():
    upLeft = jnp.array([1., -0.6259, 0.1, 1])
    upRight = jnp.array([0.5197, 0.1, 0.1, 0.4])
    downLeft = jnp.array([0.8, 0.1, 0.1, 1])
    downRight = jnp.array([1., 0.1, -0.6259, 1])
    return Test(upLeft, upRight, downLeft, downRight)

def Test9():
    upLeft = jnp.array([2, 0, -0.3, 1])
    upRight = jnp.array([1, 0, 0.3, 1])
    downLeft = jnp.array([1.039, 0, -0.8133, 0.4])
    downRight = jnp.array([0.5197, 0, 0.4297, 0.4])
    return Test(upLeft, upRight, downLeft, downRight)

def Test10():
    upLeft = jnp.array([0.5, 0, 0.6076, 1.])
    upRight = jnp.array([1., 0, 0.4297, 1])
    downLeft = jnp.array([0.2281, 0, -0.6076, 0.333])
    downRight = jnp.array([0.4562, 0, -0.4297, 0.333])
    return Test(upLeft, upRight, downLeft, downRight)

def Test11():
    upLeft = jnp.array([0.5313, 0.8276, 0, 0.4])
    upRight = jnp.array([1, 0.1, 0, 1])
    downLeft = jnp.array([0.8, 0.1, 0, 0.4])
    downRight = jnp.array([0.5313, 0.1, 0.7276, 0.4])
    return Test(upLeft, upRight, downLeft, downRight)

def Test12():
    upLeft = jnp.array([1, 0.7276, 0., 1])
    upRight = jnp.array([0.5313, 0, 0., 0.4])
    downLeft = jnp.array([0.8, 0, 0., 1.])
    downRight = jnp.array([1., 0., 0.7276, 1.])
    return Test(upLeft, upRight, downLeft, downRight)

def Test13():
    upLeft = jnp.array([2., 0, 0.3, 1.])
    upRight = jnp.array([1, 0., -0.3, 1.])
    downLeft = jnp.array([1.0625, 0, 0.8145, 0.4])
    downRight = jnp.array([0.5313, 0, 0.4276, 0.4])
    return Test(upLeft, upRight, downLeft, downRight)

def Test14():
    upLeft = jnp.array([1., 0., -1.2172, 8.])
    upRight = jnp.array([2., 0., -0.5606, 8.])
    downLeft = jnp.array([0.4736, 0., 1.2172, 2.6667])
    downRight = jnp.array([0.9474, 0., 1.1606, 2.6667])
    return Test(upLeft, upRight, downLeft, downRight)

def Test15():
    upLeft = jnp.array([0.5197, -0.6259, -0.3, 0.4])
    upRight = jnp.array([1., 0.1, -0.3, 1.])
    downLeft = jnp.array([0.8, 0.1, -0.3, 0.4])
    downRight = jnp.array([0.5313, 0.1, 0.4276, 0.4])
    return Test(upLeft, upRight, downLeft, downRight)

def Test16():
    upLeft = jnp.array([1.0222, -0.6179, 0.1, 1.])
    upRight = jnp.array([0.5313, 0.1, 0.1, 0.4])
    downLeft = jnp.array([0.8, 0.1, 0.1, 1.])
    downRight = jnp.array([1., 0.1, 0.8276, 1.])
    return Test(upLeft, upRight, downLeft, downRight)

def Test17():
    upLeft = jnp.array([2., 0., -0.3, 1.])
    upRight = jnp.array([1., 0., -0.4, 1.])
    downLeft = jnp.array([1.0625, 0., 0.2145, 0.4])
    downRight = jnp.array([0.5197, 0., -1.1259, 0.4])
    return Test(upLeft, upRight, downLeft, downRight)

def Test18():
    upLeft = jnp.array([2., 0., -0.3, 1.])
    upRight = jnp.array([1., 0., 1., 1.])
    downLeft = jnp.array([1.0625, 0., 0.2145, 0.4])
    downRight = jnp.array([0.5197, 0., 0.2741, 0.4])
    return Test(upLeft, upRight, downLeft, downRight)

def Test19():
    upLeft = jnp.array([2., 0., -0.3, 1.])
    upRight = jnp.array([1., 0., 0.3, 1.])
    downLeft = jnp.array([1.0625, 0., 0.2145, 0.4])
    downRight = jnp.array([0.5197, 0., -0.4259, 0.4])
    return Test(upLeft, upRight, downLeft, downRight)



##########################################################################
#                      incompressible NS                                #    
##########################################################################


# class TaylorGreenVortex():
#     def build(self, mesh):
#         N = len(mesh.area)
#         L = jnp.max(mesh.points[...,0]) - jnp.min(mesh.points[...,0])

#         def velocity_field(x, y):
#             u =  - jnp.sin(2 * jnp.pi * x / L) * jnp.cos(2 * jnp.pi * y / L)
#             v =    jnp.cos(2 * jnp.pi * x / L) * jnp.sin(2 * jnp.pi * y / L)
#             return u, v


#         Primitives = jnp.zeros((N, 2))
#         u, v = velocity_field(mesh.barycenter[:,0], mesh.barycenter[:,1])
#         Primitives = Primitives.at[...,0].set(u)
#         Primitives = Primitives.at[...,1].set(v)
#         return Primitives
    
class advected_sinus():
    def build(self, mesh, u = 1., v = 0., p = 1):
        N = len(mesh.area)
        L = jnp.max(mesh.points[...,0]) - jnp.min(mesh.points[...,0])
        
        def scalar_field(x, y):
            return 1.5 - jnp.sin(2 * jnp.pi * x / L) * jnp.cos(2 * jnp.pi * y / L)

        Primitives = jnp.zeros((N, 4))
        s = scalar_field(mesh.barycenter[:,0], mesh.barycenter[:,1])
        Primitives = Primitives.at[...,0].set(s)
        Primitives = Primitives.at[...,1].set(u)
        Primitives = Primitives.at[...,2].set(v)
        Primitives = Primitives.at[...,3].set(p)
        return Primitives
        