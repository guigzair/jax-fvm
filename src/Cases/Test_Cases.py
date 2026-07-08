from jax import P
import jax.numpy as jnp
import numpy as np
import meshpy.triangle as triangle
from dataclasses import dataclass

from jax_fvm.src.solvers.helper import getConserved

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

class MergingPair():
    def __init__(self, r0=0.18, D0=1.,
                 center=(0., 0.), rho_inf=1., omega=300, mach = 0.01):
        self.r0 = r0
        self.D0 = D0
        self.center = center
        self.rho_inf = rho_inf
        self.omega = omega
        self.mach = mach

    def build(self, mesh):
        N = len(mesh.area)
        cx, cy = self.center
        # two cores placed symmetrically about the centre, separation D0 along x
        centres = jnp.array([[cx - self.D0 / 2., cy],
                              [cx + self.D0 / 2., cy]])
        
        x = mesh.barycenter[:,0]
        y = mesh.barycenter[:,1]
        r1 = jnp.linalg.norm(jnp.stack([x - centres[0,0], y - centres[0,1]], axis=-1), axis=-1)
        r2 = jnp.linalg.norm(jnp.stack([x - centres[1,0], y - centres[1,1]], axis=-1), axis=-1)

        def velocity_field(x, y):
            u =  self.omega / 2  * ( - (y-centres[1,1])  * jnp.exp(-(r1/self.r0)**2) - (y-centres[0,1]) * jnp.exp(-(r2/self.r0)**2))
            v = self.omega / 2  * ( (x-centres[1,0])  * jnp.exp(-(r1/self.r0)**2) + (x-centres[0,0]) * jnp.exp(-(r2/self.r0)**2))
            return u, v
        
        def pressure_field():
            p = (self.omega * self.r0 / 4)**2 * (jnp.exp(-2*(r1/self.r0)**2) + jnp.exp(-2*(r2/self.r0)**2))
            return p
        
        u, v = velocity_field(mesh.barycenter[:,0], mesh.barycenter[:,1])
        p_0 = 1 / (1.4 * self.mach**2) #U_0 ** 2 * self.rho_0 / (1.4 * self.mach**2)
        p = p_0 - pressure_field()
        rho = self.rho_inf * (p / p_0)**(1/1.4)
        
        Primitives = jnp.zeros((N, 4))
        Primitives = Primitives.at[...,0].set(rho)
        Primitives = Primitives.at[...,1].set(u)
        Primitives = Primitives.at[...,2].set(v )
        Primitives = Primitives.at[...,3].set(p )
        return Primitives

class CorotatingVortices():
    """
    Two co-rotating Gaussian (Lamb-Oseen) vortices, low-Mach isentropic
    initialisation for the 2D compressible Euler equations.
    Primitives columns: (rho, u, v, p).

    Mach : target background Mach number based on the peak swirl speed.
           Lower Mach -> smaller density perturbation, closer to incompressible.
    """

    def __init__(self, Gamma=1., r0=0.18, D0=1., center=(0., 0.),
                 gamma=1.4, rho_inf=1., Mach=0.1, window=True):
        self.Gamma = Gamma
        self.r0 = r0
        self.D0 = D0
        self.center = center
        self.gamma = gamma
        self.rho_inf = rho_inf
        self.Mach = Mach
        self.window = window

    def _u_peak(self):
        # max over r of (Gamma/(2 pi r))(1 - exp(-r^2/r0^2))
        # occurs at r ~ 1.121 r0, value ~ 0.71533 * Gamma/(2 pi r0)
        return 0.71533 * self.Gamma / (2. * jnp.pi * self.r0)

    def _p_inf(self):
        # c_inf = u_peak / Mach ;  p_inf = rho_inf c_inf^2 / gamma
        c_inf = self._u_peak() / self.Mach
        return self.rho_inf * c_inf**2 / self.gamma
    
    def _window(self, s2):
        R_cut = 2.5 * self.D0
        delta = 4 * self.r0
        # smooth C-infinity cutoff: 1 inside R_cut-delta, 0 outside R_cut
        r = jnp.sqrt(s2 + 1e-30)
        t = (R_cut - r) / delta
        t = jnp.clip(t, 0., 1.)
        # smoothstep (C1); use the C-inf bump below if you need more smoothness
        return t * t * (3. - 2. * t)

    def build(self, mesh):
        N = len(mesh.area)
        cx, cy = self.center
        centres = ((cx - self.D0 / 2., cy),
                   (cx + self.D0 / 2., cy))
        g = self.gamma
        p_inf = self._p_inf()
        c_inf2 = g * p_inf / self.rho_inf          # = c_inf^2


        def one_vortex_velocity(x, y, xc, yc):
            dx = x - xc
            dy = y - yc
            s2 = dx * dx + dy * dy
            s = jnp.sqrt(s2 + 1e-30)
            u_theta = self.Gamma / (2. * jnp.pi * s) * (1. - jnp.exp(-s2 / (1. * self.r0**2)))
            # windowing to avoid BC interactions
            if self.window:
                w = self._window(s2)
                u_theta = u_theta * w
            return u_theta * (-dy / s), u_theta * (dx / s)

        def one_vortex_dTemp(x, y, xc, yc):
            # Isentropic temperature dip whose amplitude scales with the
            # actual swirl energy, so it shrinks like M^2 at low Mach.
            # Uses the same Gaussian shape as the classic isentropic vortex,
            # calibrated so the core depth is consistent with u_peak.
            dx = x - xc
            dy = y - yc
            s2 = dx * dx + dy * dy
            # peak dynamic-pressure scale of one core:
            amp = (g - 1.) / (2. * g) * self._u_peak()**2 / c_inf2  # ~ (g-1)/2 * M^2
            return -amp * jnp.exp(1. - s2 / (1. * self.r0**2))   # in units of T_inf

        def velocity_field(x, y):
            T_inf = 1.0                      # reference; only ratios matter
            u = jnp.zeros_like(x)
            v = jnp.zeros_like(x)
            T = T_inf * jnp.ones_like(x)
            for (xc, yc) in centres:
                du, dv = one_vortex_velocity(x, y, xc, yc)
                u = u + du
                v = v + dv
                T = T + one_vortex_dTemp(x, y, xc, yc) * T_inf

            rho = self.rho_inf * (T / T_inf) ** (1. / (g - 1.))
            p = (p_inf / self.rho_inf**g) * rho**g       # p = K rho^gamma
            return rho, u, v, p

        Primitives = jnp.zeros((N, 4))
        rho, u, v, p = velocity_field(mesh.barycenter[:, 0], mesh.barycenter[:, 1])
        Primitives = Primitives.at[..., 0].set(rho)
        Primitives = Primitives.at[..., 1].set(u)
        Primitives = Primitives.at[..., 2].set(v)
        Primitives = Primitives.at[..., 3].set(p)

        # Rebuild (rho, p) in true mechanical equilibrium grad(p) = -rho (u.grad)u
        # via a Poisson solve, so the low-Mach IC does not radiate acoustic waves.
        import jax_fvm.src.solvers.helper as helper
        Primitives = helper.equilibrium_pressure_poisson(
            Primitives, mesh, gamma=self.gamma, rho_inf=self.rho_inf)
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
        

if __name__ == "__main__":
    import sys
    sys.path.append('../../..')  
    from jax_fvm.src.mesh.mesh import Mesh
    from jax_fvm.src.solvers import helper

    mesh = Mesh()
    mesh.mesh_generator(maxV = 1e-2, bounds = [-10., 10., -10., 10.])

    prim = CorotatingVortices(Gamma=1., r0=0.18, D0=1., center=(0., 0.), gamma=1.4, rho_inf=1., Mach=0.1).build(mesh)

    mesh.plot_solution(prim[...,0])
    vorticity = helper.get_vorticity_from_field(helper.getConserved(prim), mesh)

    mesh.plot_solution(vorticity)