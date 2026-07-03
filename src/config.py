import sys
sys.path.append('../..')
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import yaml
from jax_fvm.src.mesh.mesh import Mesh
import jax_fvm.src.Cases.Test_Cases as Test_Cases
import jax_fvm.src.mesh.Mesh_cases as Mesh_Cases
import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

SRC_DIR = Path(__file__).resolve().parent

# ---- registry: string name -> IC class ----
IC_REGISTRY = {"corotating_merge": Test_Cases.CorotatingVortices,
               "dipole_vortex": Test_Cases.TestDipoleVortex2,
               "kelvin_helmholtz": Test_Cases.KevinHelmotzInstability}

MESH_REGISTRY = {"Forward_Step": Mesh_Cases.Forward_Step,
               "dipole_vortex": Mesh_Cases.TestDipoleVortex,
               "UniformMesh": Mesh_Cases.UniformMesh}

# PHYSICS_REGISTRY = {"Euler": NS.residual,
#                     "NS": NS.residual}

def register_ic(name):
    def deco(cls):
        IC_REGISTRY[name] = cls
        return cls
    return deco


# ---- typed config containers (mirror the YAML) ----
@dataclass
class MeshConfig:
    type: str = "None"
    info: Any = None
    maxV: float = 1e-3
    marker_boundary: int = 1
    bounds: list[float] = field(default_factory=list)
    min_angle: float = 30.0
    params: dict = field(default_factory=dict)

@dataclass
class ICConfig:
    type: str
    params: dict = field(default_factory=dict)

@dataclass
class TimeConfig:
    cfl: float = 0.4
    t_end: float = 1.0
    scheme: str = "rk3"

# ---- registry: string name -> solver builder ----
def _build_euler(physics, time, mesh):
    from jax_fvm.src.solvers.Euler.Euler import EulerSolver   # match your real path
    return EulerSolver(mesh=mesh, gamma=physics.gamma, **time.integrator_kwargs())

def _build_ns(physics, time, mesh):
    from jax_fvm.src.solvers.compressible_NS.solver import NSSolver
    return NSSolver(mesh=mesh, gamma=physics.gamma,
                    mu=physics.mu, Pr=physics.Pr, **time.integrator_kwargs())



@dataclass
class Config:
    name: str
    mesh: MeshConfig
    initial_condition: ICConfig
    time: TimeConfig
    kwargs: dict = field(default_factory=dict)

    @classmethod
    def from_yaml(cls, path):
        with open(path) as f:
            raw = yaml.safe_load(f)
        return cls(
            name=raw["name"],
            mesh=MeshConfig(**raw["mesh"]),
            initial_condition=ICConfig(**raw["initial_condition"]),
            time=TimeConfig(**raw.get("time", {})),
            kwargs=raw.get("kwargs", {}),
        )
    
    def write_yaml(self, path):
        with open(path, "w") as f:
            yaml.dump(self.__dict__, f)
    
@dataclass
class Case:
    config: Config
    mesh: Any = None
    primitives: Any = None  # (N, 4) array from the IC's build()

    @classmethod
    def build(cls, path, MeshClass):
        cfg = Config.from_yaml(path)

        if cfg.mesh.type == "None":
            mesh = MeshClass()
            mesh.mesh_generator(maxV = cfg.mesh.maxV, marker_boundary = cfg.mesh.marker_boundary, bounds = cfg.mesh.bounds)
        else:
            mesh = MESH_REGISTRY.get(cfg.mesh.type)(**cfg.mesh.params).build()

        ic = IC_REGISTRY.get(cfg.initial_condition.type)
        if not ic:
            raise ValueError(f"Unknown initial condition type: {cfg.initial_condition.type}")
        ic = ic(**cfg.initial_condition.params)
        primitives = ic.build(mesh)

        return cls(config=cfg, mesh=mesh, primitives=primitives)
    
if __name__ == "__main__":
    # cfg_path = SRC_DIR / "Cases" / "config" / "corotating_merge.yaml"
    # cfg_path = "../../Cases/config/KelvinHelmholtz.yaml"
    cfg_path = "Cases/config/DipoleVortex.yaml"

    case = Case.build(cfg_path, MeshClass=Mesh)