import os

import qdpgym

pkg_path = os.path.dirname(os.path.abspath(__file__))
rsc_dir = os.path.join(pkg_path, 'resources')


def is_bullet_available():
    try:
        import pybullet
    except ModuleNotFoundError:
        return False
    return True


def is_mujoco_available():
    try:
        import mujoco
        import dm_control
    except ModuleNotFoundError:
        return False
    return True


if qdpgym.sim_engine == qdpgym.Sim.MUJOCO:
    assert is_mujoco_available(), 'dm_control is not installed'
    raise NotImplementedError
else:
    assert is_bullet_available(), 'pybullet is not installed'
    from .blt.quadruped import AliengoBt as Aliengo
    from .blt.env import QuadrupedEnvBt as QuadrupedEnv
    from .blt.terrain import TerrainBt as NullTerrain
    from .blt.terrain import HillsBt as Hills
    import qdpgym.sim.blt.hooks as hooks
    import qdpgym.sim.blt.terrain as terrain

    from .blt.hooks import ViewerBtHook as ViewerHook
    from .blt.hooks import ExtraViewerBtHook as ExtraViewerHook
    from .blt.hooks import RandomPerturbBtHook as RandomPerturbHook
    from .blt.hooks import RandomTerrainBtHook as RandomTerrainHook
