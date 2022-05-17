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


# if qdpgym.sim_engine == 'mujoco':
#     assert is_mujoco_available(), 'dm_control is not installed'
# else:
#     assert is_bullet_available(), 'pybullet is not installed'
