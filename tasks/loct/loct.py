import math
from typing import Optional

import numpy as np

from qdpgym.sim.abc import Quadruped, Environment, QuadrupedHandle
from qdpgym.sim.common.tg import TgStateMachine, vertical_tg
from qdpgym.sim.task import BasicTask


class LocomotionV0(BasicTask):
    def __init__(self, reward_coeff=1.0, substep_reward_on=True):
        super().__init__(reward_coeff, substep_reward_on)
        self._cmd = np.array((0., 0., 0.))
        self._traj_gen: Optional[TgStateMachine] = None

        self._weights: Optional[np.ndarray] = None
        self._bias: Optional[np.ndarray] = None

    @property
    def cmd(self):
        return self._cmd

    @cmd.setter
    def cmd(self, value):
        self._cmd = np.array(value)

    def register_env(self, robot, env, random_state):
        super().register_env(robot, env, random_state)
        self._traj_gen = TgStateMachine(env.timestep * env.num_substeps,
                                        random_state, vertical_tg(0.12))
        self._build_weights_and_bias()

    def process_action(self, action):
        return action * np.array((0.2, 0.2, 0.1) * 4)

    def before_step(self, action):
        action = super().before_step(action)
        action = self.process_action(action)
        self._traj_gen.update()
        priori = self._traj_gen.get_priori_trajectory().reshape(4, 3)
        des_pos = action.reshape(4, 3) + priori
        des_joint_pos = []
        for i, pos in enumerate(des_pos):
            des_joint_pos.append(self._robot.inverse_kinematics(i, pos))
        return np.concatenate(des_joint_pos)

    def get_observation(self):
        r: Quadruped = self._robot
        e: Environment = self._env
        n: QuadrupedHandle = r.noisy
        if (self._cmd[:2] == 0.).all():
            cmd_obs = np.concatenate(((0.,), self._cmd))
        else:
            cmd_obs = np.concatenate(((1.,), self._cmd))

        roll_pitch = n.get_base_rpy()[:2]
        base_linear = n.get_velocimeter()
        base_angular = n.get_gyro()
        joint_pos = n.get_joint_pos()
        joint_vel = n.get_joint_vel()

        action_history = e.action_history
        joint_target = action_history[-1]
        joint_target_history = action_history[-2]

        tg_base_freq = (self._traj_gen.base_frequency,)
        tg_freq = self._traj_gen.frequency
        tg_raw_phases = self._traj_gen.phases
        tg_phases = np.concatenate((np.sin(tg_raw_phases), np.cos(tg_raw_phases)))

        joint_err = n.get_last_command() - n.get_joint_pos()
        state1, state2 = n.get_state_history(0.01), n.get_state_history(0.02)
        cmd1, cmd2 = n.get_cmd_history(0.01).command, n.get_cmd_history(0.02).command
        joint_proc_err = np.concatenate((cmd1 - state1.joint_pos, cmd2 - state2.joint_pos))
        joint_proc_vel = np.concatenate((state1.joint_vel, state2.joint_vel))

        terrain_info = self._collect_terrain_info()
        contact_states = r.get_leg_contacts()
        contact_forces = r.get_force_sensor().reshape(-1)
        foot_friction_coeffs = np.ones(4)

        perturbation = e.get_perturbation(in_robot_frame=True)
        if perturbation is None:
            perturbation = np.zeros(6)

        return (np.concatenate((
            terrain_info,
            contact_states,
            contact_forces,
            foot_friction_coeffs,
            perturbation,
            cmd_obs,
            roll_pitch,
            base_linear,
            base_angular,
            joint_pos,
            joint_vel,
            joint_target,
            tg_phases,
            tg_freq,
            joint_err,
            joint_proc_err,
            joint_proc_vel,
            joint_target_history,
            tg_base_freq
        )) - self._bias) * self._weights

    def _collect_terrain_info(self):
        yaw = self._robot.get_base_rpy()[2]
        dx, dy = 0.1 * math.cos(yaw), 0.1 * math.sin(yaw)
        points = ((dx - dy, dx + dy), (dx, dy), (dx + dy, -dx + dy),
                  (-dy, dx), (0, 0), (dy, -dx),
                  (-dx - dy, dx - dy), (-dx, -dy), (-dx + dy, -dx - dy))
        samples = []
        for x, y, z in self._robot.get_foot_pos():
            for px, py in points:
                samples.append(z - self._env.arena.get_height(x + px, y + py))
        return np.concatenate((samples, np.zeros(8)))

    def _build_weights_and_bias(self):
        self._weights = np.concatenate((
            (5.,) * 36,  # terrain scan
            (2.5,) * 8,  # terrain slope
            (2.,) * 12,  # contact
            (0.01, 0.01, 0.02) * 4,  # contact force
            (1.,) * 4,  # friction,
            (0.1, 0.1, 0.1, 0.4, 0.2, 0.2),  # perturbation
            (1.,) * 4,  # command
            (2., 2.),  # roll pitch
            (2.,) * 3,  # linear
            (2.,) * 3,  # angular
            (2.,) * 12,  # joint pos
            (0.5, 0.4, 0.3) * 4,  # joint vel
            (2.,) * 12,  # joint target
            (1.,) * 8,  # tg phases
            (100.,) * 4,  # tg freq
            (6.5, 4.5, 3.5) * 4,  # joint error
            (5.,) * 24,  # proc joint error
            (0.5, 0.4, 0.3) * 8,  # proc joint vel
            (2.,) * 12,  # joint target history
            (1,)  # tg base freq
        ))
        stance_cfg = self._robot.STANCE_CONFIG
        self._bias = np.concatenate((
            (0.,) * 36,  # terrain scan
            (0,) * 8,  # terrain slope
            (0.5,) * 12,  # contact
            (0., 0., 30.) * 4,  # contact force
            (0.,) * 4,  # friction,
            (0.,) * 6,  # perturbation
            (0.,) * 4,  # command
            (0., 0.),  # roll pitch
            (0.,) * 3,  # linear
            (0.,) * 3,  # angular
            stance_cfg,  # joint pos
            (0.,) * 12,  # joint vel
            stance_cfg,  # joint target
            (0.,) * 8,  # tg phases
            (self._traj_gen.base_frequency,) * 4,  # tg freq
            (0.,) * 12,  # joint error
            (0.,) * 24,  # joint proc error
            (0.,) * 24,  # joint proc vel
            stance_cfg,  # joint target history
            (self._traj_gen.base_frequency,)  # tg base freq
        ))
