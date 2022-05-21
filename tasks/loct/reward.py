import math
from abc import ABC

import numpy as np

from qdpgym.utils import tf

ATANH0_95 = math.atanh(0.95)
ATANH0_9 = math.atanh(0.9)
SQRT_LOG10 = math.sqrt(math.log(10))
SQRT_LOG20 = math.sqrt(math.log(20))


class Reward(ABC):
    def __call__(self, cmd, env, robot) -> float:
        raise NotImplementedError


def tanh2_reshape(lower, upper):
    w = np.sqrt(ATANH0_95) / (upper - lower)

    def _reshape(v):
        return np.tanh(np.power((v - lower) * w, 2)) if v > lower else 0

    return _reshape


def tanh2_reverse(lower, upper, value):
    w = np.sqrt(ATANH0_95) / (upper - lower)
    return np.sqrt(np.arctanh(value)) / w + lower


def tanh_reshape(lower, upper):
    """
    Zoom tanh so that f(lower) = -0.9 and f(upper) = 0.9
    :return: function f
    """
    middle = (lower + upper) / 2
    w = ATANH0_9 / (upper - middle)

    def _reshape(v):
        return np.tanh((v - middle) * w)

    return _reshape


def tanh_reverse(lower, upper, value):
    middle = (lower + upper) / 2
    w = ATANH0_9 / (upper - middle)
    return np.arctanh(value) / w + middle


def exp_m2_reshape(upper):
    w = upper / SQRT_LOG10

    def _reshape(v):
        return np.exp(-(v / w) ** 2)

    return _reshape


def quadratic_linear_reshape(upper):
    k = 1 / upper ** 2
    kl = 2 * k * upper

    def _reshape(v):
        v = abs(v)
        return v ** 2 * k if v < upper else kl * (v - upper) + 1

    return _reshape


def soft_constrain(thr, upper):
    def _reshape(v):
        excess = abs(v) - thr
        if excess > 0:
            return -(excess / upper) ** 2
        return 0.

    return _reshape


class LinearVelocityReward(Reward):
    def __init__(self, forward=0.8, lateral=0.4):
        self.forward, self.lateral = forward, lateral
        self.forward_lateral = forward * lateral

    def __call__(self, cmd, env, robot):
        linear = robot.getBaseLinearVelocityInBaseFrame()
        projected_velocity = np.dot(cmd[:2], linear[:2])
        if (cmd[:2] == 0.).all():
            return 1.
        return self.reshape(self.get_desired_velocity(cmd), projected_velocity)

    def get_desired_velocity(self, cmd):
        return self.forward_lateral / math.hypot(self.lateral * cmd[0], self.forward * cmd[1])

    @staticmethod
    def reshape(scale, value):
        return math.tanh(value * ATANH0_9 / scale)


class UnifiedLinearReward(LinearVelocityReward):
    def __init__(self, forward=1.5, lateral=1.2, ortho_upper=0.3, ortho_weight=0.33):
        super().__init__(forward, lateral)
        self.ortho_reshape = tanh2_reshape(0, ortho_upper)
        self.ortho_weight = ortho_weight

    def __call__(self, cmd, env, robot):
        lin_vel = robot.getBaseLinearVelocityInBaseFrame()[:2]
        proj_vel = np.dot(cmd[:2], lin_vel)
        ortho_vel = math.hypot(*(lin_vel - cmd[:2] * proj_vel))
        print(proj_vel, ortho_vel)
        ortho_pen = 1 - self.ortho_reshape(ortho_vel)
        if (cmd[:2] == 0.).all():
            return ortho_pen
        proj_rew = self.reshape(self.get_desired_velocity(cmd), proj_vel)
        return ((1 - self.ortho_weight) * proj_rew + self.ortho_weight * ortho_pen) * 1.5


class YawRateReward(Reward):
    def __init__(self, upper_pos=1.0, upper_neg=0.45):
        self.reshape_pos = tanh_reshape(-upper_pos, upper_pos)
        self.reshape_neg = quadratic_linear_reshape(upper_neg)

    def __call__(self, cmd, env, robot):
        yaw_cmd, yaw_rate = cmd[2], robot.getBaseRpyRate()[2]
        if yaw_cmd != 0.0:
            return self.reshape_pos(yaw_rate / yaw_cmd)
        else:
            return 1 - self.reshape_neg(abs(yaw_rate))


class RollPitchRatePenalty(Reward):
    def __init__(self, dr_upper=1.0, dp_upper=0.5):
        self.dr_reshape = tanh2_reshape(0.0, dr_upper)
        self.dp_reshape = tanh2_reshape(0.0, dp_upper)

    def __call__(self, cmd, env, robot):
        r_rate, p_rate, _ = robot.getBaseRpyRate()
        return 1 - (self.dr_reshape(abs(r_rate)) + self.dp_reshape(abs(p_rate))) / 2


class OrthogonalLinearPenalty(Reward):
    def __init__(self, linear_upper=0.3):
        self.reshape = tanh2_reshape(0, linear_upper)

    def __call__(self, cmd, env, robot):
        linear = robot.getBaseLinearVelocityInBaseFrame()[:2]
        v_o = np.asarray(linear) - np.asarray(cmd[:2]) * np.dot(linear, cmd[:2])
        return 1 - self.reshape(tf.vnorm(v_o))


class VerticalLinearPenalty(Reward):
    def __init__(self, upper=0.4):
        self.reshape = quadratic_linear_reshape(upper)

    def __call__(self, cmd, env, robot):
        return 1 - self.reshape(env.getTerrainBasedVerticalVelocityOfRobot())
        # return 1 - self.reshape(robot.getBaseLinearVelocityInBaseFrame()[2])


class BodyPosturePenalty(Reward):
    def __init__(self, roll_upper=np.pi / 12, pitch_upper=np.pi / 24):
        self.roll_reshape = quadratic_linear_reshape(roll_upper)
        self.pitch_reshape = quadratic_linear_reshape(pitch_upper)

    def __call__(self, cmd, env, robot):
        r, p, _ = env.getTerrainBasedRpyOfRobot()
        return 1 - (self.roll_reshape(r) + self.pitch_reshape(p)) / 2


class BodyHeightReward(Reward):
    def __init__(self, des=0.4, range_=0.03):
        self.des = des
        self.reshape = quadratic_linear_reshape(range_)

    def __call__(self, cmd, env, robot):
        return env.getTerrainBasedHeightOfRobot()


class ActionSmoothnessReward(Reward):
    def __init__(self, upper=400):
        self.reshape = exp_m2_reshape(upper)

    def __call__(self, cmd, env, robot):
        return self.reshape(env.getActionViolence()).sum() / 12


class JointMotionPenalty(Reward):
    def __init__(self, vel_upper=6, acc_upper=500):
        self.vel_reshape = np.vectorize(quadratic_linear_reshape(vel_upper))
        self.acc_reshape = np.vectorize(quadratic_linear_reshape(acc_upper))

    def __call__(self, cmd, env, robot):
        pen = self.vel_reshape(robot.getJointVelocities()) + self.acc_reshape(robot.getJointAccelerations())
        return 1 - pen.sum() / 12


class TorqueGradientPenalty(Reward):
    def __init__(self, upper=200):
        self.reshape = quadratic_linear_reshape(upper)

    def __call__(self, cmd, env, robot):
        torque_grad = robot.getTorqueGradients()
        return -sum(self.reshape(grad) for grad in torque_grad) / 12


class FootSlipPenalty(Reward):
    def __init__(self, upper=0.5):
        self.reshape = np.vectorize(quadratic_linear_reshape(upper))

    def __call__(self, cmd, env, robot):
        return 1 - sum(self.reshape(robot.getFootSlipVelocity()))


class HipAnglePenalty(Reward):
    def __init__(self, upper=0.3):
        self.reshape = np.vectorize(quadratic_linear_reshape(upper))

    def __call__(self, cmd, env, robot):
        hip_angles = robot.getJointPositions()[(0, 3, 6, 9),]
        return -sum(self.reshape(hip_angles)) / 4


class JointConstraintPenalty(Reward):
    def __init__(self, constraints=(0., 0., 0.), upper=(0.4, 0.6, 0.6)):
        self.hip_reshape = np.vectorize(soft_constrain(constraints[0], upper[0]))
        self.thigh_reshape = np.vectorize(soft_constrain(constraints[1], upper[1]))
        self.shank_reshape = np.vectorize(soft_constrain(constraints[2], upper[2]))

    def __call__(self, cmd, env, robot):
        joint_angles = robot.getJointPositions() - robot.STANCE_POSTURE
        return (self.hip_reshape(joint_angles[((0, 3, 6, 9),)]).sum() +
                self.thigh_reshape(joint_angles[((1, 4, 7, 10),)]).sum() +
                self.shank_reshape(joint_angles[((2, 5, 8, 11),)]).sum()) / 12


class TrivialStridePenalty(Reward):
    def __init__(self, lower=-0.2, upper=0.4):
        self.reshape = tanh_reshape(lower, upper)

    def __call__(self, cmd, env, robot):
        strides = [np.dot(s, cmd[:2]) for s in robot.getStrides()]
        return sum(self.reshape(s) for s in strides if s != 0.0)
        # return 1 - sum(1 - self.reshape(s) for s in strides if s != 0.0)


class FootClearanceReward(Reward):
    def __init__(self, upper=0.08):
        self.reshape = tanh2_reshape(0., upper)

    def __call__(self, cmd, env, robot):
        foot_clearances = robot.getFootClearances()
        return sum(self.reshape(c) for c in foot_clearances)


class AliveReward(Reward):
    def __call__(self, cmd, env, robot):
        return 1.0


class ClearanceOverTerrainReward(Reward):
    def __call__(self, cmd, env, robot):
        reward = 0.
        for leg in range(4):
            x, y, z = robot.getFootPositionInWorldFrame(leg)
            if z - robot.FOOT_RADIUS > max(env.getTerrainScan(x, y, robot.rpy.y)) + 0.01:
                reward += 1 / 2
        return reward


class BodyCollisionPenalty(Reward):
    def __call__(self, cmd, env, robot):
        contact_states = list(robot.getContactStates())
        for i in range(1, 5):
            contact_states[i * 3] = 0
        return 1 - sum(contact_states)


class TorquePenalty(Reward):
    def __init__(self, upper=900):
        self.reshape = np.vectorize(quadratic_linear_reshape(upper))

    def __call__(self, cmd, env, robot):
        return 1 - sum(self.reshape(robot.getLastAppliedTorques() ** 2))