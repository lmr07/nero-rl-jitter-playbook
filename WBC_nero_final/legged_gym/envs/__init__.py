from legged_gym import LEGGED_GYM_ROOT_DIR, LEGGED_GYM_ENVS_DIR


from legged_gym.envs.b2.b2z1_pos_force_config import B2Z1PosForceRoughCfg, B2Z1PosForceRoughCfgPPO
from legged_gym.envs.a2.a2nero_pos_force_config import A2NEROPosForceRoughCfg, A2NEROPosForceRoughCfgPPO

from .base.legged_robot import LeggedRobot
from .b2.legged_robot_b2z1_pos_force import LeggedRobot_b2z1_pos_force
from .a2.legged_robot_a2nero_pos_force import LeggedRobot_a2nero_pos_force

from legged_gym.utils.task_registry import task_registry


task_registry.register( "b2z1_pos_force", LeggedRobot_b2z1_pos_force, B2Z1PosForceRoughCfg(), B2Z1PosForceRoughCfgPPO())
task_registry.register( "a2nero_pos_force", LeggedRobot_a2nero_pos_force, A2NEROPosForceRoughCfg(), A2NEROPosForceRoughCfgPPO())
from .nero.nero_fixed import NeroFixed
from .nero.nero_fixed_config import NeroFixedCfg, NeroFixedCfgPPO
task_registry.register('nero_fixed', NeroFixed, NeroFixedCfg(), NeroFixedCfgPPO())

from .nero.nero_joint import NeroJoint
from .nero.nero_joint_config import NeroJointCfg, NeroJointCfgPPO
task_registry.register('nero_joint', NeroJoint, NeroJointCfg(), NeroJointCfgPPO())

from .nero.nero_seven_cycle import NeroSevenCycle
from .nero.nero_seven_cycle_config import NeroSevenCycleCfg, NeroSevenCycleCfgPPO
task_registry.register('nero_seven_cycle', NeroSevenCycle, NeroSevenCycleCfg(), NeroSevenCycleCfgPPO())
from .nero.nero_seven_cycle_smooth_config import NeroSevenCycleSmoothCfg, NeroSevenCycleSmoothCfgPPO
task_registry.register('nero_seven_cycle_smooth', NeroSevenCycle, NeroSevenCycleSmoothCfg(), NeroSevenCycleSmoothCfgPPO())

from .nero.nero_ee_tracking import NeroEETracking
from .nero.nero_ee_tracking_config import NeroEETrackingCfg, NeroEETrackingCfgPPO
task_registry.register('nero_ee_tracking', NeroEETracking, NeroEETrackingCfg(), NeroEETrackingCfgPPO())

from .nero.nero_ee_tracking_stable import NeroEETrackingStable
from .nero.nero_ee_tracking_stable_config import NeroEETrackingStableCfg, NeroEETrackingStableCfgPPO
task_registry.register('nero_ee_tracking_stable', NeroEETrackingStable,
                       NeroEETrackingStableCfg(), NeroEETrackingStableCfgPPO())

from .nero.nero_ee_tracking_limit import NeroEETrackingLimit
from .nero.nero_ee_tracking_limit_config import NeroEETrackingLimitCfg, NeroEETrackingLimitCfgPPO
task_registry.register('nero_ee_tracking_limit', NeroEETrackingLimit,
                       NeroEETrackingLimitCfg(), NeroEETrackingLimitCfgPPO())
