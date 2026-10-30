from typing_extensions import Literal

from ..v120.driver import Driver as V120Driver


class Driver(V120Driver):
    """Nero CAN driver for firmware >= v121 (1.21).

    Firmware 1.21 fixed the velocity-direction defect that required older SDK
    profiles to negate J1-J5/J7 velocity commands. This adapter cancels that
    inherited workaround so every joint preserves the caller's sign.
    """

    def move_mit(
        self,
        joint_index: Literal[1, 2, 3, 4, 5, 6, 7],
        p_des: float = 0.0,
        v_des: float = 0.0,
        kp: float = 10.0,
        kd: float = 0.8,
        t_ff: float = 0.0,
    ):
        v_des = v_des if joint_index == 6 else -v_des
        return super().move_mit(
            joint_index=joint_index,
            p_des=p_des,
            v_des=v_des,
            kp=kp,
            kd=kd,
            t_ff=t_ff,
        )

    def move_cpv_vel(
        self,
        joint_index: Literal[1, 2, 3, 4, 5, 6, 7],
        vel: float,
    ) -> None:
        vel = vel if joint_index == 6 else -vel
        return super().move_cpv_vel(joint_index, vel)
