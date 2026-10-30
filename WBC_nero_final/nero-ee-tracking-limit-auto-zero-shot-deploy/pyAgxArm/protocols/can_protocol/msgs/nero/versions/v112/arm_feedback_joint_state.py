from typing import Union

from ....core.attritube_base import AttributeBase


class ArmMsgFeedbackJointStateV112(AttributeBase):
    """One Nero V112 joint position from CAN IDs 0x511 through 0x517.

    The position occupies bytes 4..7 as an IEEE-754 little-endian float in
    radians.  The meaning/scaling of bytes 0..3 has not yet been verified on
    hardware, so this read-only message intentionally exposes position only.
    """

    def __init__(
        self,
        position: Union[int, float] = 0.0,
        valid: bool = False,
    ):
        self.position = position
        self.valid = valid


class ArmMsgFeedbackJointStateV1121(ArmMsgFeedbackJointStateV112):
    """CAN ID 0x511."""


class ArmMsgFeedbackJointStateV1122(ArmMsgFeedbackJointStateV112):
    """CAN ID 0x512."""


class ArmMsgFeedbackJointStateV1123(ArmMsgFeedbackJointStateV112):
    """CAN ID 0x513."""


class ArmMsgFeedbackJointStateV1124(ArmMsgFeedbackJointStateV112):
    """CAN ID 0x514."""


class ArmMsgFeedbackJointStateV1125(ArmMsgFeedbackJointStateV112):
    """CAN ID 0x515."""


class ArmMsgFeedbackJointStateV1126(ArmMsgFeedbackJointStateV112):
    """CAN ID 0x516."""


class ArmMsgFeedbackJointStateV1127(ArmMsgFeedbackJointStateV112):
    """CAN ID 0x517."""
