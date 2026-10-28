"""Isaac-independent single-joint command ABI, shared by training and deployment."""

RATE_LIMITED_MODE = "absolute_position_full_urdf_range_slew_v2"
LEGACY_MODE = "absolute_position_full_urdf_range"
A2NERO_STAGE1_MODE = "a2nero_stage1_offset_slew_v3"
POLICY_DT = 0.02
VELOCITY_LIMITS = (3., 3., 3., 4., 4., 4., 4.)
COMMAND_SPEED_FRACTION = 0.99


def limit_position_target(desired, previous, lower, upper, max_delta):
    """NumPy/Torch array operation; previous is the last issued, pre-delay target.

    This bounds command slew, not measured mechanical velocity. The explicit
    margin below vmax also covers ordinary float32 rounding at joint limits.
    """
    desired = desired.clip(lower, upper)
    return (previous + (desired - previous).clip(-max_delta, max_delta)).clip(lower, upper)
