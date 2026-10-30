#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import re
from typing import Tuple

from .arm_options import ArmModel, NeroFW, PiperFW


_PIPER_MODELS = {
    ArmModel.PIPER,
    ArmModel.PIPER_H,
    ArmModel.PIPER_L,
    ArmModel.PIPER_X,
}
_PIPER_VERSION_PATTERN = re.compile(r"S-V(\d+)\.(\d+)-(\d+)")
_NERO_VERSION_PATTERN = re.compile(r"(\d+)\.(\d{2,})")


def _parse_version(pattern, firmware_version: str, robot: str) -> Tuple[int, ...]:
    if not isinstance(firmware_version, str):
        raise TypeError("firmware_version must be a string")
    match = pattern.fullmatch(firmware_version.strip())
    if match is None:
        raise ValueError(
            "Invalid firmware version {!r} for robot {!r}".format(
                firmware_version, robot
            )
        )
    return tuple(int(part) if part is not None else 0 for part in match.groups())


def _parse_nero_version(firmware_version: str, robot: str) -> Tuple[int, int, int]:
    if not isinstance(firmware_version, str):
        raise TypeError("firmware_version must be a string")
    match = _NERO_VERSION_PATTERN.fullmatch(firmware_version.strip())
    if match is None:
        raise ValueError(
            "Invalid firmware version {!r} for robot {!r}".format(
                firmware_version, robot
            )
        )
    compact_minor = match.group(2)
    return (
        int(match.group(1)),
        int(compact_minor[:2]),
        int(compact_minor[2:] or "0"),
    )


def resolve_firmware_profile(robot: str, firmware_version: str) -> str:
    """Resolve a reported device firmware version to an SDK driver profile."""
    if robot in _PIPER_MODELS:
        version = _parse_version(_PIPER_VERSION_PATTERN, firmware_version, robot)
        if version >= (1, 8, 9):
            return PiperFW.V189
        if version >= (1, 8, 8):
            return PiperFW.V188
        if version >= (1, 8, 3):
            return PiperFW.V183
        return PiperFW.DEFAULT

    if robot == ArmModel.NERO:
        version = _parse_nero_version(firmware_version, robot)
        if version >= (1, 21, 0):
            return NeroFW.V121
        if version >= (1, 20, 0):
            return NeroFW.V120
        if version >= (1, 12, 0):
            return NeroFW.V112
        if version >= (1, 11, 0):
            return NeroFW.V111
        return NeroFW.DEFAULT

    raise ValueError("Unsupported robot model: {!r}".format(robot))
