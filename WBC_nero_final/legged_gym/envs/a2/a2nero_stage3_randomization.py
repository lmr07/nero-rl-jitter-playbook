"""Isaac-Gym-independent helpers for A2NERO rigid-body randomization.

The functions in this module deliberately operate on arrays plus rigid-body
names.  Isaac Gym may change body indices when fixed joints are collapsed, so
an index is never assigned a physical meaning here.

The 24 legacy privileged mass parameters are packed as::

    [base added mass (1), base COM delta (3), payload added mass (1),
     rigid mass delta slots (19)]

Only sixteen of the final nineteen slots are used.  The final three are kept
at zero to preserve the checkpoint/critic ABI without assigning arm or sensor
bodies accidental "leg" semantics.
"""

from dataclasses import dataclass
from typing import Dict, Mapping, Optional, Sequence, Tuple, Union

import numpy as np


BASE_BODY_NAME = "base_link"

# This order is an ABI: do not derive it from an asset's current body order.
A2_LEG_BODY_NAMES: Tuple[str, ...] = (
    "FL_hip",
    "FL_thigh",
    "FL_calf",
    "FL_foot",
    "FR_hip",
    "FR_thigh",
    "FR_calf",
    "FR_foot",
    "RL_hip",
    "RL_thigh",
    "RL_calf",
    "RL_foot",
    "RR_hip",
    "RR_thigh",
    "RR_calf",
    "RR_foot",
)

# With the repository asset's collapse_fixed_joints=True import, the fixed
# gripper flange/base assembly is merged into nero_link7.  Keep this separate
# from asset.gripper_name, which currently names one moving finger.
DEFAULT_PAYLOAD_BODY_NAME = "nero_link7"

NUM_RIGID_MASS_DELTA_SLOTS = 19
RIGID_MASS_DELTA_SLOT_BODY_NAMES: Tuple[Optional[str], ...] = (
    A2_LEG_BODY_NAMES + (None, None, None)
)
NUM_PRIVILEGED_MASS_PARAMS = 24


FloatArrayLike = Union[float, Sequence[float], np.ndarray]


@dataclass(frozen=True)
class BodySelection:
    """Resolved, name-based body selection for one imported asset."""

    index_by_name: Dict[str, int]
    base_index: int
    leg_indices: Tuple[int, ...]
    payload_index: Optional[int]

    @property
    def mapping_text(self) -> str:
        return format_body_name_index_mapping(self.index_by_name)


@dataclass(frozen=True)
class RigidBodyRandomizationParameters:
    """One episode-persistent draw of rigid-body parameters."""

    base_added_mass: float
    base_com_delta: np.ndarray
    leg_mass_scales: np.ndarray
    payload_added_mass: float


@dataclass(frozen=True)
class RigidBodyRandomizationResult:
    """Randomized properties and their fixed-size privileged encoding."""

    masses: np.ndarray
    coms: np.ndarray
    mass_delta_by_body: np.ndarray
    rigid_mass_delta_slots: np.ndarray
    privileged_mass_params: np.ndarray
    selection: BodySelection


def format_body_name_index_mapping(
    body_names_or_mapping: Union[Sequence[str], Mapping[str, int]],
) -> str:
    """Return a stable, human-readable ``body name -> index`` mapping."""

    if isinstance(body_names_or_mapping, Mapping):
        items = sorted(body_names_or_mapping.items(), key=lambda item: item[1])
    else:
        items = list(zip(body_names_or_mapping, range(len(body_names_or_mapping))))
    return ", ".join("{}={}".format(name, index) for name, index in items)


def build_body_name_to_index(body_names: Sequence[str]) -> Dict[str, int]:
    """Build a checked body-name mapping.

    Duplicate names are rejected because silently selecting the final duplicate
    would make a physical randomization ambiguous.
    """

    index_by_name: Dict[str, int] = {}
    duplicates = []
    for index, name in enumerate(body_names):
        if name in index_by_name:
            duplicates.append(name)
        index_by_name[name] = index
    if duplicates:
        raise ValueError(
            "Duplicate rigid-body names {}. Imported mapping: {}".format(
                sorted(set(duplicates)), format_body_name_index_mapping(body_names)
            )
        )
    return index_by_name


def resolve_body_selection(
    body_names: Sequence[str],
    *,
    base_body_name: str = BASE_BODY_NAME,
    leg_body_names: Sequence[str] = A2_LEG_BODY_NAMES,
    payload_body_name: Optional[str] = DEFAULT_PAYLOAD_BODY_NAME,
) -> BodySelection:
    """Resolve and validate the bodies that Stage-3 is allowed to modify."""

    index_by_name = build_body_name_to_index(body_names)
    requested_names = [base_body_name] + list(leg_body_names)
    if payload_body_name is not None:
        requested_names.append(payload_body_name)
    missing = [name for name in requested_names if name not in index_by_name]
    if missing:
        raise ValueError(
            "Missing configured rigid bodies {}. Imported mapping: {}".format(
                missing, format_body_name_index_mapping(index_by_name)
            )
        )

    if len(set(leg_body_names)) != len(leg_body_names):
        raise ValueError("leg_body_names must be unique")
    if base_body_name in leg_body_names:
        raise ValueError("The base body must not also be selected as an A2 leg body")
    if payload_body_name is not None and (
        payload_body_name == base_body_name or payload_body_name in leg_body_names
    ):
        raise ValueError("The payload body must be distinct from base and leg bodies")

    return BodySelection(
        index_by_name=index_by_name,
        base_index=index_by_name[base_body_name],
        leg_indices=tuple(index_by_name[name] for name in leg_body_names),
        payload_index=(
            None if payload_body_name is None else index_by_name[payload_body_name]
        ),
    )


def assert_body_selection(
    body_names: Sequence[str],
    *,
    base_body_name: str = BASE_BODY_NAME,
    leg_body_names: Sequence[str] = A2_LEG_BODY_NAMES,
    payload_body_name: Optional[str] = DEFAULT_PAYLOAD_BODY_NAME,
) -> BodySelection:
    """Assertion-flavoured wrapper whose error includes the imported mapping."""

    try:
        return resolve_body_selection(
            body_names,
            base_body_name=base_body_name,
            leg_body_names=leg_body_names,
            payload_body_name=payload_body_name,
        )
    except ValueError as error:
        raise AssertionError(str(error)) from error


def _coerce_leg_mass_scales(
    leg_mass_scales: Union[FloatArrayLike, Mapping[str, float]],
    leg_body_names: Sequence[str],
) -> np.ndarray:
    if isinstance(leg_mass_scales, Mapping):
        unexpected = sorted(set(leg_mass_scales) - set(leg_body_names))
        if unexpected:
            raise ValueError("Unexpected leg mass scale names: {}".format(unexpected))
        return np.asarray(
            [leg_mass_scales.get(name, 0.0) for name in leg_body_names],
            dtype=np.float64,
        )

    scales = np.asarray(leg_mass_scales, dtype=np.float64)
    if scales.ndim == 0:
        return np.full(len(leg_body_names), float(scales), dtype=np.float64)
    if scales.shape != (len(leg_body_names),):
        raise ValueError(
            "leg_mass_scales must be scalar or shape ({},), got {}".format(
                len(leg_body_names), scales.shape
            )
        )
    return scales.copy()


def pack_rigid_mass_delta_slots(
    mass_delta_by_name: Mapping[str, float],
    slot_body_names: Sequence[Optional[str]] = RIGID_MASS_DELTA_SLOT_BODY_NAMES,
) -> np.ndarray:
    """Pack body mass deltas into the fixed nineteen-slot critic ABI."""

    if len(slot_body_names) != NUM_RIGID_MASS_DELTA_SLOTS:
        raise ValueError(
            "Rigid mass slot ABI requires {} entries, got {}".format(
                NUM_RIGID_MASS_DELTA_SLOTS, len(slot_body_names)
            )
        )
    used_names = [name for name in slot_body_names if name is not None]
    if len(used_names) != len(set(used_names)):
        raise ValueError("A rigid body may occupy at most one mass-delta slot")
    return np.asarray(
        [0.0 if name is None else mass_delta_by_name.get(name, 0.0)
         for name in slot_body_names],
        dtype=np.float64,
    )


def apply_rigid_body_randomization(
    body_names: Sequence[str],
    nominal_masses: FloatArrayLike,
    nominal_coms: FloatArrayLike,
    parameters: RigidBodyRandomizationParameters,
    *,
    base_body_name: str = BASE_BODY_NAME,
    leg_body_names: Sequence[str] = A2_LEG_BODY_NAMES,
    payload_body_name: Optional[str] = DEFAULT_PAYLOAD_BODY_NAME,
    slot_body_names: Sequence[Optional[str]] = RIGID_MASS_DELTA_SLOT_BODY_NAMES,
) -> RigidBodyRandomizationResult:
    """Apply one draw from nominal properties using body names only.

    Inputs are copied.  Calling this function repeatedly with the same nominal
    properties and episode parameters is therefore persistent and cannot
    accidentally accumulate mass/COM deltas across resets.
    """

    selection = resolve_body_selection(
        body_names,
        base_body_name=base_body_name,
        leg_body_names=leg_body_names,
        payload_body_name=payload_body_name,
    )
    masses = np.asarray(nominal_masses, dtype=np.float64)
    coms = np.asarray(nominal_coms, dtype=np.float64)
    body_count = len(body_names)
    if masses.shape != (body_count,):
        raise ValueError(
            "nominal_masses must have shape ({},), got {}".format(
                body_count, masses.shape
            )
        )
    if coms.shape != (body_count, 3):
        raise ValueError(
            "nominal_coms must have shape ({}, 3), got {}".format(
                body_count, coms.shape
            )
        )
    if not np.all(np.isfinite(masses)) or not np.all(np.isfinite(coms)):
        raise ValueError("Nominal rigid-body properties must be finite")
    if np.any(masses <= 0.0):
        raise ValueError("Every nominal rigid-body mass must be positive")

    masses = masses.copy()
    coms = coms.copy()
    nominal_masses_array = masses.copy()
    base_com_delta = np.asarray(parameters.base_com_delta, dtype=np.float64)
    if base_com_delta.shape != (3,):
        raise ValueError("base_com_delta must have shape (3,)")
    leg_scales = _coerce_leg_mass_scales(
        parameters.leg_mass_scales, leg_body_names
    )
    if not (
        np.isfinite(parameters.base_added_mass)
        and np.all(np.isfinite(base_com_delta))
        and np.all(np.isfinite(leg_scales))
        and np.isfinite(parameters.payload_added_mass)
    ):
        raise ValueError("Randomization parameters must be finite")

    masses[selection.base_index] += float(parameters.base_added_mass)
    coms[selection.base_index] += base_com_delta
    for body_index, scale in zip(selection.leg_indices, leg_scales):
        masses[body_index] = nominal_masses_array[body_index] * (1.0 + scale)

    if selection.payload_index is None:
        if parameters.payload_added_mass != 0.0:
            raise ValueError("payload_added_mass requires an explicit payload body")
    else:
        masses[selection.payload_index] += float(parameters.payload_added_mass)

    if np.any(masses <= 0.0):
        raise ValueError("Randomization produced a non-positive rigid-body mass")

    mass_delta_by_body = masses - nominal_masses_array
    mass_delta_by_name = {
        name: float(mass_delta_by_body[index])
        for name, index in selection.index_by_name.items()
    }
    rigid_slots = pack_rigid_mass_delta_slots(
        mass_delta_by_name, slot_body_names=slot_body_names
    )
    privileged_mass_params = np.concatenate(
        (
            np.asarray([parameters.base_added_mass], dtype=np.float64),
            base_com_delta,
            np.asarray([parameters.payload_added_mass], dtype=np.float64),
            rigid_slots,
        )
    )
    if privileged_mass_params.shape != (NUM_PRIVILEGED_MASS_PARAMS,):
        raise AssertionError("Privileged mass parameter ABI changed unexpectedly")

    return RigidBodyRandomizationResult(
        masses=masses,
        coms=coms,
        mass_delta_by_body=mass_delta_by_body,
        rigid_mass_delta_slots=rigid_slots,
        privileged_mass_params=privileged_mass_params,
        selection=selection,
    )


def _checked_range(name: str, value_range: Sequence[float]) -> Tuple[float, float]:
    values = np.asarray(value_range, dtype=np.float64)
    if values.shape != (2,) or not np.all(np.isfinite(values)):
        raise ValueError("{} must be a finite [low, high] range".format(name))
    low, high = float(values[0]), float(values[1])
    if low > high:
        raise ValueError("{} has low > high".format(name))
    return low, high


def sample_rigid_body_randomization_parameters(
    rng: np.random.Generator,
    *,
    base_added_mass_range: Sequence[float],
    base_com_ranges: Sequence[Sequence[float]],
    leg_mass_scale_range: Sequence[float],
    payload_added_mass_range: Sequence[float],
    num_leg_bodies: int = len(A2_LEG_BODY_NAMES),
) -> RigidBodyRandomizationParameters:
    """Sample a parameter object once per reset, then reuse it for the episode."""

    base_mass_low, base_mass_high = _checked_range(
        "base_added_mass_range", base_added_mass_range
    )
    leg_low, leg_high = _checked_range(
        "leg_mass_scale_range", leg_mass_scale_range
    )
    payload_low, payload_high = _checked_range(
        "payload_added_mass_range", payload_added_mass_range
    )
    com_ranges = np.asarray(base_com_ranges, dtype=np.float64)
    if com_ranges.shape != (3, 2) or not np.all(np.isfinite(com_ranges)):
        raise ValueError("base_com_ranges must have shape (3, 2)")
    if np.any(com_ranges[:, 0] > com_ranges[:, 1]):
        raise ValueError("base_com_ranges contains low > high")
    if num_leg_bodies <= 0:
        raise ValueError("num_leg_bodies must be positive")

    return RigidBodyRandomizationParameters(
        base_added_mass=float(rng.uniform(base_mass_low, base_mass_high)),
        base_com_delta=rng.uniform(com_ranges[:, 0], com_ranges[:, 1]),
        leg_mass_scales=rng.uniform(leg_low, leg_high, size=num_leg_bodies),
        payload_added_mass=float(rng.uniform(payload_low, payload_high)),
    )


def sample_stratified_base_com_deltas(
    rng: np.random.Generator,
    count: int,
    base_com_ranges: Sequence[Sequence[float]],
) -> np.ndarray:
    """Return a reproducible Latin-hypercube draw for static GPU COMs.

    Isaac Gym's GPU tensor pipeline cannot change a rigid body's COM after
    ``prepare_sim``.  Stage-3 therefore needs one well-covered COM assignment
    at actor creation instead of an ordinary sequence of reset-time draws.
    Each axis occupies every one of ``count`` equal-probability strata exactly
    once, and independent permutations prevent an env-id/axis correlation.
    """

    if isinstance(count, bool) or not isinstance(count, (int, np.integer)):
        raise ValueError("count must be a positive integer")
    count = int(count)
    if count <= 0:
        raise ValueError("count must be a positive integer")
    com_ranges = np.asarray(base_com_ranges, dtype=np.float64)
    if com_ranges.shape != (3, 2) or not np.all(np.isfinite(com_ranges)):
        raise ValueError("base_com_ranges must have shape (3, 2)")
    if np.any(com_ranges[:, 0] > com_ranges[:, 1]):
        raise ValueError("base_com_ranges contains low > high")

    samples = np.empty((count, 3), dtype=np.float64)
    for axis in range(3):
        # Jitter within each stratum preserves a uniform marginal while the
        # permutation removes monotonic coupling to actor/terrain indices.
        unit_samples = (np.arange(count, dtype=np.float64) + rng.random(count)) / count
        unit_samples = unit_samples[rng.permutation(count)]
        low, high = com_ranges[axis]
        samples[:, axis] = low + (high - low) * unit_samples
    return samples


__all__ = [
    "A2_LEG_BODY_NAMES",
    "BASE_BODY_NAME",
    "DEFAULT_PAYLOAD_BODY_NAME",
    "NUM_PRIVILEGED_MASS_PARAMS",
    "NUM_RIGID_MASS_DELTA_SLOTS",
    "RIGID_MASS_DELTA_SLOT_BODY_NAMES",
    "BodySelection",
    "RigidBodyRandomizationParameters",
    "RigidBodyRandomizationResult",
    "apply_rigid_body_randomization",
    "assert_body_selection",
    "build_body_name_to_index",
    "format_body_name_index_mapping",
    "pack_rigid_mass_delta_slots",
    "resolve_body_selection",
    "sample_stratified_base_com_deltas",
    "sample_rigid_body_randomization_parameters",
]
