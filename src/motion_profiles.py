"""Named planner motion limits shared by CPU request builders and audits."""
from __future__ import annotations

from collections.abc import MutableMapping
from types import MappingProxyType


_PROFILES = MappingProxyType({
    "baseline": MappingProxyType({
        "maximum_velocity_rad_s": .25,
        "maximum_acceleration_rad_s2": .5,
        "maximum_jerk_rad_s3": 5.,
    }),
    "brisk": MappingProxyType({
        "maximum_velocity_rad_s": .32,
        "maximum_acceleration_rad_s2": .5,
        "maximum_jerk_rad_s3": 5.,
    }),
    # Intermediate velocity-only candidate. This isolates the usable speed
    # envelope after brisk36 failed its strict physical stabilization gate.
    "brisk34": MappingProxyType({
        "maximum_velocity_rad_s": .34,
        "maximum_acceleration_rad_s2": .5,
        "maximum_jerk_rad_s3": 5.,
    }),
    # Candidate speed profile: velocity-only increase from brisk.  It is not
    # promoted until strict physical prefix validation supplies fresh evidence.
    "brisk36": MappingProxyType({
        "maximum_velocity_rad_s": .36,
        "maximum_acceleration_rad_s2": .5,
        "maximum_jerk_rad_s3": 5.,
    }),
})

PROFILE_NAMES = tuple(_PROFILES)
LIMIT_FIELDS = tuple(next(iter(_PROFILES.values())))


def resolve_motion_profile(name="baseline"):
    """Return a fresh JSON-ready receipt for a registered profile name."""
    if type(name) is not str or name not in _PROFILES:
        raise ValueError("Unknown motion profile")
    return {"schema": "depallet.motion_profile.v1", "profile": name,
            **dict(_PROFILES[name])}


def apply_motion_profile(request, name="baseline"):
    """Bind a mutable request to one registered profile and its exact limits."""
    if not isinstance(request, MutableMapping):
        raise ValueError("Planner request must be a mutable mapping")
    profile = resolve_motion_profile(name)
    request["motion_profile"] = profile["profile"]
    for field in LIMIT_FIELDS:
        request[field] = profile[field]
    return request
