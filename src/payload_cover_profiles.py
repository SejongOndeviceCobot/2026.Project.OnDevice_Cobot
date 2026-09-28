"""Named payload sphere-grid selection; no geometry or physics changes.

Absent cover_profile preserves the caller's existing arbitrary sphere-budget
path. Explicit profiles bind a registered grid to an exact JSON integer count.
"""
from collections.abc import Mapping
from types import MappingProxyType

_PROFILES = MappingProxyType({
    "legacy64": ((4, 4, 4), 64),
    "grid12x12x3": ((12, 12, 3), 432),
    "grid8x8x8": ((8, 8, 8), 512),
    "grid10x7x7": ((10, 7, 7), 490),
})


def resolve_payload_cover_profile(name):
    """Return a fresh receipt for a registered name; reject every other value."""
    if type(name) is not str or name not in _PROFILES:
        raise ValueError("Unknown payload cover profile; expected a registered payload grid")
    cells, count = _PROFILES[name]
    return {
        "schema": "depallet.payload_cover_profile.v1",
        "profile": name,
        "cells": list(cells),
        "sphere_count": count,
    }


def payload_cover_cells(payload):
    """Return a named grid or None without changing the legacy budget contract.

    payload is the JSON payload object, not the surrounding planner request.
    Non-payload requests should skip this helper or pass an empty object.
    """
    if not isinstance(payload, Mapping):
        raise ValueError("Payload must be a mapping")
    if "cover_profile" not in payload:
        return None
    profile = resolve_payload_cover_profile(payload["cover_profile"])
    count = payload.get("num_spheres")
    if type(count) is not int or count != profile["sphere_count"]:
        raise ValueError("Explicit payload cover profile requires its exact integer num_spheres")
    return list(profile["cells"])
