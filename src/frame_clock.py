"""Validate rendered-frame clocks without importing Isaac or allocating GPU state."""
from __future__ import annotations

import math
from fractions import Fraction
from numbers import Integral, Real


class RenderClock:
    """Accept fresh rendered timestamps; preserve state on duplicates and errors.

    Missing or zero annotator values are warm-up only before the first accepted
    frame. Simulation and Fabric reference clocks are compared independently;
    their epochs and units are not assumed to match.
    """

    def __init__(self) -> None:
        self._last_image_time: float | None = None
        self._last_reference: Fraction | None = None

    @staticmethod
    def _time(value: object, name: str) -> float:
        if isinstance(value, bool) or not isinstance(value, Real):
            raise ValueError(f"{name} must be a finite real number")
        try:
            result = float(value)
        except (ValueError, OverflowError) as exc:
            raise ValueError(f"{name} must be finite") from exc
        if not math.isfinite(result) or result < 0:
            raise ValueError(f"{name} must be finite and nonnegative")
        return result

    @staticmethod
    def _integer(value: object, name: str) -> int:
        if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
            raise ValueError(f"{name} must be a nonnegative integer")
        return int(value)

    def accept(self, sim_time_data: dict | None, reference_data: dict | None, physics_time: float) -> dict | None:
        """Return three clock fields for a fresh frame, or None during warm-up/duplicates.

        Raises ValueError on malformed input, post-start missing clocks, rewind,
        inconsistent clock advancement or an image more than 1 us in the future.
        """
        physics = self._time(physics_time, "physics_time")
        for payload in (sim_time_data, reference_data):
            if payload is not None and not isinstance(payload, dict):
                raise ValueError("annotator clock payload must be a dictionary or None")
        sim_time_data = sim_time_data or {}
        reference_data = reference_data or {}
        raw_image = sim_time_data.get("simulationTime")
        raw_numerator = reference_data.get("referenceTimeNumerator")
        raw_denominator = reference_data.get("referenceTimeDenominator")
        image = None if raw_image is None else self._time(raw_image, "simulationTime")
        numerator = None if raw_numerator is None else self._integer(raw_numerator, "referenceTimeNumerator")
        denominator = None if raw_denominator is None else self._integer(raw_denominator, "referenceTimeDenominator")
        if image is not None and image > physics + 1e-6:
            raise ValueError("rendered image time is in the future of the physics snapshot")
        if image in (None, 0.0) or numerator in (None, 0) or denominator in (None, 0):
            if self._last_image_time is None:
                return None
            raise ValueError("missing or zero clock after rendered frames started; reset is not permitted")
        reference = Fraction(numerator, denominator)
        if self._last_image_time is not None:
            if reference == self._last_reference and image == self._last_image_time:
                return None
            if reference <= self._last_reference or image <= self._last_image_time:
                raise ValueError("image and reference clocks must advance together and strictly increase")
        self._last_image_time = image
        self._last_reference = reference
        return {
            "image_sim_time": image,
            "reference_time_numerator": numerator,
            "reference_time_denominator": denominator,
        }
