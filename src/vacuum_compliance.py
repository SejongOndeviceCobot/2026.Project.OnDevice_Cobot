"""CPU-only assumed D6 compliance parameters in explicit SI and USD units.

The payload is an aligned solid cuboid, with a top-center contact and a carrier
approximated as fixed. These are diagnostic parameters, not VGP20 calibration.
No USD, PhysX, GPU, environment, or scene is initialized by this module.
"""
import math
from numbers import Real


def _number(value, name, lower, upper):
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a real number")
    value = float(value)
    if not math.isfinite(value) or not lower <= value <= upper:
        raise ValueError(f"{name} must be finite and within [{lower}, {upper}]")
    return value


def build_vacuum_compliance(mass_kg, dimensions_m, tcp_to_com_m,
                            *, physics_dt_s=1/240, position_iterations=32,
                            axial_stiffness_n_per_m=None, damping_ratio=1.,
                            lateral_compliance_m=0., lateral_stiffness_n_per_m=2500.):
    """Return force-drive settings; angular USD gains are per degree.

    ``drives[axis]`` has stiffness_usd, damping_usd, type, and zero targets.
    ``limits[axis]`` has low/high; low > high retains the transX/transY lock.
    Damping is 2*zeta*sqrt(K*I) (rotation) or 2*zeta*sqrt(K*m) (translation).
    Auto axial stiffness is max(2500, mass*9.81/0.003) N/m; explicit settings
    must also meet the assumed 3mm static-gravity-extension design budget.
    Damping ratio zeta is applied to every driven axis, using the
    assumed payload inertia about the top contact. Actual articulated carrier
    effective inertia and vacuum force/torque limits are not certified.
    lateral_compliance_m=0 preserves the XY hard locks. A positive limit
    enables symmetric XY translation springs/dampers with the same zeta.
    This is assumed rubber-pad compliance, not a measured material model.
    """
    mass = _number(mass_kg, "mass_kg", .001, 20.)
    if not isinstance(dimensions_m, (list, tuple)) or len(dimensions_m) != 3:
        raise ValueError("dimensions_m must contain three SI lengths")
    dims = [_number(v, f"dimensions_m[{i}]", .001, 2.) for i, v in enumerate(dimensions_m)]
    distance = _number(tcp_to_com_m, "tcp_to_com_m", .0005, 1.02)
    if not dims[2]/2-1e-6 <= distance <= dims[2]/2+.02:
        raise ValueError("Top-contact distance must be between half-height and half-height + 20mm")
    dt = _number(physics_dt_s, "physics_dt_s", 1/2000, 1/60)
    if isinstance(position_iterations, bool) or not isinstance(position_iterations, int) or not 1 <= position_iterations <= 255:
        raise ValueError("position_iterations must be an integer within 1..255")
    ratio = _number(damping_ratio, "damping_ratio", .25, 4.)
    lateral = _number(lateral_compliance_m, "lateral_compliance_m", 0., .01)
    k_lateral = _number(lateral_stiffness_n_per_m, "lateral_stiffness_n_per_m", 1000., 100000.)
    k_linear = (max(2500., mass*9.81/.003) if axial_stiffness_n_per_m is None
                else _number(axial_stiffness_n_per_m, "axial_stiffness_n_per_m", 1000., 100000.))
    static_extension = mass*9.81/k_linear
    if static_extension > .003+1e-12:
        raise ValueError("Axial stiffness exceeds the 3mm static gravity extension budget")
    x, y, z = dims
    inertia_com = [mass*(y*y+z*z)/12, mass*(x*x+z*z)/12, mass*(x*x+y*y)/12]
    inertia_tcp = [inertia_com[0]+mass*distance*distance,
                   inertia_com[1]+mass*distance*distance, inertia_com[2]]
    angular_to_usd = math.pi/180.
    si, drives, limits, omega_dt, limit_torque = {}, {}, {}, {}, {}
    for axis, stiffness, inertia in zip(("rotX", "rotY", "rotZ"), (100., 100., 100.), inertia_tcp):
        damping = 2*ratio*math.sqrt(stiffness*inertia)
        natural = math.sqrt(stiffness/inertia)
        si[axis] = {"stiffness_nm_per_rad":stiffness, "damping_nm_s_per_rad":damping,
                    "payload_inertia_at_tcp_kg_m2":inertia, "natural_frequency_rad_s":natural,
                    "assumed_damping_ratio":ratio}
        drives[axis] = {"type":"force", "stiffness_usd":stiffness*angular_to_usd,
                        "damping_usd":damping*angular_to_usd, "target_position":0., "target_velocity":0.,
                        "stiffness_units":"N*m/degree", "damping_units":"N*m*s/degree",
                        "target_position_units":"degree", "target_velocity_units":"degree/s"}
        limits[axis] = {"low":-3., "high":3., "units":"degree"}
        omega_dt[axis] = natural*dt
        limit_torque[axis] = stiffness*math.radians(3.)
    d_linear = 2*ratio*math.sqrt(k_linear*mass)
    si["transZ"] = {"stiffness_n_per_m":k_linear, "damping_n_s_per_m":d_linear,
                    "effective_mass_kg":mass, "natural_frequency_rad_s":math.sqrt(k_linear/mass),
                    "assumed_damping_ratio":ratio}
    drives["transZ"] = {"type":"force", "stiffness_usd":k_linear, "damping_usd":d_linear,
                         "target_position":0., "target_velocity":0., "stiffness_units":"N/m",
                         "damping_units":"N*s/m", "target_position_units":"m", "target_velocity_units":"m/s"}
    limits["transZ"] = {"low":0., "high":.01, "units":"m"}
    for axis in ("transX", "transY"):
        if lateral == 0.:
            limits[axis] = {"low":1., "high":-1., "units":"m", "locked":True}
            continue
        damping_lateral = 2*ratio*math.sqrt(k_lateral*mass)
        natural_lateral = math.sqrt(k_lateral/mass)
        si[axis] = {"stiffness_n_per_m":k_lateral, "damping_n_s_per_m":damping_lateral,
                    "effective_mass_kg":mass, "natural_frequency_rad_s":natural_lateral,
                    "assumed_damping_ratio":ratio}
        drives[axis] = {"type":"force", "stiffness_usd":k_lateral, "damping_usd":damping_lateral,
                        "target_position":0., "target_velocity":0., "stiffness_units":"N/m",
                        "damping_units":"N*s/m", "target_position_units":"m", "target_velocity_units":"m/s"}
        limits[axis] = {"low":-lateral, "high":lateral, "units":"m", "locked":False}
        omega_dt[axis] = natural_lateral*dt
    omega_dt["transZ"] = math.sqrt(k_linear/mass)*dt
    return {"schema":"depallet.vacuum_compliance.v1", "parameter_set":"assumed_lateral_compliance_si_v3" if lateral else "assumed_damped_si_v2",
            "inputs":{"mass_kg":mass, "dimensions_m":dims, "tcp_to_com_distance_m":distance,
                      "physics_dt_s":dt, "position_iterations":position_iterations,
                      "requested_axial_stiffness_n_per_m":axial_stiffness_n_per_m,
                      "damping_ratio":ratio, "lateral_compliance_m":lateral,
                      "lateral_stiffness_n_per_m":k_lateral},
            "inertia":{"com_diagonal_kg_m2":inertia_com, "tcp_diagonal_kg_m2":inertia_tcp,
                       "parallel_axis_offset_m":[0.,0.,distance], "carrier_approximation":"fixed",
                       "cuboid_axes_assumed_aligned_with_tcp":True},
            "si_drives":si, "drives":drives, "limits":limits,
            "axes":{axis:{**limits[axis],**drive} for axis,drive in drives.items()},
            "locked_axes":{axis:dict(limits[axis]) for axis in ("transX","transY") if limits[axis].get("locked",False)},
            "lateral_compliance":{"enabled":bool(lateral), "limit_per_axis_m":lateral,
                "stiffness_n_per_m":k_lateral if lateral else None,
                "damping_ratio":ratio if lateral else None,
                "maximum_xy_limit_vector_norm_m":math.sqrt(2)*lateral,
                "assumption":"unmeasured rubber vacuum-pad lateral compliance",
                "parameters_measured":False},
            "units":{"meters_per_unit":1., "kilograms_per_unit":1.,
                     "angular_si_to_usd_multiplier":angular_to_usd,
                     "source":"OpenUSD PhysicsDriveAPI angular stiffness/damping are per degree"},
            "diagnostics":{"natural_frequency_times_physics_dt":omega_dt,
                           "natural_frequency_times_tgs_internal_dt":{k:v/position_iterations for k,v in omega_dt.items()},
                           "spring_torque_at_three_degree_limit_nm":limit_torque,
                           "static_gravity_extension_m":static_extension,
                           "static_extension_design_budget_m":.003,
                           "axial_stiffness_selection":"max(2500, mass_kg*9.81/0.003)" if axial_stiffness_n_per_m is None else "explicit N/m",
                           "damping_ratio_applies_to_all_driven_axes":True,
                           "lateral_spring_force_at_limit_N":{axis:k_lateral*lateral for axis in ("transX","transY")} if lateral else {},
                           "gravity_acceleration_assumed_m_s2":9.81,
                           "physics_step_frequency_product_above_one":[k for k,v in omega_dt.items() if v>1.]},
            "runtime_monitor":{"provided_by_caller_separately":True, "limits":None, "execution_authorized":False},
            "warnings":["Gains and payload properties are assumptions, not measured VGP20 parameters.",
                        "Damping uses the chosen ratio times payload-only critical damping about an approximately fixed carrier; actual coupled inertia differs.",
                        "The 100 N*m/rad angular springs each reach 5.236 N*m at 3 degrees; no vacuum torque capacity is certified.",
                        "Finite drive force/torque limits and measured non-target stack support require separate validation.",
                        "Optional XY compliance is assumed rubber-pad motion, not measured VGP20 behavior. Combined-axis motion can exceed the caller-provided attachment bounds; this helper neither selects runtime monitor limits nor authorizes execution."],
            "parameters_measured":False, "real_gripper_calibration":False, "force_torque_capacity_validated":False,
            "physical_execution_validated":False, "gpu_used":False}


# Descriptive alias retained for direct CPU callers.
vacuum_compliance_parameters = build_vacuum_compliance
