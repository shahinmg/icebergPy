"""
Updated melt.py - Using constants module

This shows how to update your melt functions to use the constants module.
Replace your existing melt.py with these changes.
"""

import numpy as np
from scipy.interpolate import interp1d, interp2d
from scipy.spatial import cKDTree
import xarray as xr

# Import constants
from . import constants as const


def melt_wave(wind_speed, sea_surface_temp, sea_ice_concentration):
    """
    Calculate wave erosion melt rate using Silva et al. formulation.
    
    Parameters
    ----------
    wind_speed : float or ndarray
        Wind speed in m/s (relative to water, assumed Wind >> water speeds)
    sea_surface_temp : float or ndarray
        Surface water temperature in °C
    sea_ice_concentration : float or ndarray
        Sea ice concentration, fraction 0-1 (0 = no ice, 1 = full coverage)
    
    Returns
    -------
    melt_rate : float or ndarray
        Melt rate in m/s
    
    References
    ----------
    Silva et al., Bigg et al. (1997), Gladstone et al. (2001)
    
    """
    # Wave height estimate from wind speed
    sea_state = (const.WAVE_HEIGHT_WIND_COEFF_1 * np.sqrt(wind_speed) + 
                 const.WAVE_HEIGHT_WIND_COEFF_2 * wind_speed)
    
    # Ice concentration effect (corrected from paper typo)
    ice_term = 1 + np.cos(np.power(sea_ice_concentration, 3) * np.pi)
    
    # Melt rate: m/day
    melt_rate = ((1 / const.WAVE_MELT_DIVISOR) * sea_state * ice_term * 
                 (sea_surface_temp + const.WAVE_MELT_TEMP_OFFSET))
    
    # Convert to m/s
    melt_rate = melt_rate / const.SECONDS_PER_DAY
    
    return melt_rate


def melt_solar(solar_radiation):
    """
    Calculate melt from solar radiation (above water).
    
    Based on Condron's mitberg formulation. Affects thickness above water only.
    Assumes constant albedo.
    
    Parameters
    ----------
    solar_radiation : float or ndarray
        Solar radiation flux downward (SW and LW) in W/m²
    
    Returns
    -------
    melt_rate : float or ndarray
        Melt rate in m/s
    
    Notes
    -----
    Albedo assumed to be 0.7 for ice
    """
    # Energy absorbed = solar flux * (1 - albedo)
    absorbed_energy = const.SOLAR_ABSORPTION_FRACTION * solar_radiation
    
    # Convert energy to melt: Q = ρ * L * m  =>  m = Q / (ρ * L)
    # where Q is heat flux (W/m²), ρ is ice density (kg/m³), L is latent heat (J/kg)
    melt_rate = absorbed_energy / (const.RHO_ICE * const.LATENT_HEAT_FUSION)
    
    return melt_rate  # m/s


def melt_forcedwater(temp_far, salinity_far, pressure_base, velocity_relative, 
                     factor=None, use_constant_tf=False, constant_tf=None):
    """
    Calculate forced convection melt in water (turbulent melt).
    
    Uses Silva et al. equation with parameters from Holland and Jenkins.
    
    Parameters
    ----------
    temp_far : float or ndarray
        Far-field water temperature in °C
    salinity_far : float or ndarray
        Far-field salinity in PSU
    pressure_base : float or ndarray
        Pressure at base in dbar
    velocity_relative : float or ndarray
        Water speed relative to iceberg surface in m/s
    factor : float, optional
        Adjustment factor for transfer coefficients (Jackson et al. 2020).
        Default is 4.
    use_constant_tf : bool, optional
        If True, use constant thermal forcing instead of calculating from T/S
    constant_tf : float, optional
        Constant thermal forcing value if use_constant_tf is True
    
    Returns
    -------
    melt_rate : float or ndarray
        Melt rate in m/s (positive = melting)
    thermal_forcing : float or ndarray
        Temperature above freezing point in °C (or constant_tf if specified)
    freezing_point : float or ndarray or None
        Freezing point temperature in °C (None if using constant_tf)
    
    References
    ----------
    Holland & Jenkins (1999), Silva et al., Jackson et al. (2020)
    """
    # Use default factor if not provided
    if factor is None:
        factor = const.DEFAULT_TRANSFER_COEFFICIENT_FACTOR
    
    # Transfer coefficients (adjusted by factor)
    heat_transfer = const.HEAT_TRANSFER_COEFFICIENT_GT * factor
    salt_transfer = const.SALT_TRANSFER_COEFFICIENT_GS * factor
    
    # Thermal forcing calculation
    if use_constant_tf:
        thermal_forcing = constant_tf
        freezing_point = None
    else:
        # Freezing point: T_fp = a*S + b + c*P
        freezing_point = (const.FREEZING_POINT_SALINITY_COEFF * salinity_far + 
                         const.FREEZING_POINT_CONSTANT + 
                         const.FREEZING_POINT_PRESSURE_COEFF * pressure_base)
        thermal_forcing = temp_far - freezing_point
    
    # Quadratic formula coefficients: A*m² + B*m + C = 0
    A = ((const.LATENT_HEAT_FUSION + const.TEMPERATURE_DIFFERENCE_CORE_SURFACE * const.SPECIFIC_HEAT_ICE) / 
         (velocity_relative * heat_transfer * const.SPECIFIC_HEAT_WATER))
    
    B = -1 * (((const.LATENT_HEAT_FUSION + const.TEMPERATURE_DIFFERENCE_CORE_SURFACE * const.SPECIFIC_HEAT_ICE) * 
               salt_transfer / (heat_transfer * const.SPECIFIC_HEAT_WATER)) - 
              const.FREEZING_POINT_SALINITY_COEFF * salinity_far - thermal_forcing)
    
    C = -velocity_relative * salt_transfer * thermal_forcing
    
    # Solve quadratic equation
    discriminant = np.power(B, 2) - 4 * A * C
    discriminant = np.where(discriminant < 0, np.nan, discriminant)
    
    # Find quadratic roots
    root1 = (-B + np.sqrt(discriminant)) / (2 * A)
    root2 = (-B - np.sqrt(discriminant)) / (2 * A)
    
    melt_rate = np.minimum(root1, root2)
    
    # Clean data: remove melt rates when water is below freezing
    mask = thermal_forcing < 0
    melt_rate = np.where(mask, 0, melt_rate)
    melt_rate = np.where(np.isnan(melt_rate), 0, melt_rate)
    
    # Make melt rate positive (convention: positive = melting)
    melt_rate = -1 * melt_rate
    
    return melt_rate, thermal_forcing, freezing_point


def melt_forcedair(air_temp, air_velocity_relative, iceberg_length):
    """
    Calculate forced convection melt in air.
    
    Based on Condron's mitberg formulation using Nusselt number relationships.
    
    Parameters
    ----------
    air_temp : float or ndarray
        Air temperature in °C
    air_velocity_relative : float or ndarray
        Air speed relative to iceberg in m/s
    iceberg_length : float
        Characteristic length of iceberg in meters
    
    Returns
    -------
    melt_rate : float or ndarray
        Melt rate in m/s (0 if air_temp < 0°C)
    
    Notes
    -----
    Uses Nusselt-Reynolds-Prandtl relationship for forced convection
    """
    is_freezing = air_temp < 0
    
    # Dimensionless numbers
    prandtl_number = const.PRANDTL_NUMBER
    reynolds_number = np.abs(air_velocity_relative) * iceberg_length / const.AIR_KINEMATIC_VISCOSITY
    
    # Nusselt number: Nu = 0.058 * Re^0.8 / Pr^0.4
    nusselt_number = (const.NUSSELT_COEFF * 
                     np.power(reynolds_number, const.NUSSELT_REYNOLDS_EXPONENT) / 
                     np.power(prandtl_number, const.NUSSELT_PRANDTL_EXPONENT))
    
    # Heat flux: Q = (Nu * k / L) * (T_air - T_ice)
    heat_flux = ((nusselt_number * const.AIR_THERMAL_CONDUCTIVITY / iceberg_length) * 
                 (air_temp - const.ICE_SURFACE_TEMPERATURE))
    
    # Convert heat flux to melt rate: m = Q / (ρ * L)
    melt_rate = heat_flux / (const.RHO_ICE * const.LATENT_HEAT_FUSION)
    
    # No melting if air temperature is below freezing
    if np.isscalar(is_freezing):
        if is_freezing:
            melt_rate = 0
    else:
        melt_rate = np.where(is_freezing, 0, melt_rate)
    
    return melt_rate  # m/s


def melt_buoyantwater(water_temp, salinity, method='cis', 
                      use_constant_tf=False, constant_tf=None):
    """
    Calculate buoyant (free) convection melt along vertical walls.
    
    Uses Bigg et al. (1997) / El-Tahan formulation.
    
    Parameters
    ----------
    water_temp : float or ndarray
        Water temperature in °C
    salinity : float or ndarray
        Salinity in PSU
    method : {'bigg', 'cis'}, optional
        Which formulation to use. Default is 'cis'.
    use_constant_tf : bool, optional
        If True, use constant thermal forcing
    constant_tf : float, optional
        Constant thermal forcing value
    
    Returns
    -------
    melt_rate : float or ndarray
        Melt rate in m/s
    
    References
    ----------
    Bigg et al. (1997), El-Tahan formulation
    """
    # Calculate freezing point temperature
    # T_f = -0.036 - 0.0499*S - 0.0001128*S²
    temp_freeze = (const.BIGG_FP_COEFF_1 + 
                   const.BIGG_FP_COEFF_2 * salinity + 
                   const.BIGG_FP_COEFF_3 * np.power(salinity, 2))
    
    # Adjust freezing point: T_fp = T_f * exp(-0.19 * (T - T_f))
    freezing_point = temp_freeze * np.exp(const.BIGG_FP_EXPONENT_COEFF * (water_temp - temp_freeze))
    
    # Calculate thermal forcing
    if method == 'bigg':
        if use_constant_tf:
            thermal_forcing = constant_tf
        else:
            thermal_forcing = water_temp
        
        # Melt rate: m = 7.62e-3 * dT + 1.3e-3 * dT² (m/day)
        melt_rate_per_day = (const.BUOYANT_MELT_LINEAR_COEFF * thermal_forcing + 
                            const.BUOYANT_MELT_QUADRATIC_COEFF_BIGG * np.power(thermal_forcing, 2))
    
    elif method == 'cis':
        if use_constant_tf:
            thermal_forcing = constant_tf
            melt_rate_per_day = (const.BUOYANT_MELT_LINEAR_COEFF * thermal_forcing + 
                                const.BUOYANT_MELT_QUADRATIC_COEFF_CIS * np.power(thermal_forcing, 2))
        else:
            thermal_forcing = water_temp - freezing_point
            melt_rate_per_day = (const.BUOYANT_MELT_LINEAR_COEFF * thermal_forcing + 
                                const.BUOYANT_MELT_QUADRATIC_COEFF_CIS * np.power(thermal_forcing, 2))
    
    # Convert from m/day to m/s
    melt_rate = melt_rate_per_day / const.SECONDS_PER_DAY
    
    return melt_rate


# The iceberg_melt function would be updated similarly, using constants like:
# - const.SECONDS_PER_DAY instead of 86400
# - const.RHO_ICE instead of 917
# - const.DENSITY_RATIO_ICE_TO_WATER instead of rho_i/1024
# - const.DEFAULT_TRANSFER_COEFFICIENT_FACTOR instead of factor=4
# - const.STABILITY_WIDTH_FACTOR instead of 0.7
# - const.DEFAULT_LENGTH_TO_WIDTH_RATIO instead of 1.62
# - const.WAVE_LENGTH_SURFACES, const.FORCED_WATER_LENGTH_SURFACES, etc.

# Example of how to update variable names in iceberg_melt:
"""
# OLD CODE:
dt = 86400
rho_i = 917
rat_i = rho_i/1024

# NEW CODE:
timestep_seconds = const.DEFAULT_TIMESTEP_SECONDS  # or const.SECONDS_PER_DAY
density_ratio = const.DENSITY_RATIO_ICE_TO_WATER
"""
