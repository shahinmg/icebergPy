"""
STEP-BY-STEP MIGRATION GUIDE
Integrating constants.py into your iceberg model

Follow these steps to update your code to use the constants module.
"""

# ==============================================================================
# STEP 1: Add constants.py to your project
# ==============================================================================

"""
File structure should be:
    iceberg_model/
        __init__.py
        constants.py         # ← ADD THIS NEW FILE
        iceberg.py
        melt.py
        simulation.py
"""

# Save the constants.py file I created to:
# ~/icebergPy/src/iceberg_model/constants.py


# ==============================================================================
# STEP 2: Update iceberg.py
# ==============================================================================

"""
Changes to make in iceberg.py:
"""

# 2a. ADD IMPORT at top (after other imports, around line 41)
# ─────────────────────────────────────────────────────────────
# ADD THIS LINE:
from . import constants as const

# 2b. UPDATE CLASS CONSTANTS (around line 98-102)
# ─────────────────────────────────────────────────────────────
# FIND THIS:
class Iceberg:
    """..."""
    # Physical constants (can be overridden for sensitivity studies)
    RHO_ICE = 917  # kg/m³ - density of ice
    RHO_WATER = 1024  # kg/m³ - density of seawater
    STABILITY_THRESHOLD = 0.92  # Wagner et al. 2017: W/H ratio for stability
    DEFAULT_LW_RATIO = 1.62  # Dowdeswell et al.: typical length-to-width ratio

# REPLACE WITH THIS:
class Iceberg:
    """..."""
    # Physical constants (imported from constants module)
    RHO_ICE = const.RHO_ICE
    RHO_WATER = const.RHO_SEAWATER
    STABILITY_THRESHOLD = const.STABILITY_THRESHOLD_WH
    DEFAULT_LW_RATIO = const.DEFAULT_LENGTH_TO_WIDTH_RATIO


# 2c. UPDATE keeldepth() method (around line 280)
# ─────────────────────────────────────────────────────────────
# FIND THESE LINES:
L_10 = np.round(L_val / 10) * 10
barker_mask = L_10 <= 160
hotzel_mask = L_10 > 160
if method == 'barker':
    return 2.91 * np.power(L_10, 0.71)
elif method == 'hotzel':
    return 3.78 * np.power(L_10, 0.63)
elif method == 'constant':
    return 0.7 * L_10

# REPLACE WITH:
L_10 = np.round(L_val / const.KEEL_DEPTH_ROUNDING_MULTIPLE) * const.KEEL_DEPTH_ROUNDING_MULTIPLE
barker_mask = L_10 <= const.BARKER_HOTZEL_THRESHOLD
hotzel_mask = L_10 > const.BARKER_HOTZEL_THRESHOLD
if method == 'barker':
    return const.BARKER_COEFFICIENT_A * np.power(L_10, const.BARKER_EXPONENT_B)
elif method == 'hotzel':
    return const.HOTZEL_COEFFICIENT_A * np.power(L_10, const.HOTZEL_EXPONENT_B)
elif method == 'constant':
    return const.CONSTANT_KEEL_RATIO * L_10

# And in the 'mean' method section:
keel_arr[barker_mask, 0] = const.BARKER_COEFFICIENT_A * np.power(L_10[barker_mask], const.BARKER_EXPONENT_B)
keel_arr[hotzel_mask, 0] = const.HOTZEL_COEFFICIENT_A * np.power(L_10[hotzel_mask], const.HOTZEL_EXPONENT_B)
keel_arr[:, 1] = const.BARKER_COEFFICIENT_A * np.power(L_10, const.BARKER_EXPONENT_B)
keel_arr[:, 2] = const.HOTZEL_COEFFICIENT_A * np.power(L_10, const.HOTZEL_EXPONENT_B)
keel_arr[:, 3] = const.CONSTANT_KEEL_RATIO * L_10


# 2d. UPDATE barker_carea() method
# ─────────────────────────────────────────────────────────────
# FIND THIS (around line 340):
def barker_carea(self, keel_depth, dz, LWratio=1.62, tabular=200, method='barker'):

# REPLACE WITH:
def barker_carea(self, keel_depth, dz, LWratio=None, tabular=None, method='barker'):
    """..."""
    # Use defaults from constants
    if LWratio is None:
        LWratio = const.DEFAULT_LENGTH_TO_WIDTH_RATIO
    if tabular is None:
        tabular = const.TABULAR_THRESHOLD_DEPTH

# FIND THIS (around line 420):
a_s = 28.194
b_s = -1420.2

# REPLACE WITH:
a_s = const.SAIL_AREA_COEFFICIENT_A
b_s = const.SAIL_AREA_COEFFICIENT_B

# FIND THIS (around line 470):
if L < 65:
    temps[L<65] = 0.077 * np.power(L[L<65],2)

# REPLACE WITH:
if L < const.SAIL_AREA_LENGTH_THRESHOLD:
    temps[L < const.SAIL_AREA_LENGTH_THRESHOLD] = (
        const.SAIL_AREA_QUADRATIC_COEFF * np.power(L[L < const.SAIL_AREA_LENGTH_THRESHOLD], 2)
    )

# FIND THIS (around line 490):
temps[K_gtab] = 0.1211 * L[K_gtab] * keel_depth[K_gtab]

# REPLACE WITH:
temps[K_gtab] = const.TABULAR_ICEBERG_COEFFICIENT * L[K_gtab] * keel_depth[K_gtab]


# 2e. UPDATE init_iceberg_size() method (around line 650)
# ─────────────────────────────────────────────────────────────
# FIND THIS:
rho_i = 917 #kg/m3
rat_i = rho_i/1024
waterline_width = self.length/1.62

# REPLACE WITH:
density_ratio = const.DENSITY_RATIO_ICE_TO_WATER
waterline_width = self.length / const.DEFAULT_LENGTH_TO_WIDTH_RATIO

# Then replace all instances of rat_i with density_ratio in this method


# ==============================================================================
# STEP 3: Update melt.py
# ==============================================================================

"""
Changes to make in melt.py:
"""

# 3a. ADD IMPORT at top
# ─────────────────────────────────────────────────────────────
# ADD THIS LINE (after numpy, scipy imports):
from . import constants as const


# 3b. UPDATE melt_wave() function
# ─────────────────────────────────────────────────────────────
# FIND THIS:
def melt_wave(windu, sst, sea_ice_conc):
    sea_state = 1.5 * np.sqrt(windu) + 0.1 * windu
    IceTerm = 1 + np.cos(np.power(sea_ice_conc,3) * np.pi)
    melt = (1/12) * sea_state * IceTerm * (sst + 2)
    melt = melt / 86400

# REPLACE WITH:
def melt_wave(wind_speed, sea_surface_temp, sea_ice_concentration):
    sea_state = (const.WAVE_HEIGHT_WIND_COEFF_1 * np.sqrt(wind_speed) + 
                 const.WAVE_HEIGHT_WIND_COEFF_2 * wind_speed)
    ice_term = 1 + np.cos(np.power(sea_ice_concentration, 3) * np.pi)
    melt = ((1 / const.WAVE_MELT_DIVISOR) * sea_state * ice_term * 
            (sea_surface_temp + const.WAVE_MELT_TEMP_OFFSET))
    melt = melt / const.SECONDS_PER_DAY


# 3c. UPDATE melt_solar() function
# ─────────────────────────────────────────────────────────────
# FIND THIS:
def melt_solar(solar_rad):
    latent_heat = 3.33e5
    rho_i = 917
    albedo = 0.7
    absorbed = 1 - albedo
    melt = absorbed * solar_rad / (rho_i * latent_heat)

# REPLACE WITH:
def melt_solar(solar_radiation):
    absorbed_energy = const.SOLAR_ABSORPTION_FRACTION * solar_radiation
    melt = absorbed_energy / (const.RHO_ICE * const.LATENT_HEAT_FUSION)


# 3d. UPDATE melt_forcedwater() function
# ─────────────────────────────────────────────────────────────
# FIND THIS:
def melt_forcedwater(temp_far, salinity_far, pressure_base, U_rel, factor, ...):
    a = -5.73e-2
    b = 8.32e-2
    c = -7.61e-4
    GT = 1.1e-3 * factor
    GS = 3.1e-5 * factor
    L = 3.35e5
    cw = 3974
    ci = 2009
    DT = 15

# REPLACE WITH:
def melt_forcedwater(temp_far, salinity_far, pressure_base, velocity_relative, 
                     factor=None, ...):
    if factor is None:
        factor = const.DEFAULT_TRANSFER_COEFFICIENT_FACTOR
    
    heat_transfer = const.HEAT_TRANSFER_COEFFICIENT_GT * factor
    salt_transfer = const.SALT_TRANSFER_COEFFICIENT_GS * factor
    
    # Freezing point
    freezing_point = (const.FREEZING_POINT_SALINITY_COEFF * salinity_far + 
                     const.FREEZING_POINT_CONSTANT + 
                     const.FREEZING_POINT_PRESSURE_COEFF * pressure_base)
    
    A = ((const.LATENT_HEAT_FUSION + 
          const.TEMPERATURE_DIFFERENCE_CORE_SURFACE * const.SPECIFIC_HEAT_ICE) / 
         (velocity_relative * heat_transfer * const.SPECIFIC_HEAT_WATER))


# 3e. UPDATE melt_forcedair() function
# ─────────────────────────────────────────────────────────────
# FIND THIS:
def melt_forcedair(T_air, U_rel, L):
    T_ice = -4
    Li = 3.33e5
    rho_i = 917
    air_viscosity = 1.46e-5
    air_diffusivity = 2.16e-5
    air_conductivity = 0.0249

# REPLACE WITH:
def melt_forcedair(air_temp, air_velocity_relative, iceberg_length):
    prandtl_number = const.PRANDTL_NUMBER
    reynolds_number = (np.abs(air_velocity_relative) * iceberg_length / 
                      const.AIR_KINEMATIC_VISCOSITY)
    
    nusselt_number = (const.NUSSELT_COEFF * 
                     np.power(reynolds_number, const.NUSSELT_REYNOLDS_EXPONENT) / 
                     np.power(prandtl_number, const.NUSSELT_PRANDTL_EXPONENT))
    
    heat_flux = ((nusselt_number * const.AIR_THERMAL_CONDUCTIVITY / iceberg_length) * 
                 (air_temp - const.ICE_SURFACE_TEMPERATURE))
    
    melt_rate = heat_flux / (const.RHO_ICE * const.LATENT_HEAT_FUSION)


# 3f. UPDATE melt_buoyantwater() function
# ─────────────────────────────────────────────────────────────
# FIND THIS:
def melt_buoyantwater(T_w, S_w, method, ...):
    Tf = -0.036 - (0.0499 * S_w) - (0.0001128 * np.power(S_w,2))
    Tfp = Tf * np.exp(-0.19 * (T_w - Tf))
    # ...
    mday = 7.62e-3 * dT + 1.29e-3 * np.power(dT,2)
    melt = mday / 86400

# REPLACE WITH:
def melt_buoyantwater(water_temp, salinity, method='cis', ...):
    temp_freeze = (const.BIGG_FP_COEFF_1 + 
                   const.BIGG_FP_COEFF_2 * salinity + 
                   const.BIGG_FP_COEFF_3 * np.power(salinity, 2))
    freezing_point = temp_freeze * np.exp(const.BIGG_FP_EXPONENT_COEFF * (water_temp - temp_freeze))
    # ...
    melt_rate_per_day = (const.BUOYANT_MELT_LINEAR_COEFF * thermal_forcing + 
                        const.BUOYANT_MELT_QUADRATIC_COEFF_CIS * np.power(thermal_forcing, 2))
    melt_rate = melt_rate_per_day / const.SECONDS_PER_DAY


# 3g. UPDATE iceberg_melt() function
# ─────────────────────────────────────────────────────────────
# This is the big one - update throughout:

# FIND:           REPLACE WITH:
# dt = 86400      timestep_seconds = const.SECONDS_PER_DAY  (or DEFAULT_TIMESTEP_SECONDS)
# rho_i = 917     (use const.RHO_ICE directly)
# rat_i = ...     density_ratio = const.DENSITY_RATIO_ICE_TO_WATER
# 1.62            const.DEFAULT_LENGTH_TO_WIDTH_RATIO
# 0.92            const.STABILITY_THRESHOLD_WH
# 0.7             const.STABILITY_WIDTH_FACTOR
# factor=4        const.DEFAULT_TRANSFER_COEFFICIENT_FACTOR


# ==============================================================================
# STEP 4: Test Your Changes
# ==============================================================================

"""
After making changes, test that everything still works:
"""

# Test 1: Can you import?
from iceberg_model import Iceberg
from iceberg_model import constants as const

# Test 2: Do constants match?
berg = Iceberg(length=200, dz=5)
assert berg.RHO_ICE == const.RHO_ICE
assert berg.STABILITY_THRESHOLD == const.STABILITY_THRESHOLD_WH

# Test 3: Does geometry calculation work?
geometry = berg.init_iceberg_size()
print(f"Keel: {geometry.keel.values:.1f} m")
print(f"Volume: {geometry.totalV.values:.1e} m³")

# Test 4: Can you validate constants?
const.validate_constants()


# ==============================================================================
# STEP 5: Update Function Signatures (Optional but Recommended)
# ==============================================================================

"""
Consider renaming parameters to be more descriptive:
"""

# OLD:
# def melt_wave(windu, sst, sea_ice_conc):
# def melt_forcedwater(temp_far, salinity_far, pressure_base, U_rel, factor, ...):

# NEW (more readable):
# def melt_wave(wind_speed, sea_surface_temp, sea_ice_concentration):
# def melt_forcedwater(temp_far, salinity_far, pressure_base, velocity_relative, factor=None, ...):


print("""
MIGRATION COMPLETE! 🎉

Your code now uses the constants module for all physical and empirical parameters.

Benefits:
✅ All constants in one place
✅ Easy to modify for sensitivity studies
✅ Better documentation
✅ Reduced magic numbers
✅ More maintainable code
""")
