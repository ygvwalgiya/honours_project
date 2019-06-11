# %% [markdown]
# # Jason's honours project code
# ## Motor optimisation part

# %% imports
import numpy as np

# %% [markdown]
# | Name | Units | Description |
# |------|-------|-------------|
# | r_tube_In | m | Inner radius of sillicone tube |
# | r_tube_Out | m | Outer radius of sillicone tube |
# | l_core | m | Length of motor core |
# | Br_core | T | Residual magnetic field stength of core |
# | r_core | m | Radius of motor core |
# | rho_wire | S/m | Resistivity of wire |
# | V_max | V | Max voltage output of power supply |
# | I_max | A | Max current output of power supply |
# | l_max_tube | m | How much tube I have |
# | l_connector | m | How much magnetic connector at the end of the magnet |
# | l_plunger | m | Length of plunger connector |
# | thickness_shell | m | Thickness of iron shell |
# | distance_rg2 | m | Distance between plunger and shell |

# %% definition cell
r_tube_in = 0.25e-3
r_tube_out = 1.05e-3
l_core = 80e-3
Br_core = 1.2
r_core = 2.175e-3
rho_wire = 2.3e6
V_max = 12
I_max = 2
l_max_tube = 4.5
l_connector = 0.0042
l_plunger = 0.005
thickness_shell = 0.002
distance_rg2 = 0.001

# %% [markdown]
# |Name|Description|
# |----|-----------|
# |turns_in_field|Turns of wire in field
# |turns_total|Turns of wire total

# %% relative constants
turns_in_field = l_connector/(r_tube_out*2)

turns_total = np.floor((l_core+l_connector)/(2*r_tube_out))

# %% [markdown]
# | Name | Units | Description |
# |------|-------|-------------|
# | A_cross_magnet | m^2 | Cross sectional area of the magnet |
# | flux_rg0 | Wb | Magnetic flux of magnet with magnet only |
# | rM_m | H^-1 | Reluctance across magnet |
# | emmf | #NA# | Electromagneticmotive force|

# %% Electromagneticmotive force
A_cross_magnet = np.pi*r_core**2

flux_rg0 = A_cross_magnet * Br_core

# H^-1 Longer the magnet, larger the Rm?
rM_m = l_core/A_cross_magnet
emmf = flux_rg0 * rM_m

# %% [markdown]
# | Name | Units | Description |
# |------|-------|-------------|
# | layer | #NA# | The layer of wire in this calculation |
# | r_shell_in | m | Inner shell diameter |
# | r_shell_out | m | Outer shell diameter |
# | r_plunger | m | Plunger diameter |
# | rM_g1 | H^-1 | Gap1 (coil driving) reluctance |
# | rM_g2 | H^-1 | Gap2 (plunger) reluctance |
# | rM_ttl | H^-1 | Total reluctance |
# | flux_w_gap | Wb | Magnetic flux in circuit with gaps |
# | r_at_wire | m | radius of wire layer |
# | l_in_field_this | m | Length of wire in magnetic field in this layer |
# | l_this | m | Length of wire in this layer |
# | R_this | Ohm | Wire resistance this layer |
# | V_this | V | Voltage required this layer |
# | A_field_action | m^2 | Area of wire the field acts on |
# | field_over_gap | m | Magnetic field over Gap1 (coil driving) |
# | force | m | Force generated |

# %% Motor Setup
layer = 1
r_shell_in = r_core+r_tube_out*layer
r_shell_out = r_shell_in + thickness_shell
r_plunger = r_shell_in - distance_rg2
# Rg1 increases as gap increases? This isn't right will revisit
rM_g1 = np.log(r_shell_in/r_core)/(2*np.pi*l_connector)
rM_g2 = np.log(r_shell_in/r_plunger)/(2*np.pi*l_plunger)
rM_ttl = rM_m+rM_g1+rM_g2
flux_w_gap = emmf / rM_ttl
r_at_wire = r_shell_out - r_tube_out/2
l_in_field_this = np.pi*(r_core+layer*r_tube_out*2)*2*turns_in_field
l_this = np.pi*(r_core+layer*r_tube_out*2)*2*turns_total
R_this = l_this/(np.pi*r_tube_in**2)/rho_wire
V_this = R_this * I_max
A_field_action = 2*np.pi*r_at_wire*l_connector
field_over_gap = flux_w_gap/A_field_action
force = field_over_gap*I_max*l_in_field_this
