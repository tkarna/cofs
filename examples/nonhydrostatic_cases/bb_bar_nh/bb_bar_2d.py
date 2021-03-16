"""
Wave over submerged bar test case
=================================

A regular wave inlets from left boundary with amplitude of 1 cm and period of 2.02 s.
Experimental setup [1].

This example represents wave shoaling after interaction with uneven bottom.

[1] Beji, S., Battjes, J.A. (1994). Numerical simulation of nonlinear
    wave propagation over a bar. Coastal Engineering 23, 1–16.
    https://doi.org/10.1016/0378-3839(94)90012-4
"""
from thetis import *

lx = 35.0
ly = 1
nx = 700
ny = 1
mesh = RectangleMesh(nx, ny, lx, ly)
outputdir = 'outputs_bbbar_2d'

# --- bathymetry ---
P1_2d = FunctionSpace(mesh, 'CG', 1)
bathymetry_2d = Function(P1_2d, name='Bathymetry')


def get_bathymetry():
    mesh = bathymetry_2d.ufl_domain()
    x_vector = mesh.coordinates.dat.data[:, 0]
    b_vector = bathymetry_2d.dat.data
    assert x_vector.shape[0] == b_vector.shape[0]
    for i, xy in enumerate(x_vector):
        s = 25.
        if xy <= 6.:
            b_vector[i] = 0.4
        elif xy <= 12.:
            b_vector[i] = -0.05*xy + 0.7
        elif xy <= 14.:
            b_vector[i] = 0.1
        elif xy <= 17.:
            b_vector[i] = 0.1*xy - 1.3
        elif xy <= 19.:
            b_vector[i] = 0.4
        elif xy <= s:
            b_vector[i] = -0.04*xy + 1.16
        else:
            b_vector[i] = -0.04*s + 1.16
    return bathymetry_2d


bathymetry_2d = get_bathymetry()

# set time step, export interval and run duration
dt = 0.01
t_export = 0.1
t_end = 40

# choose if using non-hydrostatic model
solve_nonhydrostatic_pressure = True

# --- create solver ---
solver_obj = solver2d.FlowSolver2d(mesh, bathymetry_2d)
options = solver_obj.options
options.element_family = 'dg-dg'
options.polynomial_degree = 1
# time stepper
options.timestepper_type = 'DIRK33'
if hasattr(options.timestepper_options, 'use_automatic_timestep'):
    options.timestepper_options.use_automatic_timestep = True
options.timestep = dt
options.simulation_export_time = t_export
options.simulation_end_time = t_end
# output
options.output_directory = outputdir
options.fields_to_export = ['uv_2d', 'elev_2d', 'q_2d']
options.fields_to_export_hdf5 = ['uv_2d', 'elev_2d', 'q_2d']
# non-hydrostatic
if solve_nonhydrostatic_pressure:
    options_nh = options.nh_model_options
    options_nh.solve_nonhydrostatic_pressure = solve_nonhydrostatic_pressure
    options_nh.q_degree = 2
    options_nh.update_free_surface = True
    options_nh.free_surface_timestepper_type = 'CrankNicolson'

# create equations
solver_obj.create_equations()

# detectors
xy = [[10.5, 0.], [12.5, 0.], [13.5, 0.], [14.5, 0.],
      [15.69999999, 0.], [17.3, 0.], [19.0, 0.], [21.0, 0.]]
cb = DetectorsCallback(solver_obj, xy, ['elev_2d'], name='gauges', append_to_log=False)
solver_obj.add_callback(cb)

# input wave
pi = 4*np.arctan(1.)
eta_amp = 0.01
period = 2.02


def get_inputelevation(t):
    return eta_amp*np.sin(2*pi/period*t)


# boundary condition
t = 0.
H_2d = solver_obj.function_spaces.H_2d
ele_bc = Constant(0.)
inflow_tag = 1
solver_obj.bnd_functions['shallow_water'] = {inflow_tag: {'elev': ele_bc}}


# time updated
def update_forcings(t_new):
    """Callback function that updates all time dependent forcing fields
    for the 2d mode"""
    ele_bc.assign(get_inputelevation(t_new))


# --- initial conditions, create all function spaces, equations etc ---
solver_obj.assign_initial_conditions()
solver_obj.iterate(update_forcings=update_forcings)
