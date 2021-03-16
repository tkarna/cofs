"""
Non-hydrostatic standing wave
=============================

Test for solving dispersive surface standing waves based upon using wave equation
with non-hydrostatic pressure.

Initial condition for elevation corresponds to a standing wave. The numerical setup
corresponds to that considered in the test cases section of [1].

[1] Pan, Wei, Stephan C. Kramer, and Matthew D. Piggott. "Multi-layer non-hydrostatic
    free surface modelling using the discontinuous Galerkin method." Ocean Modelling
    134 (2019): 68-83. DOI: https://doi.org/10.1016/j.ocemod.2019.01.003
"""
from thetis import *
import pytest
import math
import h5py


@pytest.mark.parametrize("timesteps,max_rel_err", [
    (10, 1.6e-2), (20, 4e-3), (40, 1e-3)])
@pytest.mark.parametrize("timestepper", [
    # implicit
    'CrankNicolson', 'PressureProjectionPicard',
    'SSPIMEX', 'DIRK22', 'DIRK33',
    # explicit
    'SSPRK33', 'ForwardEuler'])
def test_nh_standing_wave(timesteps, max_rel_err, timestepper, tmpdir,
                          do_export=False, solve_nonhydrostatic_pressure=True):

    lx = 20.
    ly = 4.
    nx = 20
    mesh2d = RectangleMesh(nx, 1, lx, ly)

    depth = 8.
    elev_amp = 0.1
    g_grav = physical_constants['g_grav'].dat.data[0]

    c = math.sqrt(g_grav*lx/(2*pi)*tanh(2*pi*depth/lx))
    period = lx/c
    dt = period/timesteps
    t_end = period

    x = SpatialCoordinate(mesh2d)
    elev_init = elev_amp*cos(2*pi*x[0]/lx)

    # bathymetry
    P1_2d = FunctionSpace(mesh2d, 'CG', 1)
    bathymetry_2d = Function(P1_2d, name="Bathymetry")
    bathymetry_2d.assign(depth)

    # --- create solver ---
    solver_obj = solver2d.FlowSolver2d(mesh2d, bathymetry_2d)
    options = solver_obj.options
    if timestepper == 'PressureProjectionPicard':
        options.element_family = 'dg-cg'
    else:
        options.element_family = 'dg-dg'
    options.polynomial_degree = 1
    # time stepper
    options.timestepper_type = timestepper
    if hasattr(options.timestepper_options, 'use_automatic_timestep'):
        options.timestepper_options.use_automatic_timestep = False
        timesteps *= 40
        dt = period/timesteps
    options.timestep = dt
    solver_obj.options.simulation_export_time = dt
    solver_obj.options.simulation_end_time = t_end
    # output
    solver_obj.options.no_exports = not do_export
    solver_obj.options.output_directory = str(tmpdir)
    # non-hydrostatic
    if solve_nonhydrostatic_pressure:
        options_nh = options.nh_model_options
        options_nh.solve_nonhydrostatic_pressure = solve_nonhydrostatic_pressure
        options_nh.q_degree = 2
        options_nh.update_free_surface = True
        options_nh.free_surface_timestepper_type = 'CrankNicolson'
        if hasattr(options.timestepper_options, 'use_automatic_timestep'):
            # use the same explicit timestepper, but CrankNicolson is ok
            options_nh.free_surface_timestepper_type = timestepper

    # boundary conditions
    solver_obj.bnd_functions['shallow_water'] = {}

    # create equations
    solver_obj.create_equations()
    solver_obj.assign_initial_conditions(elev=elev_init)

    # detectors
    xy = [[1*lx/8., ly/2.], [4*lx/8., ly/2.], [7*lx/8., ly/2.]]
    # first set of detectors
    cb = DetectorsCallback(solver_obj, xy, ['elev_2d'], name='gauges', append_to_log=True)
    solver_obj.add_callback(cb)

    solver_obj.iterate()

    # analytical elevation field at the end of simulation time
    anal_field = elev_amp*cos(2*pi*x[0]/lx)*cos(math.sqrt(2*pi*g_grav/lx*tanh(2*pi*depth/lx))*t_end)
    rel_err = errornorm(anal_field, solver_obj.fields.elev_2d)/math.sqrt(lx*ly)
    print_output(rel_err)
    assert(rel_err < max_rel_err)
    print_output("PASSED")

    with h5py.File(str(tmpdir) + '/diagnostic_gauges.hdf5', 'r') as df:
        assert all(df.attrs['field_dims'][:] == [1, ])
        trange = np.arange(timesteps+1)*dt
        np.testing.assert_almost_equal(df['time'][:, 0], trange)
        for i in range(len(xy)):
            # location of detectors
            x_loc = xy[i][0]
            # analytical time series of elevation at specific locations
            anal_ts = elev_amp*np.cos(2*pi*x_loc/lx)*np.cos(math.sqrt(2*pi*g_grav/lx*tanh(2*pi*depth/lx))*trange)
            np.testing.assert_allclose(df['detector'+str(i)][:][:, 0], anal_ts, atol=4e-2, rtol=4e-1)


if __name__ == '__main__':
    test_nh_standing_wave(do_export=True)
