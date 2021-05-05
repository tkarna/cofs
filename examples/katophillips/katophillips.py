"""
Wind-driver entrainment test case.

Based on Kato-Phillips laboratory tests.

Initial water column is stratified with a constant density gradient.
Circulation is driven by constant wind stress at the surface. Wind-induced
mixing begins to destroy stratification, creating a well-mixed surfacelayer.
The depth of the surface layer follows an empirical relation versus time.

Surface friction velocity is u_s = 0.01 m s-1.
Initially buoyancy frequency is constant N = 0.01 s-1.


[1] http://www.gotm.net/
[2] Karna et al. (2012). Coupling of a discontinuous Galerkin finite element
    marine model with a finite difference turbulence closure model.
    Ocean Modelling, 47:55-64.
    http://dx.doi.org/10.1016/j.ocemod.2012.01.001
"""
from thetis import *

physical_constants['rho0'] = 1027.0  # NOTE must match empirical setup
depth = 50.0


class MaxNuCallback(DiagnosticCallback):
    """
    Calculates max viscosity
    """
    name = 'maxnu'
    variable_names = ['nu']

    def __call__(self):
        return [self.solver_obj.fields.eddy_visc_3d.dat.data.max()]

    def message_str(self, *args):
        return 'max viscosity: {:12.5f}'.format(*args)


def katophillips_test(layers=25, gls_closure='k-omega',
                      stability_func='Canuto B',
                      iterate=True, load_export_ix=None, **model_options):
    outputdir = 'outputs'
    # set mesh resolution
    dx = 2500.0

    # generate unit mesh and transform its coords
    nx = 3  # nb elements in flow direction
    lx = nx*dx
    ny = 3  # nb elements in cross direction
    ly = ny*dx
    mesh2d = PeriodicRectangleMesh(nx, ny, lx, ly, direction='both', reorder=True)
    # move mesh, center to (0,0)
    mesh2d.coordinates.dat.data[:, 0] -= lx/2
    mesh2d.coordinates.dat.data[:, 1] -= ly/2

    dt = 60.0
    t_end = 30 * 3600.0
    t_export = 5*60.0
    u_mag = 1.0

    if os.getenv('THETIS_REGRESSION_TEST') is not None:
        t_end = t_export

    # bathymetry
    P1_2d = get_functionspace(mesh2d, 'CG', 1)
    bathymetry2d = Function(P1_2d, name='Bathymetry')
    bathymetry2d.assign(depth)

    wind_stress_x = 0.1027  # Pa
    wind_stress_2d = Constant((wind_stress_x, 0))

    # create solver
    solver_obj = solver.FlowSolver(mesh2d, bathymetry2d, layers)
    options = solver_obj.options
    options.solve_salinity = True
    options.solve_temperature = False
    options.constant_temperature = Constant(10.0)
    options.use_implicit_vertical_diffusion = True
    options.use_bottom_friction = False
    options.use_turbulence = True
    options.use_ale_moving_mesh = False
    options.use_baroclinic_formulation = True
    options.use_limiter_for_tracers = True
    options.vertical_viscosity = Constant(1.3e-6)  # background value
    options.vertical_diffusivity = Constant(1.4e-7)  # background value
    options.wind_stress = wind_stress_2d
    options.simulation_export_time = t_export
    options.timestepper_type = 'SSPRK22'
    options.timestepper_options.use_automatic_timestep = False
    options.timestep = dt
    options.simulation_end_time = t_end
    options.output_directory = outputdir
    options.horizontal_velocity_scale = Constant(u_mag)
    options.check_salinity_overshoot = False
    options.fields_to_export = ['uv_2d', 'elev_2d', 'uv_3d',
                                'w_3d', 'w_mesh_3d', 'salt_3d',
                                'baroc_head_3d', 'uv_dav_2d', 'eddy_visc_3d',
                                'shear_freq_3d', 'buoy_freq_3d',
                                'tke_3d', 'psi_3d', 'eps_3d', 'len_3d', ]
    options.fields_to_export_hdf5 = ['uv_2d', 'elev_2d', 'uv_3d', 'salt_3d',
                                     'eddy_visc_3d', 'eddy_diff_3d',
                                     'shear_freq_3d', 'buoy_freq_3d',
                                     'tke_3d', 'psi_3d', 'eps_3d', 'len_3d', ]
    options.update(model_options)
    turb_options = options.turbulence_model_options
    turb_options.apply_defaults(gls_closure)
    turb_options.stability_function_name = stability_func
    layer_str = 'nz{:}'.format(layers)
    odir = '_'.join([outputdir, layer_str,
                     turb_options.closure_name.replace(' ', '-'),
                     turb_options.stability_function_name.replace(' ', '-')])
    options.output_directory = odir
    print_output('Exporting to ' + options.output_directory)

    solver_obj.create_function_spaces()

    # initial conditions
    N0 = 0.01
    # N = sqrt(-g/rho0 drho/dz)
    # drho/dz = -N0**2 * rho0/g
    rho0 = physical_constants['rho0']
    g = physical_constants['g_grav']
    rho_grad = -N0**2 * rho0 / g
    beta = 0.7865  # haline contraction coefficient [kg m-3 psu-1]
    salt_grad = rho_grad/beta
    salt_init3d = Function(solver_obj.function_spaces.H,
                           name='initial salinity')
    x, y, z = SpatialCoordinate(solver_obj.mesh)
    salt_init_expr = salt_grad*z
    salt_init3d.interpolate(salt_init_expr)

    if iterate:
        solver_obj.add_callback(MaxNuCallback(solver_obj))
        solver_obj.assign_initial_conditions(salt=salt_init3d)

        solver_obj.iterate()
    elif load_export_ix is not None:
        print_output('Loading state: {:}'.format(load_export_ix))
        solver_obj.load_state(load_export_ix)

    return solver_obj


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='Run Kato-Phillips test case',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('-n', '--nlevels', type=int, default=50,
                        help='number of vertical levels')
    parser.add_argument('-m', '--model', default='k-epsilon',
                        choices=['k-epsilon', 'k-omega', 'gls'],
                        help='GLS turbulence closure model')
    parser.add_argument('-s', '--stability-func', default='Canuto-A',
                        choices=['Canuto-A', 'Canuto-B', 'Cheng'],
                        help='Stability function name')

    args = parser.parse_args()
    model = args.model
    if model == 'gls':
        model = 'Generic Length Scale'
    katophillips_test(
        layers=args.nlevels,
        gls_closure=model,
        stability_func=args.stability_func.replace('-', ' '))
