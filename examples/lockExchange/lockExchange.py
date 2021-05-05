"""
Lock Exchange Test case
=======================

Solves hydrostatic flow in a closed rectangular channel.

Dianeutral mixing depends on mesh Reynolds number [1]
Re_h = U dx / nu
U = 0.5 m/s characteristic velocity ~ 0.5*sqrt(g_h drho/rho_0)
dx = horizontal mesh size
nu = background viscosity


Smagorinsky factor should be C_s = 1/sqrt(Re_h)

Mesh resolutions:
- ilicak [1]:  dx =  500 m,  20 layers
COMODO lock exchange benchmark [2]:
- coarse:      dx = 2000 m,  10 layers
- coarse2 (*): dx = 1000 m,  20 layers
- medium:      dx =  500 m,  40 layers
- medium2 (*): dx =  250 m,  80 layers
- fine:        dx =  125 m, 160 layers
(*) not part of the original benchmark

[1] Ilicak et al. (2012). Spurious dianeutral mixing and the role of
    momentum closure. Ocean Modelling, 45-46(0):37-58.
    http://dx.doi.org/10.1016/j.ocemod.2011.10.003
[2] COMODO Lock Exchange test.
    http://indi.imag.fr/wordpress/?page_id=446
[3] Petersen et al. (2015). Evaluation of the arbitrary Lagrangian-Eulerian
    vertical coordinate method in the MPAS-Ocean model. Ocean Modelling,
    86:93-113.
    http://dx.doi.org/10.1016/j.ocemod.2014.12.004
"""
from thetis import *
from diagnostics import *
from plotting import *


def run_lockexchange(reso_str='coarse', poly_order=1, element_family='dg-dg',
                     reynolds_number=1.0, use_limiter=True, dt=None,
                     viscosity='const', laxfriedrichs_vel=0.0,
                     laxfriedrichs_trc=0.0,
                     elem_type='tri',
                     load_export_ix=None, iterate=True, **custom_options):
    """
    Runs lock exchange problem with a bunch of user defined options.
    """
    comm = COMM_WORLD

    if laxfriedrichs_vel is None:
        laxfriedrichs_vel = 0.0
    if laxfriedrichs_trc is None:
        laxfriedrichs_trc = 0.0

    depth = 20.0
    refinement = {'huge': 0.6, 'coarse': 1, 'coarse2': 2, 'medium': 4,
                  'medium2': 8, 'fine': 16, 'ilicak': 4}
    # set mesh resolution
    if '-' in reso_str:
        words = reso_str.split('-')
        delta_x, delta_z = [float(f) for f in words]
        layers = int(np.ceil(depth/delta_z))
    else:
        delta_x = 2000.0/refinement[reso_str]
        layers = int(round(10*refinement[reso_str]))
        if reso_str == 'ilicak':
            layers = 20

    # generate unit mesh and transform its coords
    x_max = 32.0e3
    x_min = -32.0e3
    n_x = (x_max - x_min)/delta_x
    mesh2d = UnitSquareMesh(int(n_x), 2, quadrilateral=(elem_type == 'quad'))
    coords = mesh2d.coordinates
    # x in [x_min, x_max], y in [-dx, dx]
    coords.dat.data[:, 0] = coords.dat.data[:, 0]*(x_max - x_min) + x_min
    coords.dat.data[:, 1] = coords.dat.data[:, 1]*2*delta_x - delta_x

    # temperature and salinity, for linear eq. of state (from Petersen, 2015)
    temp_left = 5.0
    temp_right = 30.0
    salt_const = 35.0
    rho_0 = 1000.0
    physical_constants['rho0'].assign(rho_0)

    # compute horizontal viscosity
    uscale = 0.5
    nu_scale = uscale * delta_x / reynolds_number

    if reynolds_number < 0:
        reynolds_number = float("inf")
        nu_scale = 0.0

    u_max = 1.0
    w_max = 1.2e-2

    t_end = 25 * 3600
    t_export = 15*60.0

    if os.getenv('THETIS_REGRESSION_TEST') is not None:
        t_end = t_export

    lim_str = '_lim' if use_limiter else ''
    options_str = '_'.join([reso_str,
                            element_family,
                            elem_type,
                            'p{:}'.format(poly_order),
                            'visc-{:}'.format(viscosity),
                            'Re{:}'.format(reynolds_number),
                            'lf-vel{:.1f}'.format(laxfriedrichs_vel),
                            'lf-trc{:.1f}'.format(laxfriedrichs_trc),
                            ]) + lim_str
    outputdir = 'outputs_' + options_str

    # bathymetry
    p1_2d = get_functionspace(mesh2d, 'CG', 1)
    bathymetry_2d = Function(p1_2d, name='Bathymetry')
    bathymetry_2d.assign(depth)

    # create solver
    solver_obj = solver.FlowSolver(mesh2d, bathymetry_2d, layers)
    options = solver_obj.options
    options.polynomial_degree = poly_order
    options.element_family = element_family
    options.timestepper_type = 'SSPRK22'
    options.solve_salinity = False
    options.constant_salinity = Constant(salt_const)
    options.solve_temperature = True
    options.use_implicit_vertical_diffusion = False
    options.use_bottom_friction = False
    options.use_ale_moving_mesh = True
    options.use_baroclinic_formulation = True
    options.use_lax_friedrichs_velocity = laxfriedrichs_vel > 0.0
    options.use_lax_friedrichs_tracer = laxfriedrichs_trc > 0.0
    options.lax_friedrichs_velocity_scaling_factor = Constant(laxfriedrichs_vel)
    options.lax_friedrichs_tracer_scaling_factor = Constant(laxfriedrichs_trc)
    options.use_limiter_for_tracers = use_limiter
    options.use_limiter_for_velocity = use_limiter
    # To keep const grid Re_h, viscosity scales with grid: nu = U dx / Re_h
    if viscosity == 'smag':
        options.use_smagorinsky_viscosity = True
        options.smagorinsky_coefficient = Constant(1.0/np.sqrt(reynolds_number))
    elif viscosity == 'const':
        options.horizontal_viscosity = Constant(nu_scale)
    else:
        raise Exception('Unknow viscosity type {:}'.format(viscosity))
    options.vertical_viscosity = Constant(1e-4)
    options.horizontal_diffusivity = None
    options.horizontal_viscosity_scale = Constant(nu_scale)
    options.horizontal_velocity_scale = Constant(u_max)
    options.vertical_velocity_scale = Constant(w_max)
    if dt is not None:
        options.timestepper_options.use_automatic_timestep = False
        options.timestep = dt
    options.simulation_export_time = t_export
    options.simulation_end_time = t_end
    options.output_directory = outputdir
    options.check_volume_conservation_2d = True
    options.check_volume_conservation_3d = True
    options.check_temperature_conservation = True
    options.check_temperature_overshoot = True
    options.fields_to_export = ['uv_2d', 'elev_2d', 'uv_3d',
                                'w_3d', 'w_mesh_3d', 'temp_3d', 'density_3d',
                                'uv_dav_2d', 'uv_dav_3d', 'baroc_head_3d',
                                'smag_visc_3d']
    options.fields_to_export_hdf5 = list(options.fields_to_export)
    options.equation_of_state_type = 'linear'
    options.equation_of_state_options.rho_ref = rho_0
    options.equation_of_state_options.s_ref = 35.0
    options.equation_of_state_options.th_ref = 5.0
    options.equation_of_state_options.alpha = 0.2
    options.equation_of_state_options.beta = 0.0
    options.update(custom_options)

    if comm.size == 1:
        solver_obj.add_callback(RPECalculator(solver_obj))
        solver_obj.add_callback(FrontLocationCalculator(solver_obj))
        # solver_obj.add_callback(PlotCallback(solver_obj, append_to_log=False))

    solver_obj.create_equations()

    print_output('Running lock exchange problem with options:')
    print_output('Resolution: {:}'.format(reso_str))
    print_output('Reynolds number: {:}'.format(reynolds_number))
    print_output('Use slope limiters: {:}'.format(use_limiter))
    print_output('Horizontal viscosity: {:}'.format(nu_scale))
    print_output('Lax-Friedrichs factor vel: {:}'.format(laxfriedrichs_vel))
    print_output('Lax-Friedrichs factor trc: {:}'.format(laxfriedrichs_trc))
    print_output('Exporting to {:}'.format(outputdir))

    temp_init3d = Function(solver_obj.function_spaces.H, name='initial temperature')
    x, y, z = SpatialCoordinate(solver_obj.mesh)
    # vertical barrier
    # temp_init3d.interpolate(conditional(x > 0.0, temp_right, temp_left))
    # smooth condition
    sigma = 10.0
    temp_init3d.interpolate(
        temp_left - (temp_left - temp_right)*0.5*(tanh(x/sigma) + 1.0)
    )

    if load_export_ix is None:
        solver_obj.assign_initial_conditions(temp=temp_init3d)
    else:
        assert isinstance(load_export_ix, int)
        solver_obj.load_state(load_export_ix)

    if iterate:
        solver_obj.iterate()

    return solver_obj


def get_argparser():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--reso_str', type=str,
                        help='mesh resolution string. A named mesh  or "dx-dz" string',
                        default='coarse')
    parser.add_argument('--no-limiter', action='store_false', dest='use_limiter',
                        help='do not use slope limiter for tracers')
    parser.add_argument('-p', '--poly_order', type=int, default=1,
                        help='order of finite element space')
    parser.add_argument('-f', '--element-family', type=str,
                        help='finite element family', default='dg-dg')
    parser.add_argument('-re', '--reynolds-number', type=float, default=1.0,
                        help='mesh Reynolds number for Smagorinsky scheme')
    parser.add_argument('-dt', '--dt', type=float,
                        help='force value for 3D time step')
    parser.add_argument('-visc', '--viscosity', type=str,
                        help='Type of horizontal viscosity',
                        default='const',
                        choices=['const', 'smag'])
    parser.add_argument('-lf-trc', '--laxfriedrichs-trc', type=float,
                        help='Lax-Friedrichs flux factor for tracers',
                        default=0.0)
    parser.add_argument('-lf-vel', '--laxfriedrichs-vel', type=float,
                        help='Lax-Friedrichs flux factor for velocity',
                        default=0.0)
    parser.add_argument('-e', '--elem-type', type=str,
                        help='Type of 2D element, either "tri" or "quad"',
                        default='tri')
    return parser


def parse_options():
    parser = get_argparser()
    args, unknown_args = parser.parse_known_args()
    args_dict = vars(args)
    run_lockexchange(**args_dict)


if __name__ == '__main__':
    parse_options()
