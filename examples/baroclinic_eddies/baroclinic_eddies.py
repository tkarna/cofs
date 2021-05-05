"""
Baroclinic channel test case following [1] and [2]

Domain is 160 by 500 km long channel, periodic in the x direction.
Depth is 1000 m.

Vertical temperature in the northern section is

T(z) = T_b + (T_b - T_s)*(z_b - z)/z_b

where T_b = 10.1, T_s = 13.1 degC are the bottom and surface temperatures, and z_b = -975 m is the bottom z coordinate.

Density is computed using a linear equation of state that does not depend on salinity:
rho = rho_0 - alpha*(T - T_ref)
with rho_0=1000 kg/m3, alpha=0.2 kg/m3/degC, T_ref=5.0 degC.

Horizontal mesh resolution is 10, 4, or 1 km. 20 vertical levels are used.

Coriolis parameter is 1.2e-4 1/s.
Bottom drag coefficient is 0.01.

Horizontal viscosity varies between 1.0 and 200 m2/s. Vertical viscosity is
set to constant 1e-4 m2/s. Tracer diffusion is set to zero.

[1] Ilicak et al. (2012). Spurious dianeutral mixing and the role
    of momentum closure. Ocean Modelling, 45-46(0):37-58.
[2] Petersen et al. (2015). Evaluation of the arbitrary Lagrangian-Eulerian
    vertical coordinate method in the MPAS-Ocean model. Ocean Modelling, 86:93-113.
"""

from thetis import *
from diagnostics import *


def run_problem(reso_dx=10.0, poly_order=1, element_family='dg-dg',
                reynolds_number=20.0, viscosity_scale=None, dt=None,
                elem_type='tri',
                laxfriedrichs_vel=0.0, laxfriedrichs_trc=0.0,
                number_of_z_levels=None, viscosity='const'):
    """
    Runs problem with a bunch of user defined options.
    """

    def get_nlayers(dx):
        # compute number of vertical layers
        return int(60./dx*1000. + 20)

    delta_x = reso_dx*1.e3
    if number_of_z_levels is not None:
        nlayers = number_of_z_levels
    else:
        nlayers = get_nlayers(delta_x)

    lx = 160e3
    ly = 500e3
    nx = int(lx/delta_x)
    ny = int(ly/delta_x)
    delta_x = lx/nx
    mesh2d = PeriodicRectangleMesh(
        nx, ny, lx, ly, direction='x',
        quadrilateral=(elem_type == 'quad')
    )
    depth = 1000.

    u_max = 1.0
    w_max = 1e-3
    # compute horizontal viscosity
    uscale = 0.1
    if viscosity_scale is None:
        # compute viscosity scale from mesh Reynolds number
        nu_scale = uscale * delta_x / reynolds_number
        visc_str = 'Re{:}'.format(reynolds_number)
    else:
        # compute mesh Reynolds number from viscosity scale
        nu_scale = viscosity_scale
        reynolds_number = uscale * delta_x / nu_scale
        visc_str = 'nu{:}'.format(nu_scale)

    f_cori = -1.2e-4
    bottom_drag = 0.01
    t_end = 320*24*3600.  # 365*24*3600.
    t_export = 3*3600.

    if os.getenv('THETIS_REGRESSION_TEST') is not None:
        t_export = 900.
        t_end = t_export
        nlayers = 4

    reso_str = 'dx' + str(np.round(delta_x/1000., decimals=1))
    reso_str += '_nz' + str(nlayers)
    if dt is not None:
        reso_str += '_dt{:}'.format(np.round(dt, 1))

    options_str = '_'.join([reso_str,
                            element_family,
                            elem_type,
                            'p{:}'.format(poly_order),
                            'visc-{:}'.format(viscosity),
                            visc_str,
                            'lf-vel{:.1f}'.format(laxfriedrichs_vel),
                            'lf-trc{:.1f}'.format(laxfriedrichs_trc),
                            ])
    outputdir = 'outputs_' + options_str

    # bathymetry
    P1_2d = get_functionspace(mesh2d, 'CG', 1)
    bathymetry_2d = Function(P1_2d, name='Bathymetry')
    bathymetry_2d.assign(depth)

    # temperature and salinity, results in 2.0 kg/m3 density difference
    salt_const = 35.0
    temp_bot = 10.1
    temp_surf = 13.1
    rho_0 = 1000.0
    physical_constants['rho0'].assign(rho_0)

    # create solver
    solver_obj = solver.FlowSolver(mesh2d, bathymetry_2d, nlayers)
    options = solver_obj.options
    options.polynomial_degree = poly_order
    options.element_family = element_family
    options.timestepper_type = 'SSPRK22'
    options.solve_salinity = False
    options.constant_salinity = Constant(salt_const)
    options.solve_temperature = True
    options.use_implicit_vertical_diffusion = True
    options.use_bottom_friction = True
    options.quadratic_drag_coefficient = Constant(bottom_drag)
    options.use_baroclinic_formulation = True
    options.coriolis_frequency = Constant(f_cori)
    options.use_lax_friedrichs_velocity = laxfriedrichs_vel > 0.0
    options.use_lax_friedrichs_tracer = laxfriedrichs_trc > 0.0
    options.lax_friedrichs_velocity_scaling_factor = Constant(laxfriedrichs_vel)
    options.lax_friedrichs_tracer_scaling_factor = Constant(laxfriedrichs_trc)
    options.use_limiter_for_tracers = True
    options.vertical_viscosity = Constant(1.0e-4)
    options.use_limiter_for_velocity = True
    if viscosity == 'smag':
        options.use_smagorinsky_viscosity = True
        options.smagorinsky_coefficient = Constant(1.0/np.sqrt(reynolds_number))
        options.horizontal_viscosity_scale = Constant(nu_scale)
    elif viscosity == 'const':
        options.horizontal_viscosity = Constant(nu_scale)
        options.horizontal_viscosity_scale = Constant(nu_scale)
    elif viscosity != 'none':
        raise Exception('Unknow viscosity type {:}'.format(viscosity))
    options.horizontal_diffusivity = None
    if dt is not None:
        options.timestepper_options.use_automatic_timestep = False
        options.timestep = dt
    options.simulation_export_time = t_export
    options.simulation_end_time = t_end
    options.output_directory = outputdir
    options.horizontal_velocity_scale = Constant(u_max)
    options.vertical_velocity_scale = Constant(w_max)
    options.check_volume_conservation_2d = True
    options.check_volume_conservation_3d = True
    options.check_temperature_conservation = True
    options.check_temperature_overshoot = True
    options.fields_to_export = ['uv_2d', 'elev_2d', 'uv_3d',
                                'w_3d', 'w_mesh_3d', 'temp_3d', 'salt_3d', 'density_3d',
                                'uv_dav_2d', 'uv_dav_3d', 'baroc_head_3d',
                                'smag_visc_3d']
    options.fields_to_export_hdf5 = ['uv_2d', 'elev_2d', 'uv_3d',
                                     'salt_3d', 'temp_3d', 'tke_3d', 'psi_3d']
    options.equation_of_state_type = 'linear'
    options.equation_of_state_options.rho_ref = rho_0
    options.equation_of_state_options.s_ref = salt_const
    options.equation_of_state_options.th_ref = 5.0
    options.equation_of_state_options.alpha = 0.2
    options.equation_of_state_options.beta = 0.0

    solver_obj.add_callback(RPECalculator(solver_obj))

    solver_obj.create_equations()

    print_output('Running eddy test case with options:')
    print_output('Mesh resolution dx={:} nlayers={:}'.format(delta_x, nlayers))
    print_output('Reynolds number: {:}'.format(reynolds_number))
    print_output('Horizontal viscosity: {:}'.format(nu_scale))
    print_output('Lax-Friedrichs factor vel: {:}'.format(laxfriedrichs_vel))
    print_output('Lax-Friedrichs factor trc: {:}'.format(laxfriedrichs_trc))
    print_output('Exporting to {:}'.format(outputdir))

    xyz = SpatialCoordinate(solver_obj.mesh)
    # vertical background stratification
    temp_vert = temp_bot + (temp_surf - temp_bot)*(-depth - xyz[2])/-depth
    # sinusoidal temperature anomaly
    temp_delta = -1.2
    y0 = 250.e3
    ya = 40.e3
    k = 3
    yd = 40.e3
    yw = y0 - ya*sin(2*pi*k*xyz[0]/lx)
    fy = (1. - (xyz[1] - yw)/yd)
    s_lo = 0.5*(sign(fy) + 1.)
    s_hi = 0.5*(sign(1. - fy) + 1.)
    temp_wave = temp_delta*(fy*s_lo*s_hi + (1.0-s_hi))
    # perturbation of one crest
    temp_delta2 = -0.3
    x2 = 110.e3
    x3 = 130.e3
    yw2 = y0 - ya/2*sin(pi*(xyz[0] - x2)/(x3 - x2))
    fy = (1. - (xyz[1] - yw2)/(yd/2))
    s_lo = 0.5*(sign(fy) + 1.)
    s_hi = 0.5*(sign(2. - fy) + 1.)
    temp_wave2 = temp_delta2*(fy*s_lo*s_hi + (1.0-s_hi))
    s_wave2 = 0.5*(sign(xyz[0] - x2)*(-1)*sign(xyz[0] - x3) + 1.)*s_hi
    temp_expr = temp_vert + s_wave2*temp_wave2 + (1.0 - s_wave2)*temp_wave
    temp_init3d = Function(solver_obj.function_spaces.H)
    temp_init3d.interpolate(temp_expr)
    solver_obj.assign_initial_conditions(temp=temp_init3d)

    # compute 2D baroclinic head and use it to initialize elevation field
    # to remove fast 2D gravity wave caused by the initial density difference
    elev_init = Function(solver_obj.function_spaces.H_bhead_2d)

    def compute_2d_baroc_head(solver_obj, output):
        """Computes vertical integral of baroc_head_3d"""
        compute_baroclinic_head(solver_obj)
        tmp_3d = Function(solver_obj.function_spaces.H_bhead)
        bhead_av_op = VerticalIntegrator(
            solver_obj.fields.baroc_head_3d,
            tmp_3d,
            bottom_to_top=True,
            average=True,
            elevation=solver_obj.fields.elev_cg_2d.view_3d,
            bathymetry=solver_obj.fields.bathymetry_2d.view_3d)
        bhead_surf_extract_op = SubFunctionExtractor(
            tmp_3d,
            output,
            boundary='top', elem_facet='top')
        bhead_av_op.solve()
        bhead_surf_extract_op.solve()

    compute_2d_baroc_head(solver_obj, elev_init)
    elev_init *= -1  # flip sign => total pressure gradient is zero
    mean_elev = assemble(elev_init*dx)/lx/ly
    elev_init += -mean_elev  # remove mean
    solver_obj.assign_initial_conditions(temp=temp_init3d, elev=elev_init)

    # custom export of surface temperature field
    surf_temp_2d = Function(solver_obj.function_spaces.H_2d, name='Temperature')
    extract_surf_temp = SubFunctionExtractor(solver_obj.fields.temp_3d, surf_temp_2d)
    surf_temp_file = File(options.output_directory + '/SurfTemperature2d.pvd')

    def export_func():
        extract_surf_temp.solve()
        surf_temp_file.write(surf_temp_2d)

    solver_obj.iterate(export_func=export_func)


def get_argparser():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--reso_dx', type=float,
                        help='mesh resolution in kilometers',
                        default=10.0)
    parser.add_argument('-p', '--poly_order', type=int, default=1,
                        help='order of finite element space')
    parser.add_argument('-f', '--element-family', type=str,
                        help='finite element family', default='dg-dg')
    parser.add_argument('-re', '--reynolds-number', type=float, default=1.0,
                        help='mesh Reynolds number for Smagorinsky scheme')
    parser.add_argument('-nu', '--viscosity-scale', type=float,
                        help='constant viscosity scale (optional, use instead of Re)')
    parser.add_argument('-dt', '--dt', type=float,
                        help='force value for 3D time step')
    parser.add_argument('-nz', '--number-of-z-levels', type=int,
                        help='force number of vertical levels')
    parser.add_argument('-visc', '--viscosity', type=str,
                        help='Type of horizontal viscosity',
                        default='const',
                        choices=['const', 'smag', 'none'])
    parser.add_argument('-lf-trc', '--laxfriedrichs-trc', type=float,
                        help='Lax-Friedrichs flux factor for tracers',
                        default=0.0)
    parser.add_argument('-lf-vel', '--laxfriedrichs-vel', type=float,
                        help='Lax-Friedrichs flux factor for velocity',
                        default=1.0)
    parser.add_argument('-e', '--elem-type', type=str,
                        help='Type of 2D element, either "tri" or "quad"',
                        default='tri')
    return parser


def parse_options():
    parser = get_argparser()
    args, unknown_args = parser.parse_known_args()
    args_dict = vars(args)
    run_problem(**args_dict)


if __name__ == '__main__':
    parse_options()
