"""
Testing 3D vertical diffusion of tracers against analytical solution.
"""
from thetis import *
import numpy
from scipy import stats
import pytest


def run(refinement, **model_options):
    print_output('--- running refinement {:}'.format(refinement))
    implicit = model_options.pop('implicit', False)

    # domain dimensions - channel in x-direction
    lx = 7.0e3
    ly = 5.0e3
    area = lx*ly
    depth = 40.0
    vertical_diffusivity = 5e-3

    # mesh
    n_layers = 6*refinement
    nx = 3  # constant
    ny = 2  # constant
    mesh2d = RectangleMesh(nx, ny, lx, ly)

    # set time steps
    if implicit:
        dt = 100.
    else:
        # stable explicit time step for diffusion
        dz = depth/n_layers
        alpha = 1.0/200.0
        dt = alpha * dz**2/vertical_diffusivity
    # simulation run time
    t_end = 1900.
    # initial time
    t_init = 100.0  # NOTE start from t > 0 for smoother init cond
    # eliminate reminder
    ndt = np.ceil((t_end-t_init)/dt)
    dt = (t_end-t_init)/ndt
    dt_2d = dt/2
    t_export = (t_end-t_init)/6

    # outputs
    outputdir = 'outputs'

    # bathymetry
    p1_2d = get_functionspace(mesh2d, 'CG', 1)
    bathymetry_2d = Function(p1_2d, name='Bathymetry')
    bathymetry_2d.assign(depth)

    solverobj = solver.FlowSolver(mesh2d, bathymetry_2d, n_layers)
    options = solverobj.options
    options.use_nonlinear_equations = False
    options.use_limiter_for_tracers = False
    options.horizontal_velocity_scale = Constant(1.0)
    options.no_exports = True
    options.output_directory = outputdir
    options.simulation_end_time = t_end
    options.simulation_export_time = t_export
    options.timestep = dt
    options.timestep_2d = dt_2d
    options.solve_salinity = True
    options.use_implicit_vertical_diffusion = implicit
    options.use_bottom_friction = False
    options.fields_to_export = ['salt_3d']
    options.vertical_diffusivity = Constant(vertical_diffusivity)
    options.update(model_options)
    if hasattr(options.timestepper_options, 'use_automatic_timestep'):
        options.timestepper_options.use_automatic_timestep = False

    solverobj.create_equations()

    t = t_init  # simulation time

    t_const = Constant(t)
    u_max = 1.0
    u_min = -1.0
    z0 = -depth/2.0
    x, y, z = SpatialCoordinate(solverobj.mesh)
    ana_salt_expr = 0.5*(u_max + u_min) - 0.5*(u_max - u_min)*erf((z - z0)/sqrt(4*vertical_diffusivity*t_const))

    solverobj.assign_initial_conditions(salt=ana_salt_expr)

    # export analytical solution
    if not options.no_exports:
        salt_ana = Function(solverobj.function_spaces.H, name='salt analytical')
        out_salt_ana = File(os.path.join(options.output_directory, 'salt_ana.pvd'))

    def export_func():
        if not options.no_exports:
            solverobj.export()
            # update analytical solution to correct time
            t_const.assign(t)
            salt_ana.project(ana_salt_expr)
            out_salt_ana.write(salt_ana)

    # export initial conditions
    export_func()

    # custom time loop that solves tracer eq only
    if implicit:
        ti = solverobj.timestepper.timesteppers.salt_impl
    else:
        ti = solverobj.timestepper.timesteppers.salt_expl
    i = 0
    iexport = 1
    next_export_t = t + solverobj.options.simulation_export_time
    while t < t_end - 1e-8:
        ti.advance(t)
        t += solverobj.dt
        i += 1
        if t >= next_export_t - 1e-8:
            print_output('{:3d} i={:5d} t={:8.2f} s salt={:8.2f}'.format(iexport, i, t, norm(solverobj.fields.salt_3d)))
            export_func()
            next_export_t += solverobj.options.simulation_export_time
            iexport += 1

    # project analytical solultion on high order mesh
    t_const.assign(t)
    # compute L2 norm
    l2_err = errornorm(ana_salt_expr, solverobj.fields.salt_3d)/numpy.sqrt(area)
    print_output('L2 error {:.12f}'.format(l2_err))

    return l2_err


def run_convergence(ref_list, expected_rate=None, saveplot=False, **options):
    """Runs test for a list of refinements and computes error convergence rate"""
    polynomial_degree = options.get('polynomial_degree', 1)
    l2_err = []
    for r in ref_list:
        l2_err.append(run(r, **options))
    x_log = numpy.log10(numpy.array(ref_list, dtype=float)**-1)
    y_log = numpy.log10(numpy.array(l2_err))
    setup_name = 'v-diffusion'

    def check_convergence(x_log, y_log, expected_slope, field_str, saveplot):
        slope, intercept, r_value, p_value, std_err = stats.linregress(x_log, y_log)
        if saveplot:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(5, 5))
            # plot points
            ax.plot(x_log, y_log, 'k.')
            x_min = x_log.min()
            x_max = x_log.max()
            offset = 0.05*(x_max - x_min)
            npoints = 50
            xx = numpy.linspace(x_min - offset, x_max + offset, npoints)
            yy = intercept + slope*xx
            # plot line
            ax.plot(xx, yy, linestyle='--', linewidth=0.5, color='k')
            ax.text(xx[2*int(npoints/3)], yy[2*int(npoints/3)], '{:4.2f}'.format(slope),
                    verticalalignment='top',
                    horizontalalignment='left')
            ax.set_xlabel('log10(dx)')
            ax.set_ylabel('log10(L2 error)')
            ax.set_title(' '.join([setup_name, field_str, 'degree={:}'.format(polynomial_degree)]))
            ref_str = 'ref-' + '-'.join([str(r) for r in ref_list])
            degree_str = 'o{:}'.format(polynomial_degree)
            imgfile = '_'.join(['convergence', setup_name, field_str, ref_str, degree_str])
            imgfile += '.png'
            imgdir = create_directory('plots')
            imgfile = os.path.join(imgdir, imgfile)
            print_output('saving figure {:}'.format(imgfile))
            plt.savefig(imgfile, dpi=200, bbox_inches='tight')
        if expected_slope is not None:
            err_msg = '{:}: Wrong convergence rate {:.4f}, expected {:.4f}'.format(setup_name, slope, expected_slope)
            assert slope > expected_slope, err_msg
            print_output('{:}: convergence rate {:.4f} PASSED'.format(setup_name, slope))
        else:
            print_output('{:}: {:} convergence rate {:.4f}'.format(setup_name, field_str, slope))
        return slope

    if expected_rate is None:
        expected_rate = polynomial_degree+1
    check_convergence(x_log, y_log, expected_rate, 'salt', saveplot)

# ---------------------------
# standard tests for pytest
# ---------------------------


@pytest.mark.parametrize(
    ('stepper', 'polynomial_degree', 'implicit', 'expected_rate'),
    [
        ('SSPRK22', 0, False, 1.0),
        ('SSPRK22', 1, False, 1.7),
        ('SSPRK22', 0, True, 1.0),
        ('SSPRK22', 1, True, 2.2),
    ]
)
def test_vertical_diffusion(polynomial_degree, implicit, stepper,
                            expected_rate):
    run_convergence([1, 2, 4], expected_rate=expected_rate,
                    polynomial_degree=polynomial_degree,
                    implicit=implicit, timestepper_type=stepper)

# ---------------------------
# run individual setup for debugging
# ---------------------------


if __name__ == '__main__':
    run_convergence([1, 2, 3], polynomial_degree=1,
                    implicit=False,
                    element_family='dg-dg',
                    timestepper_type='SSPRK22',
                    expected_rate=1.7,
                    no_exports=False, saveplot=True)
