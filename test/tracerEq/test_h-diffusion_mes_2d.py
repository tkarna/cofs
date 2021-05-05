"""
Testing 2D horizontal diffusion of tracers against analytical solution.
"""
from thetis import *
import numpy
from scipy import stats
import pytest


def run(refinement, **model_options):
    print_output('--- running refinement {:}'.format(refinement))

    # domain dimensions - channel in x-direction
    lx = 20.0e3
    ly = 5.0e3 / refinement
    area = lx*ly
    depth = 30.0
    horizontal_diffusivity = 1.0e3

    # mesh
    nx = 8 * refinement + 1
    ny = 1  # constant -- channel
    mesh2d = RectangleMesh(nx, ny, lx, ly)

    # simulation run time
    t_end = 3000.0
    # initial time
    t_init = 1000.0  # NOTE start from t > 0 for smoother init cond
    t_export = (t_end - t_init)/8.0

    # outputs
    outputdir = 'outputs'

    # bathymetry
    p1_2d = get_functionspace(mesh2d, 'CG', 1)
    bathymetry_2d = Function(p1_2d, name='Bathymetry')
    bathymetry_2d.assign(depth)

    solverobj = solver2d.FlowSolver2d(mesh2d, bathymetry_2d)
    options = solverobj.options
    options.use_nonlinear_equations = False
    options.horizontal_velocity_scale = Constant(1.0)
    options.no_exports = True
    options.output_directory = outputdir
    options.simulation_end_time = t_end
    options.simulation_export_time = t_export
    options.solve_tracer = True
    options.use_limiter_for_tracers = True
    options.fields_to_export = ['tracer_2d']
    options.horizontal_diffusivity = Constant(horizontal_diffusivity)
    options.horizontal_viscosity_scale = Constant(horizontal_diffusivity)
    options.update(model_options)

    solverobj.create_equations()

    t = t_init  # simulation time

    t_const = Constant(t)
    u_max = 1.0
    u_min = -1.0
    x0 = lx/2.0
    x, y = SpatialCoordinate(solverobj.mesh2d)
    tracer_expr = 0.5*(u_max + u_min) - 0.5*(u_max - u_min)*erf((x - x0)/sqrt(4*horizontal_diffusivity*t_const))
    tracer_ana = Function(solverobj.function_spaces.H_2d, name='tracer analytical')
    elev_init = Function(solverobj.function_spaces.H_2d, name='elev init')
    solverobj.assign_initial_conditions(elev=elev_init, tracer=tracer_expr)

    # export analytical solution
    if not options.no_exports:
        out_tracer_ana = File(os.path.join(options.output_directory, 'tracer_ana.pvd'))

    def export_func():
        if not options.no_exports:
            solverobj.export()
            # update analytical solution to correct time
            t_const.assign(t)
            out_tracer_ana.write(tracer_ana.project(tracer_expr))

    # export initial conditions
    export_func()

    # custom time loop that solves tracer equation only
    ti = solverobj.timestepper.timesteppers.tracer

    i = 0
    iexport = 1
    next_export_t = t + solverobj.options.simulation_export_time
    while t < t_end - 1e-8:
        ti.advance(t)
        t += solverobj.dt
        i += 1
        if t >= next_export_t - 1e-8:
            print_output('{:3d} i={:5d} t={:8.2f} s tracer={:8.2f}'.format(iexport, i, t, norm(solverobj.fields.tracer_2d)))
            export_func()
            next_export_t += solverobj.options.simulation_export_time
            iexport += 1

    # project analytical solution on high order mesh
    t_const.assign(t)
    # compute L2 norm
    l2_err = errornorm(tracer_expr, solverobj.fields.tracer_2d)/numpy.sqrt(area)
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
    setup_name = 'h-diffusion'

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
    check_convergence(x_log, y_log, expected_rate, 'tracer', saveplot)

# ---------------------------
# standard tests for pytest
# ---------------------------


@pytest.mark.parametrize(
    ('stepper', 'polynomial_degree', 'expected_rate'),
    [
        ('CrankNicolson', 1, 1.8),
        ('SSPRK33', 1, 1.8),
        ('ForwardEuler', 1, 1.8),
        ('BackwardEuler', 1, 1.8),
        ('DIRK22', 1, 1.8),
        ('DIRK33', 1, 1.8),
    ]
)
def test_horizontal_diffusion(polynomial_degree, stepper, expected_rate):
    run_convergence([1, 2, 3], polynomial_degree=polynomial_degree,
                    timestepper_type=stepper, expected_rate=expected_rate,
                    )

# ---------------------------
# run individual setup for debugging
# ---------------------------


if __name__ == '__main__':
    run_convergence([1, 2, 4, 6, 8], polynomial_degree=1,
                    timestepper_type='CrankNicolson',
                    expected_rate=1.8,
                    no_exports=False, saveplot=True)
