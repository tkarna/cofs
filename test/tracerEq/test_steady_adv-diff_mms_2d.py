"""
Testing 2D tracer advection-diffusion equation with method of manufactured solution (MMS).
"""
from thetis import *
import numpy
from scipy import stats
import pytest


class Setup1:
    """
    Constant bathymetry and u velocty, zero diffusivity, non-trivial tracer
    """
    def bath(self, x, y, lx, ly):
        return Constant(40.0)

    def elev(self, x, y, lx, ly):
        return Constant(0.0)

    def uv(self, x, y, lx, ly):
        return as_vector(
            [
                Constant(1.0),
                Constant(0.0),
            ])

    def kappa(self, x, y, lx, ly):
        return Constant(0.0)

    def tracer(self, x, y, lx, ly):
        return sin(0.2*pi*(3.0*x + 1.0*y)/lx)

    def residual(self, x, y, lx, ly):
        return 0.6*pi*cos(0.2*pi*(3.0*x + 1.0*y)/lx)/lx


class Setup2:
    """
    Constant bathymetry, zero velocity, constant kappa, x-varying T
    """
    def bath(self, x, y, lx, ly):
        return Constant(40.0)

    def elev(self, x, y, lx, ly):
        return Constant(0.0)

    def uv(self, x, y, lx, ly):
        return as_vector(
            [
                Constant(1.0),
                Constant(0.0),
            ])

    def kappa(self, x, y, lx, ly):
        return Constant(50.0)

    def tracer(self, x, y, lx, ly):
        return sin(3*pi*x/lx)

    def residual(self, x, y, lx, ly):
        return 3.0*pi*cos(3*pi*x/lx)/lx - 450.0*pi**2*sin(3*pi*x/lx)/lx**2


class Setup3:
    """
    Jump in bathymetry and velocity. Zero diffusion.
    """
    def bath(self, x, y, lx, ly):
        return conditional(x > lx/2, Constant(40.), Constant(20.))

    def elev(self, x, y, lx, ly):
        return Constant(0.0)

    def uv(self, x, y, lx, ly):
        return conditional(x > lx/2, Constant(1.), Constant(2.)) * as_vector((1.0, 0.5))

    def kappa(self, x, y, lx, ly):
        return Constant(0.0)

    def tracer(self, x, y, lx, ly):
        return exp(1.25*x/lx) * exp(-2.5*y/lx)

    def residual(self, x, y, lx, ly):
        return Constant(0.0)


class Setup4:
    """
    Jump in tracer and velocity. Constant bathymetry and zero diffusion.

    NOTE: uv and elev are not a steady solution for the SWE. This case is only
    valid for conservative, as the jump conditions are now different:
    * for nonconservative the jump condition is [c]=0 (assuming all terms other
      than advection are zero)
    * for conservative where we solve for q=Hc, it is [uq]=[uHc]=0
    For steady state SWE, [uH]=0 which means that the jump conditions are equivalent.
    """
    def bath(self, x, y, lx, ly):
        return Constant(40.)

    def elev(self, x, y, lx, ly):
        return Constant(0.0)

    def uv(self, x, y, lx, ly):
        return conditional(x > lx/2, Constant(1.), Constant(2.)) * as_vector((1.0, 0.5))

    def kappa(self, x, y, lx, ly):
        return Constant(0.0)

    def tracer(self, x, y, lx, ly):
        return conditional(x > lx/2, Constant(2.), Constant(1.)) * exp(1.25*x/lx) * exp(-2.5*y/lx)

    def residual(self, x, y, lx, ly):
        return Constant(0.0)


def run(setup, refinement, do_export=True, **options):
    """Run single test and return L2 error"""
    print_output('--- running {:} refinement {:}'.format(setup.__name__, refinement))
    # domain dimensions
    lx = 15e3
    ly = 10e3
    area = lx*ly
    t_end = 200.0

    setup_obj = setup()

    # mesh
    nx = 4*refinement
    ny = 4*refinement
    mesh2d = RectangleMesh(nx, ny, lx, ly)

    # outputs
    outputdir = 'outputs_{}'.format(refinement)
    if do_export:
        out_2 = File(os.path.join(outputdir, 'Tracer.pvd'))

    # bathymetry
    x_2d, y_2d = SpatialCoordinate(mesh2d)
    p1_2d = get_functionspace(mesh2d, 'CG', 1)
    bathymetry_2d = Function(p1_2d, name='Bathymetry')
    bathymetry_2d.project(setup_obj.bath(x_2d, y_2d, lx, ly))

    solver_obj = solver2d.FlowSolver2d(mesh2d, bathymetry_2d)
    solver_obj.options.element_family = 'dg-dg'
    solver_obj.options.horizontal_velocity_scale = Constant(1.0)
    solver_obj.options.no_exports = not do_export
    solver_obj.options.output_directory = outputdir
    solver_obj.options.simulation_end_time = t_end
    solver_obj.options.fields_to_export = ['tracer_2d', 'uv_2d', ]
    solver_obj.options.horizontal_viscosity_scale = Constant(50.0)
    solver_obj.options.update(options)
    solver_obj.options.solve_tracer = True
    if hasattr(solver_obj.options.timestepper_options, 'implicitness_theta'):
        solver_obj.options.timestepper_options.implicitness_theta = 1.0
    solver_obj.create_function_spaces()

    # functions for source terms
    x, y = SpatialCoordinate(solver_obj.mesh2d)
    solver_obj.options.tracer_source_2d = setup_obj.residual(x, y, lx, ly)

    # diffusivuty
    solver_obj.options.horizontal_diffusivity = setup_obj.kappa(x, y, lx, ly)

    # analytical solution
    trac_ana = setup_obj.tracer(x, y, lx, ly)
    if solver_obj.options.use_tracer_conservative_form:
        # the tracer in the setups is depth-averaged
        # with conservative form we solve for the depth-integrated variable
        H = setup_obj.bath(x, y, lx, ly) + setup_obj.elev(x, y, lx, ly)
        trac_ana = trac_ana * H

    bnd_tracer = {'value': trac_ana}
    solver_obj.bnd_functions['tracer'] = {1: bnd_tracer, 2: bnd_tracer,
                                          3: bnd_tracer, 4: bnd_tracer}

    solver_obj.create_equations()
    solver_obj.assign_initial_conditions(elev=setup_obj.elev(x, y, lx, ly),
                                         uv=setup_obj.uv(x, y, lx, ly),
                                         tracer=trac_ana)

    # solve tracer advection-diffusion equation with residual source term
    ti = solver_obj.timestepper
    ti.timesteppers.tracer.initialize(ti.fields.tracer_2d)
    if do_export:
        out_2.write(ti.fields.tracer_2d)

    t = 0
    while t < t_end:
        ti.timesteppers.tracer.advance(t)
        if ti.options.use_limiter_for_tracers:
            ti.solver.tracer_limiter.apply(ti.fields.tracer_2d)
        t += solver_obj.dt
        if do_export:
            out_2.write(ti.fields.tracer_2d)

    l2_err = errornorm(trac_ana, solver_obj.fields.tracer_2d)/numpy.sqrt(area)
    print_output('L2 error {:.12f}'.format(l2_err))

    return l2_err


def run_convergence(setup, ref_list, do_export=False, save_plot=False, **options):
    """Runs test for a list of refinements and computes error convergence rate"""
    l2_err = []
    for r in ref_list:
        l2_err.append(run(setup, r, do_export=do_export, **options))
    x_log = numpy.log10(numpy.array(ref_list, dtype=float)**-1)
    y_log = numpy.log10(numpy.array(l2_err))

    def check_convergence(x_log, y_log, expected_slope, field_str, save_plot):
        slope_rtol = 0.2
        slope, intercept, r_value, p_value, std_err = stats.linregress(x_log, y_log)
        setup_name = setup.__name__
        if save_plot:
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
            ax.set_title('tracer adv-diff MMS DG ')
            ref_str = 'ref-' + '-'.join([str(r) for r in ref_list])

            imgfile = '_'.join(['convergence', setup_name, field_str, ref_str])
            imgfile += '.png'
            img_dir = create_directory('plots')
            imgfile = os.path.join(img_dir, imgfile)
            print_output('saving figure {:}'.format(imgfile))
            plt.savefig(imgfile, dpi=200, bbox_inches='tight')
        if expected_slope is not None:
            err_msg = '{:}: Wrong convergence rate {:.4f}, expected {:.4f}'.format(setup_name, slope, expected_slope)
            assert abs(slope - expected_slope)/expected_slope < slope_rtol, err_msg
            print_output('{:}: convergence rate {:.4f} PASSED'.format(setup_name, slope))
        else:
            print_output('{:}: {:} convergence rate {:.4f}'.format(setup_name, field_str, slope))
        return slope

    check_convergence(x_log, y_log, 2, 'tracer', save_plot)


# ---------------------------
# standard tests for pytest
# ---------------------------

@pytest.fixture(params=['CrankNicolson', 'DIRK22', 'DIRK33', 'BackwardEuler'])
def timestepper_type(request):
    return request.param


@pytest.fixture(params=[Setup1, Setup2, Setup3],
                ids=['setup1', 'setup2', 'setup3'])
def setup(request):
    return request.param


@pytest.fixture(params=[True, False],
                ids=['conservative', 'nonconservative'])
def conservative_form(request):
    return request.param


def test_convergence(setup, timestepper_type, conservative_form):
    run_convergence(setup, [1, 2, 3], save_plot=False, timestepper_type=timestepper_type,
                    use_tracer_conservative_form=conservative_form)


# setup4 is valid for conservative only
def test_convergence_conservative_only(timestepper_type):
    run_convergence(Setup4, [1, 2, 3], save_plot=False, timestepper_type=timestepper_type,
                    use_tracer_conservative_form=True)


if __name__ == '__main__':
    run_convergence(Setup1, [1, 2, 3], save_plot=True, timestepper_type='CrankNicolson')
