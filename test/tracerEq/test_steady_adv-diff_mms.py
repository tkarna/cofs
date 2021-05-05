"""
Testing 3D tracer advection-diffusion equation with method of manufactured solution (MMS).
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

    def uv(self, x, y, z, lx, ly):
        return as_vector(
            [
                Constant(1.0),
                Constant(0.0),
                Constant(0),
            ])

    def w(self, x, y, z, lx, ly):
        return as_vector(
            [
                Constant(0),
                Constant(0),
                Constant(0),
            ])

    def kappa(self, x, y, z, lx, ly):
        return Constant(0.0)

    def tracer(self, x, y, z, lx, ly):
        return sin(0.2*pi*(3.0*x + 1.0*y)/lx)

    def residual(self, x, y, z, lx, ly):
        return 0.6*pi*cos(0.2*pi*(3.0*x + 1.0*y)/lx)/lx


class Setup2:
    """
    Constant bathymetry, zero velocity, constant kappa, x-varying T
    """
    def bath(self, x, y, lx, ly):
        return Constant(40.0)

    def elev(self, x, y, lx, ly):
        return Constant(0.0)

    def uv(self, x, y, z, lx, ly):
        return as_vector(
            [
                Constant(1.0),
                Constant(0.0),
                Constant(0),
            ])

    def w(self, x, y, z, lx, ly):
        return as_vector(
            [
                Constant(0),
                Constant(0),
                Constant(0),
            ])

    def kappa(self, x, y, z, lx, ly):
        return Constant(50.0)

    def tracer(self, x, y, z, lx, ly):
        return sin(3*pi*x/lx)

    def residual(self, x, y, z, lx, ly):
        return 3.0*pi*cos(3*pi*x/lx)/lx - 450.0*pi**2*sin(3*pi*x/lx)/lx**2


class Setup3:
    """
    Constant bathymetry, zero kappa, non-trivial velocity and T
    """
    def bath(self, x, y, lx, ly):
        return Constant(40.0)

    def elev(self, x, y, lx, ly):
        return Constant(0.0)

    def uv(self, x, y, z, lx, ly):
        return as_vector(
            [
                sin(pi*z/40)*sin(pi*(y/ly + 2*x/lx)),
                sin(pi*z/40)*sin(pi*(0.3*y/ly + 0.3*x/lx)),
                Constant(0),
            ])

    def w(self, x, y, z, lx, ly):
        return as_vector(
            [
                Constant(0),
                Constant(0),
                12.0*cos(pi*z/40)*cos(pi*(0.3*y/ly + 0.3*x/lx))/ly + 12.0*cos(pi*(0.3*y/ly + 0.3*x/lx))/ly + 80*cos(pi*z/40)*cos(pi*(y/ly + 2*x/lx))/lx + 80*cos(pi*(y/ly + 2*x/lx))/lx,
            ])

    def kappa(self, x, y, z, lx, ly):
        return Constant(0.0)

    def tracer(self, x, y, z, lx, ly):
        return (0.8*cos(0.0125*pi*z) + 0.2)*cos(pi*(0.75*y/ly + 1.5*x/lx))

    def residual(self, x, y, z, lx, ly):
        return (-0.3*pi*sin(pi*z/40)*cos(pi*(0.3*y/ly + 0.3*x/lx))/ly - 2*pi*sin(pi*z/40)*cos(pi*(y/ly + 2*x/lx))/lx)*(0.8*cos(0.0125*pi*z) + 0.2)*cos(pi*(0.75*y/ly + 1.5*x/lx)) - 0.01*pi*(12.0*cos(pi*z/40)*cos(pi*(0.3*y/ly + 0.3*x/lx))/ly + 12.0*cos(pi*(0.3*y/ly + 0.3*x/lx))/ly + 80*cos(pi*z/40)*cos(pi*(y/ly + 2*x/lx))/lx + 80*cos(pi*(y/ly + 2*x/lx))/lx)*sin(0.0125*pi*z)*cos(pi*(0.75*y/ly + 1.5*x/lx)) - 0.75*pi*(0.8*cos(0.0125*pi*z) + 0.2)*sin(pi*z/40)*sin(pi*(0.3*y/ly + 0.3*x/lx))*sin(pi*(0.75*y/ly + 1.5*x/lx))/ly + 0.3*pi*(0.8*cos(0.0125*pi*z) + 0.2)*sin(pi*z/40)*cos(pi*(0.3*y/ly + 0.3*x/lx))*cos(pi*(0.75*y/ly + 1.5*x/lx))/ly - 1.5*pi*(0.8*cos(0.0125*pi*z) + 0.2)*sin(pi*z/40)*sin(pi*(0.75*y/ly + 1.5*x/lx))*sin(pi*(y/ly + 2*x/lx))/lx + 2*pi*(0.8*cos(0.0125*pi*z) + 0.2)*sin(pi*z/40)*cos(pi*(0.75*y/ly + 1.5*x/lx))*cos(pi*(y/ly + 2*x/lx))/lx


class Setup4:
    """
    Constant bathymetry, constant kappa, non-trivial velocity and T
    """
    def bath(self, x, y, lx, ly):
        return Constant(40.0)

    def elev(self, x, y, lx, ly):
        return Constant(0.0)

    def uv(self, x, y, z, lx, ly):
        return as_vector(
            [
                sin(pi*z/40)*sin(pi*(y/ly + 2*x/lx)),
                sin(pi*z/40)*sin(pi*(0.3*y/ly + 0.3*x/lx)),
                Constant(0),
            ])

    def w(self, x, y, z, lx, ly):
        return as_vector(
            [
                Constant(0),
                Constant(0),
                12.0*cos(pi*z/40)*cos(pi*(0.3*y/ly + 0.3*x/lx))/ly + 12.0*cos(pi*(0.3*y/ly + 0.3*x/lx))/ly + 80*cos(pi*z/40)*cos(pi*(y/ly + 2*x/lx))/lx + 80*cos(pi*(y/ly + 2*x/lx))/lx,
            ])

    def kappa(self, x, y, z, lx, ly):
        return Constant(50.0)

    def tracer(self, x, y, z, lx, ly):
        return (0.8*cos(0.0125*pi*z) + 0.2)*cos(pi*(0.75*y/ly + 1.5*x/lx))

    def residual(self, x, y, z, lx, ly):
        return (-0.3*pi*sin(pi*z/40)*cos(pi*(0.3*y/ly + 0.3*x/lx))/ly - 2*pi*sin(pi*z/40)*cos(pi*(y/ly + 2*x/lx))/lx)*(0.8*cos(0.0125*pi*z) + 0.2)*cos(pi*(0.75*y/ly + 1.5*x/lx)) - 0.01*pi*(12.0*cos(pi*z/40)*cos(pi*(0.3*y/ly + 0.3*x/lx))/ly + 12.0*cos(pi*(0.3*y/ly + 0.3*x/lx))/ly + 80*cos(pi*z/40)*cos(pi*(y/ly + 2*x/lx))/lx + 80*cos(pi*(y/ly + 2*x/lx))/lx)*sin(0.0125*pi*z)*cos(pi*(0.75*y/ly + 1.5*x/lx)) - 0.75*pi*(0.8*cos(0.0125*pi*z) + 0.2)*sin(pi*z/40)*sin(pi*(0.3*y/ly + 0.3*x/lx))*sin(pi*(0.75*y/ly + 1.5*x/lx))/ly + 0.3*pi*(0.8*cos(0.0125*pi*z) + 0.2)*sin(pi*z/40)*cos(pi*(0.3*y/ly + 0.3*x/lx))*cos(pi*(0.75*y/ly + 1.5*x/lx))/ly - 28.125*pi**2*(0.8*cos(0.0125*pi*z) + 0.2)*cos(pi*(0.75*y/ly + 1.5*x/lx))/ly**2 - 1.5*pi*(0.8*cos(0.0125*pi*z) + 0.2)*sin(pi*z/40)*sin(pi*(0.75*y/ly + 1.5*x/lx))*sin(pi*(y/ly + 2*x/lx))/lx + 2*pi*(0.8*cos(0.0125*pi*z) + 0.2)*sin(pi*z/40)*cos(pi*(0.75*y/ly + 1.5*x/lx))*cos(pi*(y/ly + 2*x/lx))/lx - 112.5*pi**2*(0.8*cos(0.0125*pi*z) + 0.2)*cos(pi*(0.75*y/ly + 1.5*x/lx))/(lx*ly) - 112.5*pi**2*(0.8*cos(0.0125*pi*z) + 0.2)*cos(pi*(0.75*y/ly + 1.5*x/lx))/lx**2


def run(setup, refinement, order, do_export=True, **options):
    """Run single test and return L2 error"""
    print_output('--- running {:} refinement {:}'.format(setup.__name__, refinement))
    # domain dimensions
    lx = 15e3
    ly = 10e3
    area = lx*ly
    t_end = 200.0

    setup_obj = setup()

    # mesh
    n_layers = 4*refinement
    nx = 4*refinement
    ny = 4*refinement
    mesh2d = RectangleMesh(nx, ny, lx, ly)

    # outputs
    outputdir = 'outputs'
    if do_export:
        out_t = File(os.path.join(outputdir, 'T.pvd'))

    # bathymetry
    x_2d, y_2d = SpatialCoordinate(mesh2d)
    p1_2d = get_functionspace(mesh2d, 'CG', 1)
    bathymetry_2d = Function(p1_2d, name='Bathymetry')
    bathymetry_2d.project(setup_obj.bath(x_2d, y_2d, lx, ly))

    solver_obj = solver.FlowSolver(mesh2d, bathymetry_2d, n_layers)
    solver_obj.options.element_family = 'dg-dg'
    solver_obj.options.polynomial_degree = order
    solver_obj.options.horizontal_velocity_scale = Constant(1.0)
    solver_obj.options.use_bottom_friction = False
    solver_obj.options.no_exports = not do_export
    solver_obj.options.output_directory = outputdir
    solver_obj.options.simulation_end_time = t_end
    solver_obj.options.fields_to_export = ['salt_3d', 'uv_3d', 'w_3d']
    solver_obj.options.horizontal_viscosity_scale = Constant(50.0)
    solver_obj.options.update(options)

    solver_obj.create_function_spaces()

    # functions for source terms
    x, y, z = SpatialCoordinate(solver_obj.mesh)
    solver_obj.options.salinity_source_3d = setup_obj.residual(x, y, z, lx, ly)

    # diffusivuty
    solver_obj.options.horizontal_diffusivity = setup_obj.kappa(x, y, z, lx, ly)

    # analytical solution
    trac_ana = setup_obj.tracer(x, y, z, lx, ly)

    bnd_salt = {'value': trac_ana}
    solver_obj.bnd_functions['salt'] = {1: bnd_salt, 2: bnd_salt,
                                        3: bnd_salt, 4: bnd_salt}
    # NOTE use symmetic uv condition to get correct w
    bnd_mom = {'symm': None}
    solver_obj.bnd_functions['momentum'] = {1: bnd_mom, 2: bnd_mom,
                                            3: bnd_mom, 4: bnd_mom}

    solver_obj.create_equations()
    dt = solver_obj.dt
    # elevation field
    solver_obj.fields.elev_2d.project(setup_obj.elev(x_2d, y_2d, lx, ly))
    # update mesh and fields
    solver_obj.mesh_updater.update_mesh_coordinates()

    # salinity field
    solver_obj.fields.salt_3d.project(setup_obj.tracer(x, y, z, lx, ly))
    # velocity field
    solver_obj.fields.uv_3d.project(setup_obj.uv(x, y, z, lx, ly))
    solver_obj.w_solver.solve()

    if do_export:
        out_t.write(trac_ana)
        solver_obj.export()

    # solve salinity advection-diffusion equation with residual source term
    ti = solver_obj.timestepper
    ti.timesteppers.salt_expl.initialize(ti.fields.salt_3d)
    t = 0
    while t < t_end - 1e-5:
        ti.timesteppers.salt_expl.advance(t)
        if ti.options.use_limiter_for_tracers:
            ti.solver.tracer_limiter.apply(ti.fields.salt_3d)
        t += dt

    if do_export:
        out_t.write(trac_ana)
        solver_obj.export()

    l2_err = errornorm(trac_ana, solver_obj.fields.salt_3d)/numpy.sqrt(area)
    print_output('L2 error {:.12f}'.format(l2_err))

    return l2_err


def run_convergence(setup, ref_list, order, do_export=False, save_plot=False, **options):
    """Runs test for a list of refinements and computes error convergence rate"""
    l2_err = []
    for r in ref_list:
        l2_err.append(run(setup, r, order, do_export=do_export, **options))
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
            ax.set_title('tracer adv-diff MMS DG p={:}'.format(order))
            ref_str = 'ref-' + '-'.join([str(r) for r in ref_list])
            order_str = 'o{:}'.format(order)
            imgfile = '_'.join(['convergence', setup_name, field_str, ref_str, order_str])
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

    check_convergence(x_log, y_log, order+1, 'tracer', save_plot)


# ---------------------------
# standard tests for pytest
# ---------------------------

@pytest.fixture(params=['LeapFrog', 'SSPRK22'])
def timestepper_type(request):
    return request.param


@pytest.fixture(params=[Setup1,
                        Setup2,
                        Setup3,
                        Setup4],
                ids=['setup1', 'setup2', 'setup3', 'setup4'])
def setup(request):
    return request.param


def test_convergence(setup, timestepper_type):
    run_convergence(setup, [1, 2, 3], 1, save_plot=False,
                    timestepper_type=timestepper_type)


if __name__ == '__main__':
    run_convergence(Setup4, [1, 2, 3], 1, save_plot=True, timestepper_type='SSPRK22')
