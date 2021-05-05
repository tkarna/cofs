from thetis import *
import math
import numpy
import pytest


@pytest.fixture(params=['rt-dg', 'dg-dg', 'dg-cg', 'bdm-dg'])
def element_family(request):
    return request.param


def test_steady_state_channel_mms(element_family, do_exports=False):
    lx = 5e3
    ly = 1e3

    order = 1
    # minimum resolution
    min_cells = 48
    n = 1  # number of timesteps
    dt = 1.
    g = physical_constants['g_grav'].dat.data[0]
    H0 = 10.  # depth at rest
    area = lx*ly

    k = 4.0*math.pi/lx
    Q = H0*1.0  # flux (depth-integrated velocity)
    eta0 = 1.0  # free surface amplitude
    C_D = 0.0025  # quadratic drag coefficient

    xhat = Identity(2)[0, :]

    eta_errs = []
    u_errs = []
    for i in range(4):
        mesh2d = RectangleMesh(min_cells*2**i, 1, lx, ly)
        x = mesh2d.coordinates
        eta_expr = eta0*cos(k*x[0])
        H = H0+eta0*cos(k*x[0])
        u_expr = Q/H
        source_expr = k*eta0*(pow(Q, 2)/pow(H, 3)-g)*sin(k*x[0]) + C_D*abs(u_expr)*(u_expr)/H
        eta_bcval = Constant(eta0)

        # bathymetry
        p1_2d = get_functionspace(mesh2d, 'CG', 1)
        bathymetry_2d = Function(p1_2d, name="bathymetry")
        bathymetry_2d.assign(H0)

        source_space = get_functionspace(mesh2d, 'DG', order+2, vector=True)
        source_func = project(source_expr*xhat, source_space, name="Source")

        # --- create solver ---
        solver_obj = solver2d.FlowSolver2d(mesh2d, bathymetry_2d)
        solver_obj.options.element_family = element_family
        solver_obj.options.polynomial_degree = order
        solver_obj.options.use_nonlinear_equations = True
        solver_obj.options.quadratic_drag_coefficient = Constant(C_D)
        solver_obj.options.simulation_export_time = dt
        solver_obj.options.simulation_end_time = n*dt
        solver_obj.options.momentum_source_2d = source_func
        solver_obj.options.timestepper_type = 'SteadyState'
        solver_obj.options.timestepper_options.solver_parameters = {
            'ksp_type': 'preonly',
            'pc_type': 'lu',
            'pc_factor_mat_solver_type': 'mumps',
            'snes_type': 'newtonls',
        }
        if hasattr(solver_obj.options.timestepper_options, 'use_automatic_timestep'):
            solver_obj.options.timestepper_options.use_automatic_timestep = False
        solver_obj.options.timestep = dt
        solver_obj.options.no_exports = not do_exports

        # boundary conditions
        inflow_tag = 1
        outflow_tag = 2

        inflow_func = Function(p1_2d)
        inflow_func.interpolate(-u_expr)
        inflow_bc = {'un': inflow_func}
        outflow_func = Function(p1_2d)
        outflow_func.interpolate(eta_bcval)
        outflow_bc = {'elev': outflow_func}
        solver_obj.bnd_functions['shallow_water'] = {inflow_tag: inflow_bc, outflow_tag: outflow_bc}
        # parameters['quadrature_degree']=5

        solver_obj.create_equations()
        # hack to avoid picking up prefixed petsc options from other py.test tests:
        solver_obj.create_timestepper()
        solver_obj.timestepper.name = 'test_steady_state_channel_mms'
        solver_obj.timestepper.update_solver()
        solver_obj.assign_initial_conditions(uv=Constant((1.0, 0.0)))

        if do_exports:
            File('source_{}.pvd'.format(i)).write(source_func)
        solver_obj.iterate()

        uv, eta = solver_obj.fields.solution_2d.split()

        eta_ana = project(eta_expr, solver_obj.function_spaces.H_2d)
        if do_exports:
            File('pdiff_{}.pvd'.format(i)).write(project(eta_ana-eta, solver_obj.function_spaces.H_2d, name="diff"))
        eta_l2norm = assemble(pow(eta-eta_ana, 2)*dx)
        eta_errs.append(math.sqrt(eta_l2norm/area))

        u_ana = project(u_expr*xhat, solver_obj.function_spaces.U_2d)
        if do_exports:
            File('udiff_{}.pvd'.format(i)).write(project(u_ana-uv, solver_obj.function_spaces.U_2d, name="diff"))
        u_l2norm = assemble(inner(u_ana-uv, u_ana-uv)*dx)
        u_errs.append(math.sqrt(u_l2norm/area))

    # NOTE: these currently only pass for order==1
    expected_order = order + 1
    eta_errs = numpy.array(eta_errs)
    print_output('eta errors: {:}'.format(eta_errs))
    print_output('convergence: {:} {:}'.format(eta_errs[:-1]/eta_errs[1:], eta_errs[0]/eta_errs[-1]))
    assert(all(eta_errs[:-1]/eta_errs[1:] > 2.**expected_order*0.75))
    assert(eta_errs[0]/eta_errs[-1] > (2.**expected_order)**(len(eta_errs)-1)*0.75)
    print_output("PASSED")

    expected_order = order + 1
    u_errs = numpy.array(u_errs)
    print_output('u errors: {:}'.format(u_errs))
    print_output('convergence: {:} {:}'.format(u_errs[:-1]/u_errs[1:], u_errs[0]/u_errs[-1]))
    assert(all(u_errs[:-1]/u_errs[1:] > 2.**expected_order*0.75))
    assert(u_errs[0]/u_errs[-1] > (2.**expected_order)**(len(u_errs)-1)*0.75)
    print_output("PASSED")


if __name__ == '__main__':
    test_steady_state_channel_mms('dg-cg', False)
