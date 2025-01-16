"""
Two turbine array
=================

A simple two turbine array is positioned in a rectangular channel which experiences a uniform inflow
velocity at the left hand boundary. These turbines act as momentum sinks, since they extract energy
from the system.

The setup corresponds to the 'offset' configuration considered in the numerical experiments section
of [1]. The main contributions of [1] are the derivation of a goal-oriented error estimate for
shallow water modelling and subsequent implementation of mesh adaptation algorithms. An adapted mesh
resulting from this process is used in this test. The mesh is anisotropic in the flow direction.

[1] J.G. Wallwork, N. Barral, S.C. Kramer, D.A. Ham, M.D. Piggott, "Goal-Oriented Error Estimation
    and Mesh Adaptation in Shallow Water Modelling", Springer Nature Applied Sciences, volume 2,
    pp.1053--1063 (2020), DOI: 10.1007/s42452-020-2745-9, URL: https://rdcu.be/b35wZ.
"""
from thetis import *
import thetis.diagnostics as diagnostics
from petsc4py import PETSc
import pytest
import os


def load_mesh():
    """
    Load an anisotropic mesh from file.
    """
    plex = PETSc.DMPlex().create()
    pwd = os.path.dirname(__file__)
    plex.createFromFile(os.path.join(pwd, "anisotropic_plex.h5"))
    return Mesh(plex)


def run(solve_adjoint=False, mesh=None, **model_options):
    mesh2d = mesh or load_mesh()
    P1_2d = FunctionSpace(mesh2d, "CG", 1)

    # physics
    viscosity = Constant(0.5)
    inflow_velocity = Constant(as_vector([0.5, 0.0]))
    depth = 40.0
    drag_coefficient = Constant(0.0025)
    bathymetry2d = Function(P1_2d).assign(depth)

    # create steady state solver object
    solver_obj = solver2d.FlowSolver2d(mesh2d, bathymetry2d)
    options = solver_obj.options
    options.timestep = 20.0
    options.simulation_export_time = 20.0
    options.simulation_end_time = 18.0
    options.swe_timestepper_type = 'SteadyState'
    options.swe_timestepper_options.solver_parameters = {
        'mat_type': 'aij',
        'snes_type': 'newtonls',
        'snes_linesearch_type': 'bt',
        'snes_rtol': 1e-8,
        'snes_max_it': 100,
        'snes_monitor': None,
        'snes_converged_reason': None,
        'ksp_type': 'preonly',
        'ksp_converged_reason': None,
        'pc_type': 'lu',
        'pc_factor_mat_solver_type': 'mumps',
    }
    options.output_directory = 'outputs'
    options.fields_to_export = ['uv_2d', 'elev_2d', 'hessian_2d']
    options.use_grad_div_viscosity_term = False
    options.horizontal_viscosity = viscosity
    options.quadratic_drag_coefficient = drag_coefficient
    options.use_lax_friedrichs_velocity = True
    options.lax_friedrichs_velocity_scaling_factor = Constant(1.0)
    options.use_grad_depth_viscosity_term = False
    options.no_exports = True
    options.update(model_options)
    solver_obj.create_equations()

    # recover Hessian
    if not solve_adjoint:
        if 'hessian_2d' in field_metadata:
            field_metadata.pop('hessian_2d')
        P1t_2d = get_functionspace(mesh2d, 'CG', 1, tensor=True)
        hessian_2d = Function(P1t_2d)
        u_2d = solver_obj.fields.uv_2d[0]
        hessian_calculator = thetis.diagnostics.HessianRecoverer2D(u_2d, hessian_2d)
        solver_obj.add_new_field(hessian_2d,
                                 'hessian_2d',
                                 'Hessian of x-velocity',
                                 'Hessian2d',
                                 preproc_func=hessian_calculator)

    # apply boundary conditions
    solver_obj.bnd_functions['shallow_water'] = {
        1: {'uv': inflow_velocity},  # inflow condition used upstream
        2: {'elev': Constant(0.0)},  # we need to impose a condition on elevation to close system
        3: {'un': Constant(0.0)},    # freeslip on channel walls
    }

    # --- setup turbine array

    L = 1200.0       # domain length
    W = 500.0        # domain width
    D = 18.0         # turbine diameter
    A = pi*(D/2)**2  # turbine area
    S = 8            # turbine separation in x-direction

    # turbine locations
    locs = [(L/2-S*D, W/2-D, D/2), (L/2+S*D, W/2+D, D/2)]

    def bump(mesh, locs, scale=1.0):
        """
        Smooth approximation to indicator function used to represent tidal turbines.
        Scaled bump function for turbines.

        :arg locs: a list of (x, y, r) triples, each of which describing a single turbine in the
                   array. (x, y) gives the centre of the turbine and r gives its radius.
        :kwarg scale: optional scaling parameter which is useful for normalisation.
        """
        x, y = SpatialCoordinate(mesh)
        i = 0
        for j in range(len(locs)):
            x0, y0, r = locs[j]
            expr = scale*exp(1.0 - 1.0/(1.0 - ((x-x0)/r)**2))*exp(1.0 - 1.0/(1.0 - ((y-y0)/r)**2))
            i += conditional(lt((x-x0)**2 + (y-y0)**2, r*r), expr, 0)
        return i

    thrust_coefficient = 0.8
    # NOTE: We include a correction to account for the fact that the thrust coefficient is based
    #       on an upstream velocity, whereas we are using a depth averaged at-the-turbine velocity
    #       (see Kramer and Piggott 2016, eq. (15)).
    correction = 4.0/(1.0 + sqrt(1.0 - A/(depth*D)))**2
    scaling = len(locs)/assemble(bump(solver_obj.function_spaces.P1DG_2d, locs)*dx)
    farm_options = TidalTurbineFarmOptions()
    farm_options.turbine_density = bump(solver_obj.function_spaces.P1DG_2d, locs, scale=scaling)
    farm_options.turbine_options.diameter = D
    C_T = thrust_coefficient * correction
    farm_options.turbine_options.thrust_coefficient = C_T
    solver_obj.options.tidal_turbine_farms['everywhere'] = farm_options

    # apply initial guess of inflow velocity, solve and return number of nonlinear solver iterations
    solver_obj.assign_initial_conditions(uv=inflow_velocity)
    solver_obj.iterate()
    if not solve_adjoint:
        return solver_obj.timestepper.solver.snes.getIterationNumber()

    # quantity of interest: power output
    q_2d = solver_obj.fields.solution_2d
    uv_2d, elev_2d = split(q_2d)
    C_D = 0.5 * C_T * A * farm_options.turbine_density
    J = C_D * dot(uv_2d, uv_2d) ** 1.5 * dx

    # solve adjoint problem
    F = solver_obj.timestepper.F
    V_2d = solver_obj.function_spaces.V_2d
    adj_sol = Function(V_2d)
    dFdq = derivative(F, q_2d, TrialFunction(V_2d))
    dFdq_transpose = adjoint(dFdq)
    dJdq = derivative(J, q_2d, TestFunction(V_2d))
    solve(dFdq_transpose == dJdq, adj_sol)
    return solver_obj, adj_sol


def estimate_error(mesh, **model_options):
    model_options["solve_adjoint"] = True

    # Create a two level mesh hierarchy
    mesh0, mesh1 = MeshHierarchy(mesh, 1)
    tm = TransferManager()

    # Solve both forward and adjoint on both meshes
    solver_obj, a0 = run(mesh=mesh0, **model_options)
    f0 = solver_obj.fields.solution_2d
    P0 = solver_obj.function_spaces.P0_2d
    solver_obj, a1 = run(mesh=mesh1, **model_options)

    # Approximate adjoint error
    V1 = solver_obj.function_spaces.V_2d
    a0plg = Function(V1)
    tm.prolong(a0, a0plg)
    a1err = Function(V1).assign(a1 - a0plg)

    # Compute dual weighted residual
    ei = diagnostics.ShallowWaterDualWeightedResidual2D(solver_obj, a1err)
    ei.solve()

    # Project down to base space
    error = Function(P0, name="Error indicator")
    error.project(ei.error)
    error.interpolate(abs(error))

    # Plot
    if not model_options.get("no_exports", False):
        VTKFile("outputs/forward.pvd").write(*f0.subfunctions)
        a0u, a0eta = a0.subfunctions
        a0u.rename("uv_2d (adjoint)")
        a0eta.rename("elev_2d (adjoint)")
        VTKFile("outputs/adjoint.pvd").write(a0u, a0eta)
        VTKFile("outputs/error.pvd").write(error)

    return f0, a0, error


# ---------------------------
# standard tests for pytest
# ---------------------------

@pytest.fixture(params=['dg-cg', 'dg-dg', 'rt-dg', 'bdm-dg'])
def family(request):
    return request.param


def test_sipg(family):
    snes_it = run(element_family=family)
    expected = 3
    msg = f'snes iterations exceed expected: {snes_it} > {expected}'
    assert snes_it <= expected, msg


def test_dwr(family):
    n = 5
    mesh = RectangleMesh(12 * n, 5 * n, 1200, 500)
    # NOTE: Building a MeshHierarchy on a mesh loaded
    #       from a DMPlex stored as HDF5 appears to
    #       be broken.
    estimate_error(mesh, element_family=family, no_exports=True)

# ---------------------------
# run individual setup for debugging
# ---------------------------


if __name__ == '__main__':
    n = 5
    mesh = RectangleMesh(12 * n, 5 * n, 1200, 500)
    estimate_error(mesh, element_family="dg-cg")
