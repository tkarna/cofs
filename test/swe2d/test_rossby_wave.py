"""
'Equatorial Rossby soliton' test case, taken from p.3-6 of [1].

An initial two peak Rossby soliton propagates around a non-dimensionalised periodic equatorial
channel. The test is inviscid and is set up such that the solution at time T=120 should be identical
to the initial condition.

The test case admits approximate solutions by asymptotic expansion, so we can compute error metrics
at the final time to assess how accurately Thetis handles dispersion and nonlinearity. The error
metrics chosen are relative mean peak height, relative mean phase speed and RMS error. In order to
compute the error metrics, we project onto the same high resolution mesh used for FVCOM comparisons.

[1] H. Huang, C. Chen, G.W. Cowles, C.D. Winant, R.C. Beardsley, K.S. Hedstrom and D.B. Haidvogel,
"FVCOM validation experiments: Comparisons with ROMS for three idealized barotropic test problems"
(2008), Journal of Geophysical Research: Oceans, 113(C7).
"""
from thetis import *
import pytest


def asymptotic_expansion_uv(U_2d, order=1, time=0.0, soliton_amplitude=0.395):
    x, y = SpatialCoordinate(U_2d.mesh())

    # Variables for asymptotic expansion
    t = Constant(time)
    B = Constant(soliton_amplitude)
    modon_propagation_speed = -1.0/3.0
    if order != 0:
        assert order == 1
        modon_propagation_speed -= 0.395*B*B
    c = Constant(modon_propagation_speed)
    xi = x - c*t
    psi = exp(-0.5*y*y)
    phi = 0.771*(B/cosh(B*xi))**2
    dphidx = -2*B*phi*tanh(B*xi)
    C = -0.395*B*B

    # Zeroth order terms
    u_terms = phi*0.25*(-9 + 6*y*y)*psi
    v_terms = 2*y*dphidx*psi
    if order == 0:
        return interpolate(as_vector([u_terms, v_terms]), U_2d)

    # Unnormalised Hermite series coefficients for u
    u = np.zeros(28)
    u[0] = 1.7892760e+00
    u[2] = 0.1164146e+00
    u[4] = -0.3266961e-03
    u[6] = -0.1274022e-02
    u[8] = 0.4762876e-04
    u[10] = -0.1120652e-05
    u[12] = 0.1996333e-07
    u[14] = -0.2891698e-09
    u[16] = 0.3543594e-11
    u[18] = -0.3770130e-13
    u[20] = 0.3547600e-15
    u[22] = -0.2994113e-17
    u[24] = 0.2291658e-19
    u[26] = -0.1178252e-21

    # Unnormalised Hermite series coefficients for v
    v = np.zeros(28)
    v[3] = -0.6697824e-01
    v[5] = -0.2266569e-02
    v[7] = 0.9228703e-04
    v[9] = -0.1954691e-05
    v[11] = 0.2925271e-07
    v[13] = -0.3332983e-09
    v[15] = 0.2916586e-11
    v[17] = -0.1824357e-13
    v[19] = 0.4920951e-16
    v[21] = 0.6302640e-18
    v[23] = -0.1289167e-19
    v[25] = 0.1471189e-21

    # Hermite polynomials
    polynomials = [Constant(1.0), 2*y]
    for i in range(2, 28):
        polynomials.append(2*y*polynomials[i-1] - 2*(i-1)*polynomials[i-2])

    # First order terms
    u_terms += C*phi*0.5625*(3 + 2*y*y)*psi
    u_terms += phi*phi*psi*sum(u[i]*polynomials[i] for i in range(28))
    v_terms += dphidx*phi*psi*sum(v[i]*polynomials[i] for i in range(28))

    return interpolate(as_vector([u_terms, v_terms]), U_2d)


def asymptotic_expansion_elev(H_2d, order=1, time=0.0, soliton_amplitude=0.395):
    x, y = SpatialCoordinate(H_2d.mesh())

    # Variables for asymptotic expansion
    t = Constant(time)
    B = Constant(soliton_amplitude)
    modon_propagation_speed = -1.0/3.0
    if order != 0:
        assert order == 1
        modon_propagation_speed -= 0.395*B*B
    c = Constant(modon_propagation_speed)
    xi = x - c*t
    psi = exp(-0.5*y*y)
    phi = 0.771*(B/cosh(B*xi))**2
    C = -0.395*B*B

    # Zeroth order terms
    eta_terms = phi*0.25*(3 + 6*y*y)*psi
    if order == 0:
        return interpolate(eta_terms, H_2d)

    # Unnormalised Hermite series coefficients for eta
    eta = np.zeros(28)
    eta[0] = -3.0714300e+00
    eta[2] = -0.3508384e-01
    eta[4] = -0.1861060e-01
    eta[6] = -0.2496364e-03
    eta[8] = 0.1639537e-04
    eta[10] = -0.4410177e-06
    eta[12] = 0.8354759e-09
    eta[14] = -0.1254222e-09
    eta[16] = 0.1573519e-11
    eta[18] = -0.1702300e-13
    eta[20] = 0.1621976e-15
    eta[22] = -0.1382304e-17
    eta[24] = 0.1066277e-19
    eta[26] = -0.1178252e-21

    # Hermite polynomials
    polynomials = [Constant(1.0), 2*y]
    for i in range(2, 28):
        polynomials.append(2*y*polynomials[i-1] - 2*(i-1)*polynomials[i-2])

    # First order terms
    eta_terms += C*phi*0.5625*(-5 + 2*y*y)*psi
    eta_terms += phi*phi*psi*sum(eta[i]*polynomials[i] for i in range(28))

    return interpolate(eta_terms, H_2d)


def run(refinement_level, **model_options):
    order = model_options.pop('expansion_order')
    family = model_options.get('element_family')
    stepper = model_options.get('timestepper_type')
    print_output("--- running refinement level {:d} in {:s} space".format(refinement_level, family))

    # Set up domain
    lx, ly = 48, 24
    nx, ny = 2*refinement_level, refinement_level
    mesh2d = PeriodicRectangleMesh(nx, ny, lx, ly, direction='x')
    x, y = SpatialCoordinate(mesh2d)
    mesh2d.coordinates.interpolate(as_vector([x-lx/2, y-ly/2]))

    # Physics
    g = physical_constants['g_grav'].values()[0]
    physical_constants['g_grav'].assign(1.0)
    P1_2d = FunctionSpace(mesh2d, "CG", 1)
    bathymetry2d = Function(P1_2d).assign(1.0)

    # Create solver object
    solver_obj = solver2d.FlowSolver2d(mesh2d, bathymetry2d)
    options = solver_obj.options
    options.timestepper_type = stepper
    options.timestep = 0.96/refinement_level if stepper == 'SSPRK33' else 9.6/refinement_level
    options.simulation_export_time = 5.0
    options.simulation_end_time = model_options.get('simulation_end_time')
    options.use_grad_div_viscosity_term = False
    options.use_grad_depth_viscosity_term = False
    options.horizontal_viscosity = None
    solver_obj.create_function_spaces()
    options.coriolis_frequency = interpolate(y, solver_obj.function_spaces.P1_2d)
    options.timestepper_options.solver_parameters['ksp_rtol'] = 1.0e-04
    if hasattr(options.timestepper_options, 'use_automatic_timestep'):
        options.timestepper_options.use_automatic_timestep = False
    options.no_exports = True
    options.update(model_options)

    # Apply boundary conditions
    for tag in mesh2d.exterior_facets.unique_markers:
        solver_obj.bnd_functions['shallow_water'][tag] = {'uv': Constant(as_vector([0., 0.]))}

    # Apply initial conditions (asymptotic solution at initial time)
    uv_a = asymptotic_expansion_uv(solver_obj.function_spaces.U_2d, order=order)
    elev_a = asymptotic_expansion_elev(solver_obj.function_spaces.H_2d, order=order)
    solver_obj.assign_initial_conditions(uv=uv_a, elev=elev_a)

    # Solve PDE
    solver_obj.iterate()
    physical_constants['g_grav'].assign(g)  # Revert g_grav value

    # Get mean peak heights
    elev = interpolate(sign(y)*solver_obj.fields.elev_2d, P1_2d)  # Flip sign in southern hemisphere
    xcoords = interpolate(mesh2d.coordinates[0], P1_2d)
    with elev.dat.vec_ro as v:
        i_n, h_n = v.max()
        i_s, h_s = v.min()

    # Get mean phase speeds
    with xcoords.dat.vec_ro as xdat:
        x_n = xdat[i_n]
        x_s = xdat[i_s]

    # Get relative versions of metrics using high resolution FVCOM data
    h_n /= 0.1567020
    h_s /= -0.1567020  # Flip sign back
    c_n = (48.0 - x_n)/47.18
    c_s = (48.0 - x_s)/47.18
    return h_n, h_s, c_n, c_s


def run_convergence(ref_list, **options):
    """
    Runs test for a sequence of refinements and computes the metric
    convergence rate.

    Note that the metrics should tend to unity, rather than zero. Since
    this could be from above or from below, we assess the quantity

  ..math::
        1 - |1 - m|,

    where :math:`m` is the metric. This quantity has its maximum at unity,
    so must approach from below.
    """
    setup_name = 'rossby-soliton'

    # Compute metrics for each refinement level
    labels = ('h+', 'h-', 'c+', 'c-')
    metrics = {metric: [] for metric in labels}
    for r in ref_list:
        msg = "Error metrics:"
        for metric, value in zip(labels, run(r, **options)):
            metrics[metric].append(value)
            msg += ' {:s} {:6.4f}'.format(metric, value)
        print_output(msg)

    # Check convergence of relative mean peak height and phase speed
    rtol = 0.02
    for m in ('h+', 'h-', 'c+', 'c-'):
        for i in range(1, len(ref_list)):
            slope = (1 - abs(1 - metrics[m][i]))/(1 - abs(1 - metrics[m][i-1]))
            msg = "{:s}: Divergence of error metric {:s}, expected {:.4f} > 1"
            assert slope > 1.0 - rtol, msg.format(setup_name, m, slope)
            print_output("{:s}: error metric {:s} index {:d} PASSED".format(setup_name, m, i))


# ---------------------------
# standard tests for pytest
# ---------------------------

@pytest.fixture(params=['CrankNicolson', 'SSPRK33', 'DIRK22'])
def stepper(request):
    return request.param


@pytest.fixture(params=['dg-dg', 'dg-cg', 'rt-dg', 'bdm-dg'])
def family(request):
    return request.param


def test_convergence(stepper, family):
    run_convergence([24, 48], timestepper_type=stepper,
                    simulation_end_time=30.0, polynomial_degree=1, element_family=family,
                    no_exports=True, expansion_order=1)


# ---------------------------
# run individual setup for debugging
# ---------------------------

if __name__ == '__main__':
    test_convergence('DIRK22', 'bdm-dg')
