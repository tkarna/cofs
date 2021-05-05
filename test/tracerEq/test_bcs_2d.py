from thetis import *
import pytest


def fourier_series_solution(mesh, lx, diff_flux, **model_options):
    r"""
    Consider a diffusion problem with a inhomogeneous Neumann condition and zero initial condition:

    .. math::
       c_t = \nu c_{xx}, c_x(0, t) = D, c_x(l, t) = 0, c(x, 0) = 0

    where :math:`D` is the diffusive flux boundary condition imposed, `diff_flux`.

    In order to solve it analytically, we decompose it into two diffusion problems:
     - a diffusion problem with homogeneous Neumann conditions and a nonzero initial condition,

    .. math::
           z_t = \nu z_{xx}, z_x(0, t) = 0, z_x(l, t) = 0, z(x, 0) = -I;

     - and a diffusion problem with homogeneous Neumann conditions and a nonzero source term,

    .. math::
           w_t = \nu w_{xx} + S, w_x(0, t) = 0, w_x(l, t) = 0, w(x, 0) = 0.

    Here :math:`I = I(x)` is set as :math:`I(x) = \alpha(x) D`, where
    :math:`\alpha(x) = -\frac{(l - x)^2}{2l}` and :math:`l` is the horizontal length of the domain.
    The source term :math:`S = S(x, t)` is given by :math:`S = I_t - \nu I_xx = -\nu\frac Dl`.

    Solving the inhomogeneous problem amounts to summing the solutions of the homogeneous problems
    and subtracting :math:`I`.

    The exact solution takes the form of a Fourier series, which we truncate appropriately.
    """
    x, y = SpatialCoordinate(mesh)
    nu = model_options['horizontal_diffusivity']
    time = model_options['simulation_end_time']

    # Initial condition and source term for two homogeneous Neumann problems
    P1 = get_functionspace(mesh, 'CG', 1)
    ic = Function(P1).interpolate(diff_flux*0.5*(lx - x)*(lx - x)/lx)
    source = Constant(-nu*diff_flux/lx, domain=mesh)

    # The solution uses truncated Fourier expansions, meaning we need the following...

    def phi(n):
        return cos(n*pi*x/lx)

    def ic_fourier_coeff(n):
        return assemble(2/lx*ic*phi(n)*dx)

    def source_fourier_coeff(n):
        return assemble(2/lx*source*phi(n)*dx)

    def source_term(n):
        """
        A simple quadrature routine is used to approximate the time integral.
        """
        I = 0
        tau = 0
        dt = 0.05
        while tau < time - 0.5*dt:
            I += exp(-nu*(n*pi/lx)**2*(t-tau))
            tau += dt
        I *= source_fourier_coeff(n)
        return I*phi(n)

    def ic_term(n):
        return ic_fourier_coeff(n)*exp(-nu*(n*pi/lx)**2*time)*phi(n)

    # Assemble truncated Fourier expansion
    sol = Function(P1, name='Fourier expansion')
    num_terms_source = 1  # Only one needed since source is constant
    num_terms_ic = 100
    expr = Constant(0.5*source_fourier_coeff(0)*time)
    expr = expr + Constant(0.5*ic_fourier_coeff(0))
    for k in range(1, num_terms_source):
        expr = expr + source_term(k)
    for k in range(1, num_terms_ic):
        expr = expr + ic_term(k)
    expr -= ic
    sol.interpolate(-expr)

    return sol


def run(refinement, **model_options):

    # Domain
    lx = 10
    ly = 1
    nx = 40*refinement
    ny = 4
    mesh2d = RectangleMesh(nx, ny, lx, ly)
    depth = 40.0

    # Time interval
    dt = 0.1/refinement
    t_end = 1.0
    t_export = 0.1
    model_options['simulation_end_time'] = t_end

    # Bathymetry
    P1_2d = get_functionspace(mesh2d, 'CG', 1)
    bathy_2d = Function(P1_2d, name='Bathymetry')
    bathy_2d.assign(depth)

    # Diffusivity and diffusive flux BC to impose
    nu = Constant(0.1)
    diff_flux = 0.2

    # Solver
    solver_obj = solver2d.FlowSolver2d(mesh2d, bathy_2d)
    options = solver_obj.options
    options.output_directory = 'outputs'
    options.no_exports = True
    options.timestep = dt
    options.simulation_export_time = t_export
    options.solve_tracer = True
    options.tracer_only = True
    options.horizontal_diffusivity = nu
    options.horizontal_diffusivity_scale = nu
    options.horizontal_velocity_scale = Constant(0.0)
    options.fields_to_export = ['tracer_2d']
    options.update(model_options)
    options.use_limiter_for_tracers = options.tracer_element_family == 'dg'
    options.use_supg_tracer = options.tracer_element_family == 'cg'
    options.simulation_end_time = t_end - 0.5*dt

    # Boundary conditions
    solver_obj.bnd_functions['tracer'] = {1: {'diff_flux': diff_flux*nu}}
    # NOTE: Zero diff_flux boundaries are enforced elsewhere by default

    # Run model
    solver_obj.assign_initial_conditions()
    solver_obj.iterate()
    sol = solver_obj.fields.tracer_2d

    # Get truncated Fourier series solution
    fsol = fourier_series_solution(mesh2d, lx, diff_flux, **model_options)
    if not options.no_exports:
        File('outputs/fourier_series_solution.pvd').write(fsol)

    return errornorm(sol, fsol)


def run_convergence(**model_options):
    errors = []
    for refinement in (1, 2, 4):
        errors.append(run(refinement, **model_options))
    msg = "Wrong convergence rate {:.4f}, expected 2.0000."
    slope = errors[0]/errors[1]
    assert slope > 2, msg.format(slope)
    slope = errors[1]/errors[2]
    assert slope > 2, msg.format(slope)


@pytest.fixture(params=['dg', 'cg'])
def family(request):
    return request.param


@pytest.fixture(params=[1])
def polynomial_degree(request):
    return request.param


@pytest.fixture(params=['CrankNicolson', 'SSPRK33', 'ForwardEuler', 'BackwardEuler', 'DIRK22', 'DIRK33'])
def stepper(request):
    return request.param


@pytest.mark.parametrize(('diffusivity'),
                         [(Constant(0.1))])
def test_horizontal_advection(polynomial_degree, stepper, diffusivity, family):
    run_convergence(polynomial_degree=polynomial_degree,
                    timestepper_type=stepper,
                    horizontal_diffusivity=diffusivity,
                    tracer_element_family=family)


if __name__ == '__main__':
    run_convergence(polynomial_degree=1,
                    timestepper_type='CrankNicolson',
                    horizontal_diffusivity=Constant(0.1),
                    tracer_element_family='cg',
                    no_exports=False)
