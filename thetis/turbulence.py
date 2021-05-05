r"""
Generic Length Scale Turbulence Closure model
=============================================

This model solves two dynamic equations, for turbulent kinetic energy
(TKE, :math:`k`) and one for an additional variable, the generic length scale
:math:`\psi` [1]:

.. math::
    \frac{\partial k}{\partial t} + \nabla_h \cdot (\textbf{u} k)
        + \frac{\partial (w k)}{\partial z}
        = \frac{\partial}{\partial z}\left(\frac{\nu}{\sigma_k} \frac{\partial k}{\partial z}\right)
        + P + B - \varepsilon
    :label: turb_tke_eq

.. math::
    \frac{\partial \psi}{\partial t} + \nabla_h \cdot (\textbf{u} \psi)
        + \frac{\partial (w \psi)}{\partial z}
        = \frac{\partial}{\partial z}\left(\frac{\nu}{\sigma_\psi} \frac{\partial \psi}{\partial z}\right)
        + \frac{\psi}{k} (c_1 P + c_3 B - c_2 \varepsilon f_{wall})
    :label: turb_psi_eq

with the production :math:`P` and buoyancy production :math:`B` are

.. math::
    P &= \nu M^2 \\
    B &= -\mu N^2

where :math:`M` and :math:`N` are the shear and buoyancy frequencies

.. math::
    M^2 &= \left(\frac{\partial u}{\partial z}\right)^2
        + \left(\frac{\partial v}{\partial z}\right)^2 \\
    N^2 &= -\frac{g}{\rho_0}\frac{\partial \rho}{\partial z}

The generic length scale variable is defined as

.. math::
    \psi = (c_\mu^0)^p k^m l^n

where :math:`p, m, n` are parameters and :math:`c_\mu^0` is an empirical constant.

The parameters :math:`c_1,c_2,c_3,f_{wall}` depend on the chosen closure.
The parameter :math:`c_3` takes two values: :math:`c_3^-` in stably stratified
regime, and :math:`c_3^+` in unstably stratified cases.

Turbulent length scale :math:`l`, and the TKE dissipation rate
:math:`\varepsilon` are obtained diagnostically as

.. math::
    l &= (c_\mu^0)^3 k^{3/2} \varepsilon^{-1} \\
    \varepsilon &= (c_\mu^0)^{3+p/n} k^{3/2 + m/n} \psi^{-1/n}

Finally the vertical eddy viscosity and diffusivity are also computed
diagnostically

.. math::
    \nu &= \sqrt{2k} l S_m \\
    \mu &= \sqrt{2k} l S_\rho

Stability functions :math:`S_m` and :math:`S_\rho` are defined in [2] or [3].
Implementation follows [4].

Supported closures
------------------

The GLS model parameters are controlled via the :class:`.GLSModelOptions` class.

The parameters can be accessed from the solver object:

.. code-block:: python

    solver = FlowSolver(...)
    solver.options.turbulence_model_type = 'gls'  # activate GLS model (default)
    turbulence_model_options = solver.options.turbulence_model_options
    turbulence_model_options.closure_name = 'k-omega'
    turbulence_model_options.stability_function_name = 'CB'
    turbulence_model_options.compute_c3_minus = True

Currently the following closures have been implemented:

- :math:`k-\varepsilon` model
    This closure is obtained with :math:`p=3, m=3/2, n=-1`, resulting in
    :math:`\psi=\varepsilon`.
    To use this option set ``closure_name = k-epsilon``
- :math:`k-\omega` model
    This closure is obtained with :math:`p=-1, m=1/2, n=-1`, resulting in
    :math:`\psi=\omega`.
    To use this option set ``closure_name = k-omega``
- GLS model A
    This closure is obtained with :math:`p=2, m=1, n=-2/3`, resulting in
    :math:`\psi=\omega`.
    To use this option set ``closure_name = gen``

The following stability functions have been implemented

- Canuto A [3]
    To use this option set ``closure_name = CA``
- Canuto B [3]
    To use this option set ``closure_name = CB``
- Kantha-Clayson [2]
    To use this option set ``closure_name = KC``
- Cheng [6]
    To use this option set ``closure_name = CH``

See :mod:`.stability_functions` for more information.

[1] Umlauf, L. and Burchard, H. (2003). A generic length-scale equation for
    geophysical turbulence models. Journal of Marine Research, 61:235--265(31).
    http://dx.doi.org/10.1357/002224003322005087

[2] Kantha, L. H. and Clayson, C. A. (1994). An improved mixed layer model for
    geophysical applications. Journal of Geophysical Research: Oceans,
    99(C12):25235--25266.
    http://dx.doi.org/10.1029/94JC02257

[3] Canuto et al. (2001). Ocean Turbulence. Part I: One-Point Closure Model -
    Momentum and Heat Vertical Diffusivities. Journal of Physical Oceanography,
    31(6):1413-1426.
    http://dx.doi.org/10.1175/1520-0485(2001)031

[4] Warner et al. (2005). Performance of four turbulence closure models
    implemented using a generic length scale method. Ocean Modelling,
    8(1-2):81--113.
    http://dx.doi.org/10.1016/j.ocemod.2003.12.003

[5] Umlauf, L. and Burchard, H. (2005). Second-order turbulence closure models
    for geophysical boundary layers. A review of recent work. Continental Shelf
    Research, 25(7-8):795--827.
    http://dx.doi.org/10.1016/j.csr.2004.08.004

[6] Cheng et al. (2002). An improved model for the turbulent PBL.
    J. Atmos. Sci., 59:1550-1565.
    http://dx.doi.org/10.1175/1520-0469(2002)059%3C1550:AIMFTT%3E2.0.CO;2

[7] Burchard and Petersen (1999). Models of turbulence in the marine
    environment - a comparative study of two-equation turbulence models.
    Journal of Marine Systems, 21(1-4):29-53.
    http://dx.doi.org/10.1016/S0924-7963(99)00004-4
"""
from __future__ import absolute_import
from .utility import *
from .equation import Equation
from .tracer_eq import *
from .stability_functions import *
from .log import *
from .options import GLSModelOptions, PacanowskiPhilanderModelOptions


def set_func_min_val(f, minval):
    """
    Sets a minimum value to a :class:`Function`
    """
    f.dat.data[f.dat.data < minval] = minval


def set_func_max_val(f, maxval):
    """
    Sets a minimum value to a :class:`Function`
    """
    f.dat.data[f.dat.data > maxval] = maxval


class VerticalGradSolver(object):
    """
    Computes vertical gradient in the weak sense.

    """
    def __init__(self, source, solution, solver_parameters=None):
        """
        :arg source: A :class:`Function` or expression to differentiate.
        :arg solution: A :class:`Function` where the solution will be stored.
            Must be in P0 space.
        :kwarg dict solver_parameters: PETSc solver options
        """
        if solver_parameters is None:
            solver_parameters = {}
        solver_parameters.setdefault('snes_type', 'ksponly')
        solver_parameters.setdefault('ksp_type', 'preonly')
        solver_parameters.setdefault('pc_type', 'bjacobi')
        solver_parameters.setdefault('sub_ksp_type', 'preonly')
        solver_parameters.setdefault('sub_pc_type', 'ilu')

        self.source = source
        self.solution = solution

        self.fs = self.solution.function_space()
        self.mesh = self.fs.mesh()

        # weak gradient evaluator
        test = TestFunction(self.fs)
        tri = TrialFunction(self.fs)
        normal = FacetNormal(self.mesh)
        a = inner(test, tri)*dx
        p = self.source
        l = -inner(p, Dx(test, 2))*dx
        l += avg(p)*jump(test, normal[2])*dS_h
        l += p*test*normal[2]*(ds_t + ds_b)
        prob = LinearVariationalProblem(a, l, self.solution, constant_jacobian=True)
        self.weak_grad_solver = LinearVariationalSolver(prob, solver_parameters=solver_parameters)

    def solve(self):
        """Computes the gradient"""
        self.weak_grad_solver.solve()


class ShearFrequencySolver(object):
    r"""
    Computes vertical shear frequency squared form the given horizontal
    velocity field.

    .. math::
        M^2 = \left(\frac{\partial u}{\partial z}\right)^2
            + \left(\frac{\partial v}{\partial z}\right)^2
    """
    def __init__(self, uv, m2, mu, mv, mu_tmp, relaxation=1.0, minval=1e-12):
        """
        :arg uv: horizontal velocity field
        :type uv: :class:`Function`
        :arg m2: :math:`M^2` field
        :type m2: :class:`Function`
        :arg mu: field for x component of :math:`M^2`
        :type mu: :class:`Function`
        :arg mv: field for y component of :math:`M^2`
        :type mv: :class:`Function`
        :arg mu_tmp: temporary field
        :type mu_tmp: :class:`Function`
        :kwarg float relaxation: relaxation coefficient for mixing old and new values
            M2 = relaxation*M2_new + (1-relaxation)*M2_old
        :kwarg float minval: minimum value for :math:`M^2`
        """
        self.mu = mu
        self.mv = mv
        self.m2 = m2
        self.mu_tmp = mu_tmp
        self.minval = minval
        self.relaxation = relaxation

        self.var_solvers = []
        for i_comp in range(2):
            self.var_solvers.append(VerticalGradSolver(uv[i_comp], mu_tmp))

    def solve(self, init_solve=False):
        """
        Computes buoyancy frequency

        :kwarg bool init_solve: Set to True if solving for the first time, skips
            relaxation
        """
        with timed_stage('shear_freq_solv'):
            mu_comp = [self.mu, self.mv]
            self.m2.assign(0.0)
            for i_comp, solver in enumerate(self.var_solvers):
                solver.solve()
                gamma = self.relaxation if not init_solve else 1.0
                mu_comp[i_comp].assign(gamma*self.mu_tmp
                                       + (1.0 - gamma)*mu_comp[i_comp])
                self.m2 += mu_comp[i_comp]*mu_comp[i_comp]
            # crop small/negative values
            set_func_min_val(self.m2, self.minval)


class BuoyFrequencySolver(object):
    r"""
    Computes buoyancy frequency squared form the given horizontal
    velocity field.

    .. math::
        N^2 = -\frac{g}{\rho_0}\frac{\partial \rho}{\partial z}
    """
    def __init__(self, rho, n2, n2_tmp, relaxation=1.0, minval=1e-12):
        """
        :arg rho: water density field
        :type rho: :class:`Function`
        :arg n2: :math:`N^2` field
        :type n2: :class:`Function`
        :arg n2_tmp: temporary field
        :type n2_tmp: :class:`Function`
        :kwarg float relaxation: relaxation coefficient for mixing old and new
            values N2 = relaxation*N2_new + (1-relaxation)*N2_old
        :kwarg float minval: minimum value for :math:`N^2`
        """
        self._no_op = False
        if rho is None:
            self._no_op = True

        if not self._no_op:

            self.n2 = n2
            self.n2_tmp = n2_tmp
            self.relaxation = relaxation

            g = physical_constants['g_grav']
            rho0 = physical_constants['rho0']
            p = -g/rho0 * rho
            solver = VerticalGradSolver(p, self.n2_tmp)
            self.var_solver = solver

    def solve(self, init_solve=False):
        """
        Computes buoyancy frequency

        :kwarg bool init_solve: Set to True if solving for the first time, skips
            relaxation
        """
        with timed_stage('buoy_freq_solv'):
            if not self._no_op:
                self.var_solver.solve()
                gamma = self.relaxation if not init_solve else 1.0
                self.n2.assign(gamma*self.n2_tmp
                               + (1.0 - gamma)*self.n2)


class TurbulenceModel(object):
    """Base class for all vertical turbulence models"""

    @abstractmethod
    def initialize(self):
        """Initialize all turbulence fields"""
        pass

    @abstractmethod
    def preprocess(self, init_solve=False):
        """
        Computes all diagnostic variables that depend on the mean flow model
        variables.

        To be called before updating the turbulence PDEs.
        """
        pass

    @abstractmethod
    def postprocess(self):
        """
        Updates all diagnostic variables that depend on the turbulence state
        variables.

        To be called after updating the turbulence PDEs.
        """
        pass


class GenericLengthScaleModel(TurbulenceModel):
    """
    Generic Length Scale turbulence closure model implementation
    """
    def __init__(self, solver, k_field, psi_field, uv_field, rho_field,
                 l_field, epsilon_field,
                 eddy_diffusivity, eddy_viscosity,
                 n2, m2, options=None):
        """
        :arg solver: FlowSolver object
        :arg k_field: turbulent kinetic energy (TKE) field
        :type k_field: :class:`Function`
        :arg psi_field: generic length scale field
        :type psi_field: :class:`Function`
        :arg uv_field: horizontal velocity field
        :type uv_field: :class:`Function`
        :arg rho_field: water density field
        :type rho_field: :class:`Function`
        :arg l_field: turbulence length scale field
        :type l_field: :class:`Function`
        :arg epsilon_field: TKE dissipation rate field
        :type epsilon_field: :class:`Function`
        :arg eddy_diffusivity: eddy diffusivity field
        :type eddy_diffusivity: :class:`Function`
        :arg eddy_viscosity: eddy viscosity field
        :type eddy_viscosity: :class:`Function`
        :arg n2: field for buoyancy frequency squared
        :type n2: :class:`Function`
        :arg m2: field for vertical shear frequency squared
        :type m2: :class:`Function`
        :kwarg options: GLS model options
        """
        self.solver = solver
        # 3d model fields
        self.uv = uv_field
        self.rho = rho_field
        # prognostic fields
        self.k = k_field
        self.psi = psi_field
        # diagnostic fields
        # NOTE redundant for k-epsilon model where psi==epsilon
        self.epsilon = epsilon_field
        self.l = l_field
        self.viscosity = eddy_viscosity
        self.diffusivity = eddy_diffusivity
        self.n2 = n2
        self.m2 = m2
        self.mu_tmp = Function(self.m2.function_space(),
                               name='tmp Shear frequency')
        self.mu = Function(self.m2.function_space(), name='Shear frequency X')
        self.mv = Function(self.m2.function_space(), name='Shear frequency Y')
        self.n2_tmp = Function(self.n2.function_space(),
                               name='tmp buoyancy frequency')
        self.n2_pos = Function(self.n2.function_space(),
                               name='positive buoyancy frequency')
        self.n2_neg = Function(self.n2.function_space(),
                               name='negative buoyancy frequency')

        # parameter to mix old and new viscosity values (1 => new only)
        self.relaxation = 1.0

        if options is not None:
            self.options = options
        else:
            self.options = GLSModelOptions()

        o = self.options
        stability_function_name = o.stability_function_name
        stab_args = {'lim_alpha_shear': True,
                     'lim_alpha_buoy': True,
                     'smooth_alpha_buoy_lim': False}
        if stability_function_name == 'Kantha-Clayson':
            self.stability_func = StabilityFunctionKanthaClayson(**stab_args)
        elif stability_function_name == 'Canuto A':
            self.stability_func = StabilityFunctionCanutoA(**stab_args)
        elif stability_function_name == 'Canuto B':
            self.stability_func = StabilityFunctionCanutoB(**stab_args)
        elif stability_function_name == 'Cheng':
            self.stability_func = StabilityFunctionCheng(**stab_args)
        else:
            raise Exception('Unknown stability function type: ' + stability_function_name)

        if o.compute_cmu0:
            o.cmu0 = self.stability_func.compute_cmu0()
        if o.compute_kappa:
            kappa = self.stability_func.compute_kappa(o.schmidt_nb_psi, o.cmu0, o.n, o.c1, o.c2)
            o.kappa = kappa
            # update mean flow model value as well
            physical_constants['von_karman'].assign(kappa)
        elif o.compute_schmidt_nb_psi:
            o.schmidt_nb_psi = self.stability_func.compute_sigma_psi(o.kappa, o.cmu0, o.n, o.c1, o.c2)
        if o.compute_c3_minus:
            o.c3_minus = self.stability_func.compute_c3_minus(o.c1, o.c2, o.ri_st)
        if o.compute_psi_min:
            o.psi_min = (o.cmu0**(3.0 + o.p/o.n)*o.k_min**(3.0/2.0 + o.m/o.n)*o.eps_min**(-1.0))**o.n
        # minimum length scale
        if o.compute_len_min:
            o.len_min = o.cmu0**3 * o.k_min**1.5 / o.eps_min
        if o.compute_galperin_clim:
            o.galperin_clim = self.stability_func.compute_length_clim(o.cmu0, o.ri_st)

        self.shear_frequency_solver = ShearFrequencySolver(self.uv, self.m2,
                                                           self.mu, self.mv,
                                                           self.mu_tmp)
        if self.rho is not None:
            self.buoy_frequency_solver = BuoyFrequencySolver(self.rho, self.n2,
                                                             self.n2_tmp)
        print_output(self.options)
        self._initialized = False

    def initialize(self, l_init=0.01):
        """
        Initialize turbulent fields.

        k is set to k_min value. epsilon and psi are set to a value that
        corresponds to l_init length scale.

        :arg l_init: initial value for turbulent length scale.
        """
        if not self.solver._simulation_continued:
            o = self.options
            eps_init = o.cmu0**3.0 * o.k_min**(3.0/2.0) / l_init
            psi_init = (o.cmu0**(3.0 + o.p/o.n)*o.k_min**(3.0/2.0 + o.m/o.n)*eps_init**(-1.0))**o.n
            self.k.assign(self.options.k_min)
            self.psi.assign(psi_init)
            self.epsilon.assign(eps_init)
        self.n2.assign(1e-12)
        self.n2_pos.assign(1e-12)
        self.n2_neg.assign(0.0)
        self._initialized = True
        self.preprocess(init_solve=True)
        self.postprocess()

    def preprocess(self, init_solve=False):
        """
        Computes all diagnostic variables that depend on the mean flow model
        variables.

        To be called before updating the turbulence PDEs.
        """

        if self._initialized is False:
            self.initialize()

        self.shear_frequency_solver.solve(init_solve=init_solve)

        if self.rho is not None:
            self.buoy_frequency_solver.solve(init_solve=init_solve)
            # split to positive and negative parts
            self.n2_pos.assign(1e-12)
            self.n2_neg.assign(0.0)
            pos_ix = self.n2.dat.data[:] >= 0.0
            self.n2_pos.dat.data[pos_ix] = self.n2.dat.data[pos_ix]
            self.n2_neg.dat.data[~pos_ix] = self.n2.dat.data[~pos_ix]

    def postprocess(self):
        r"""
        Updates all diagnostic variables that depend on the turbulence state
        variables :math:`k,\psi`.

        To be called after updating the turbulence PDEs.
        """
        with timed_stage('turb_postproc'):
            o = self.options
            cmu0 = o.cmu0
            p = o.p
            n = o.n
            m = o.m

            # limit k
            set_func_min_val(self.k, o.k_min)

            k_arr = self.k.dat.data[:]
            n2_pos = self.n2_pos.dat.data[:]
            n2_pos_eps = 1e-12
            galp_clim = o.galperin_clim
            if o.limit_psi:
                # impose Galperin limit on psi
                # psi^(1/n) <= sqrt(0.56)* (cmu0)^(p/n) *k^(m/n+0.5)* n2^(-0.5)
                val = (np.sqrt(2)*galp_clim * (cmu0)**(p / n) * k_arr**(m / n + 0.5) * (n2_pos + n2_pos_eps)**(-0.5))**n
                if n > 0:
                    # impose max value
                    np.minimum(self.psi.dat.data, val, self.psi.dat.data)
                else:
                    # impose min value
                    np.maximum(self.psi.dat.data, val, self.psi.dat.data)
            set_func_min_val(self.psi, o.psi_min)

            # udpate epsilon
            self.epsilon.assign(cmu0**(3.0 + p/n)*self.k**(3.0/2.0 + m/n)*self.psi**(-1.0/n))
            if o.limit_eps:
                # impose Galperin limit on eps
                eps_min = cmu0**3.0/(np.sqrt(2)*galp_clim)*np.sqrt(n2_pos)*k_arr
                np.maximum(self.epsilon.dat.data, eps_min, self.epsilon.dat.data)
            # impose minimum value
            set_func_min_val(self.epsilon, o.eps_min)

            # update L
            self.l.assign(cmu0**3.0 * self.k**(3.0/2.0) / self.epsilon)
            if o.limit_len_min:
                set_func_min_val(self.l, o.len_min)
            if o.limit_len:
                # Galperin length scale limitation
                len_max = galp_clim*np.sqrt(2*k_arr/(n2_pos + n2_pos_eps))
                np.minimum(self.l.dat.data, len_max, self.l.dat.data)
            if self.l.dat.data.max() > 10.0:
                warning(' * large L: {:f}'.format(self.l.dat.data.max()))

            # update stability functions
            s_m, s_h = self.stability_func.evaluate(self.m2.dat.data,
                                                    self.n2.dat.data,
                                                    self.k.dat.data,
                                                    self.epsilon.dat.data)
            # update diffusivity/viscosity
            b = np.sqrt(self.k.dat.data[:])*self.l.dat.data[:]
            lam = self.relaxation
            new_visc = b*s_m/cmu0**3
            new_diff = b*s_h/cmu0**3
            self.viscosity.dat.data[:] = lam*new_visc + (1.0 - lam)*self.viscosity.dat.data[:]
            self.diffusivity.dat.data[:] = lam*new_diff + (1.0 - lam)*self.diffusivity.dat.data[:]

            set_func_min_val(self.viscosity, o.visc_min)
            set_func_min_val(self.diffusivity, o.diff_min)

    def print_debug(self):
        """
        Print diagnostic field values for debugging.
        """
        fmt = '{:8s} {:10.3e} {:10.3e}'
        print_output(' -----')
        print_output(fmt.format('k', self.k.dat.data.min(), self.k.dat.data.max()))
        print_output(fmt.format('psi', self.psi.dat.data.min(), self.psi.dat.data.max()))
        print_output(fmt.format('eps', self.epsilon.dat.data.min(), self.epsilon.dat.data.max()))
        print_output(fmt.format('L', self.l.dat.data.min(), self.l.dat.data.max()))
        print_output(fmt.format('M2', self.m2.dat.data.min(), self.m2.dat.data.max()))
        print_output(fmt.format('N2', self.n2.dat.data.min(), self.n2.dat.data.max()))
        print_output(fmt.format('N2+', self.n2_pos.dat.data.min(), self.n2_pos.dat.data.max()))
        print_output(fmt.format('N2-', self.n2_neg.dat.data.min(), self.n2_neg.dat.data.max()))
        print_output(fmt.format('s_h', s_h.min(), s_h.max()))
        print_output(fmt.format('s_m', s_m.min(), s_m.max()))
        print_output(fmt.format('nuv', self.viscosity.dat.data.min(), self.viscosity.dat.data.max()))
        print_output(fmt.format('muv', self.diffusivity.dat.data.min(), self.diffusivity.dat.data.max()))


class TKESourceTerm(TracerTerm):
    r"""
    Production and destruction terms of the TKE equation :eq:`turb_tke_eq`

    .. math::
        F_k = P + B - \varepsilon

    To ensure positivity we use Patankar-type time discretization: all source
    terms are treated explicitly and sink terms are treated implicitly.
    To this end the buoyancy production term :math:`B` is split in two:

    .. math::
        F_k = P + B_{source} + \frac{k^{n+1}}{k^n}(B_{sink} - \varepsilon)

    with :math:`B_{source} \ge 0` and :math:`B_{sink} < 0`.
    """
    def __init__(self, function_space, gls_model,
                 bathymetry=None, v_elem_size=None, h_elem_size=None):
        """
        :arg function_space: :class:`FunctionSpace` where the solution belongs
        :arg gls_model: :class:`.GenericLengthScaleModel` object
        :kwarg bathymetry: bathymetry of the domain
        :type bathymetry: 3D :class:`Function` or :class:`Constant`
        :kwarg v_elem_size: scalar :class:`Function` that defines the vertical
            element size
        :kwarg h_elem_size: scalar :class:`Function` that defines the horizontal
            element size
        """
        super(TKESourceTerm, self).__init__(function_space,
                                            bathymetry, v_elem_size, h_elem_size)
        self.gls_model = gls_model

    def residual(self, solution, solution_old, fields, fields_old, bnd_conditions=None):
        # TKE: P + B - eps
        # P = viscosity M**2           (production)
        # B = - diffusivity N**2       (byoyancy production)
        # M**2 = (du/dz)**2 + (dv/dz)**2 (shear frequency)
        # N**2 = -g\rho_0 (drho/dz)      (buoyancy frequency)
        # eps = (cmu0)**(3+p/n)*tke**(3/2+m/n)*psi**(-1/n)
        #                                (tke dissipation rate)
        eddy_viscosity = fields_old['viscosity_v']
        eddy_diffusivity = fields_old['diffusivity_v']
        epsilon = fields_old['epsilon']
        shear_freq2 = fields_old['shear_freq2']
        buoy_freq2_neg = fields_old['buoy_freq2_neg']
        buoy_freq2_pos = fields_old['buoy_freq2_pos']
        p = eddy_viscosity * shear_freq2
        b_source = - eddy_diffusivity * buoy_freq2_neg
        b_sink = - eddy_diffusivity * buoy_freq2_pos

        source = p + b_source + (b_sink - epsilon)/solution_old*solution  # patankar
        f = inner(source, self.test)*dx
        return f


class PsiSourceTerm(TracerTerm):
    r"""
    Production and destruction terms of the Psi equation :eq:`turb_psi_eq`

    .. math::
        F_\psi = \frac{\psi}{k} (c_1 P + c_3 B - c_2 \varepsilon f_{wall})

    To ensure positivity we use Patankar-type time discretization: all source
    terms are treated explicitly and sink terms are treated implicitly.
    To this end the buoyancy production term :math:`c_3 B` is split in two:

    .. math::
        F_\psi = \frac{\psi^n}{k^n} (c_1 P + B_{source})
            + \frac{\psi^{n+1}}{k^n} (B_{sink} - c_2 \varepsilon f_{wall})

    with :math:`B_{source} \ge 0` and :math:`B_{sink} < 0`.

    Also implements Neumann boundary condition at top and bottom [7]

    .. math::
        \left( \frac{\nu}{\sigma_\psi} \frac{\psi}{z} \right)\Big|_{\Gamma_b} =
        n_z \frac{\nu}{\sigma_\psi} (c_\mu^0)^p k^m \kappa^n (z_b + z_0)^{n-1}

    where :math:`z_b` is the distance from boundary, and :math:`z_0` is the
    roughness length.
    """
    def __init__(self, function_space, gls_model,
                 bathymetry=None, v_elem_size=None, h_elem_size=None):
        """
        :arg function_space: :class:`FunctionSpace` where the solution belongs
        :arg gls_model: :class:`.GenericLengthScaleModel` object
        :kwarg bathymetry: bathymetry of the domain
        :type bathymetry: 3D :class:`Function` or :class:`Constant`
        :kwarg v_elem_size: scalar :class:`Function` that defines the vertical
            element size
        :kwarg h_elem_size: scalar :class:`Function` that defines the horizontal
            element size
        """
        super(PsiSourceTerm, self).__init__(function_space,
                                            bathymetry, v_elem_size, h_elem_size)
        self.gls_model = gls_model

    def residual(self, solution, solution_old, fields, fields_old, bnd_conditions=None):
        # psi: psi/k*(c1*P + c3*B - c2*eps*f_wall)
        # P = viscosity M**2           (production)
        # B = - diffusivity N**2       (byoyancy production)
        # M**2 = (du/dz)**2 + (dv/dz)**2 (shear frequency)
        # N**2 = -g\rho_0 (drho/dz)      (buoyancy frequency)
        # eps = (cmu0)**(3+p/n)*tke**(3/2+m/n)*psi**(-1/n)
        #                                (tke dissipation rate)
        eddy_viscosity = fields_old['viscosity_v']
        eddy_diffusivity = fields_old['diffusivity_v']
        diffusivity_v = eddy_viscosity/self.gls_model.options.schmidt_nb_psi
        k = fields_old['k']
        epsilon = fields_old['epsilon']
        shear_freq2 = fields_old['shear_freq2']
        buoy_freq2_neg = fields_old['buoy_freq2_neg']
        buoy_freq2_pos = fields_old['buoy_freq2_pos']
        p = eddy_viscosity * shear_freq2
        c1 = self.gls_model.options.c1
        c2 = self.gls_model.options.c2
        # c3 switch: c3 = c3_minus if n2 > 0 else c3_plus
        c3_minus = self.gls_model.options.c3_minus
        c3_plus = self.gls_model.options.c3_plus  # > 0
        assert c3_plus >= 0, 'c3_plus has unexpected sign'
        b_shear = c3_plus * -eddy_diffusivity * buoy_freq2_neg
        b_buoy = c3_minus * -eddy_diffusivity * buoy_freq2_pos
        if c3_minus > 0:
            b_source = b_shear
            b_sink = b_buoy
        else:
            b_source = b_shear + b_buoy
            b_sink = 0
        f_wall = self.gls_model.options.f_wall
        source = solution_old/k*(c1*p + b_source) + solution/k*(b_sink - c2*f_wall*epsilon)  # patankar
        f = inner(source, self.test)*dx

        # add bottom/top boundary condition for psi
        # (nuv_v/sigma_psi * dpsi/dz)_b = n * nuv_v/sigma_psi * (cmu0)^p * k^m * kappa^n * z_b^(n-1)
        # z_b = distance_from_bottom + z_0
        cmu0 = self.gls_model.options.cmu0
        p = self.gls_model.options.p
        m = self.gls_model.options.m
        n = self.gls_model.options.n
        bfr_roughness = fields_old.get('bottom_roughness')
        if bfr_roughness is None:
            bfr_roughness = Constant(0)
        kappa = physical_constants['von_karman']
        if self.v_elem_size is None:
            raise Exception('v_elem_size required')
        # bottom condition
        elem_frac = Constant(0.5)
        z_b = elem_frac*self.v_elem_size + bfr_roughness
        diff_flux = (n*diffusivity_v*(cmu0)**p
                     * k**m * kappa**n * z_b**(n - 1.0))
        f += diff_flux*self.test*self.normal[2]*ds_bottom
        # surface condition
        elem_frac = Constant(0.5)
        z0_surface = Constant(0.05)  # TODO generalize
        z_s = elem_frac*self.v_elem_size + z0_surface
        diff_flux = -(n*diffusivity_v*(cmu0)**p
                      * k**m * kappa**n * z_s**(n - 1.0))
        f += diff_flux*self.test*self.normal[2]*ds_surf

        return f


class GLSVerticalDiffusionTerm(VerticalDiffusionTerm):
    """
    Vertical diffusion term where the diffusivity is replaced by
    viscosity/Schmidt number.
    """
    def __init__(self, function_space, schmidt_nb,
                 bathymetry=None, v_elem_size=None, h_elem_size=None,
                 sipg_factor=Constant(1.0)):
        """
        :arg function_space: :class:`FunctionSpace` where the solution belongs
        :arg schmidt_nb: the Schmidt number of TKE or Psi
        :kwarg bathymetry: bathymetry of the domain
        :type bathymetry: 3D :class:`Function` or :class:`Constant`
        :kwarg v_elem_size: scalar :class:`Function` that defines the vertical
            element size
        :kwarg h_elem_size: scalar :class:`Function` that defines the horizontal
            element size
        :kwarg sipg_factor: :class: `Constant` or :class: `Function` vertical SIPG penalty scaling factor

        """
        super(GLSVerticalDiffusionTerm, self).__init__(function_space,
                                                       bathymetry, v_elem_size, h_elem_size,
                                                       sipg_factor_vertical=sipg_factor)
        self.schmidt_nb = schmidt_nb

    def residual(self, solution, solution_old, fields, fields_old, bnd_conditions=None):
        d = {'diffusivity_v': fields_old['viscosity_v']/self.schmidt_nb}
        f = super(GLSVerticalDiffusionTerm, self).residual(solution, solution_old,
                                                           d, d, bnd_conditions=None)
        return f


class TKEEquation(Equation):
    """
    Turbulent kinetic energy equation :eq:`turb_tke_eq` without advection terms.

    Advection of TKE is implemented using the standard tracer equation.
    """
    def __init__(self, function_space, gls_model,
                 bathymetry=None, v_elem_size=None, h_elem_size=None):
        """
        :arg function_space: :class:`FunctionSpace` where the solution belongs
        :arg gls_model: :class:`.GenericLengthScaleModel` object
        :kwarg bathymetry: bathymetry of the domain
        :type bathymetry: 3D :class:`Function` or :class:`Constant`
        :kwarg v_elem_size: scalar :class:`Function` that defines the vertical
            element size
        :kwarg h_elem_size: scalar :class:`Function` that defines the horizontal
            element size
        """
        super(TKEEquation, self).__init__(function_space)

        diff = GLSVerticalDiffusionTerm(function_space,
                                        gls_model.options.schmidt_nb_tke,
                                        bathymetry, v_elem_size, h_elem_size)
        source = TKESourceTerm(function_space,
                               gls_model,
                               bathymetry, v_elem_size, h_elem_size)
        self.add_term(source, 'implicit')
        self.add_term(diff, 'implicit')


class PsiEquation(Equation):
    r"""
    Generic length scale equation :eq:`turb_psi_eq` without advection terms.

    Advection of :math:`\psi` is implemented using the standard tracer equation.
    """
    def __init__(self, function_space, gls_model,
                 bathymetry=None, v_elem_size=None, h_elem_size=None):
        """
        :arg function_space: :class:`FunctionSpace` where the solution belongs
        :arg gls_model: :class:`.GenericLengthScaleModel` object
        :kwarg bathymetry: bathymetry of the domain
        :type bathymetry: 3D :class:`Function` or :class:`Constant`
        :kwarg v_elem_size: scalar :class:`Function` that defines the vertical
            element size
        :kwarg h_elem_size: scalar :class:`Function` that defines the horizontal
            element size
        """
        super(PsiEquation, self).__init__(function_space)

        diff = GLSVerticalDiffusionTerm(function_space,
                                        gls_model.options.schmidt_nb_psi,
                                        bathymetry, v_elem_size, h_elem_size)
        source = PsiSourceTerm(function_space,
                               gls_model,
                               bathymetry, v_elem_size, h_elem_size)
        self.add_term(diff, 'implicit')
        self.add_term(source, 'implicit')


class PacanowskiPhilanderModel(TurbulenceModel):
    r"""
    Gradient Richardson number based model by Pacanowski and Philander (1981).

    Given the gradient Richardson number :math:`Ri` the eddy viscosity and
    diffusivity are computed diagnostically as

    .. math::
        \nu &= \frac{\nu_{max}}{(1 + \alpha Ri)^n} \\
        \mu &= \frac{\nu}{1 + \alpha Ri}

    where :math:`\nu_{max},\alpha,n` are constant parameters.
    In unstably stratified cases where :math:`Ri<0`, value :math:`Ri=0` is used.

    Pacanowski and Philander (1981). Parameterization of vertical mixing in
    numerical models of tropical oceans. Journal of Physical Oceanography,
    11(11):1443-1451.
    http://dx.doi.org/10.1175/1520-0485(1981)011%3C1443:POVMIN%3E2.0.CO;2
    """
    def __init__(self, solver, uv_field, rho_field,
                 eddy_diffusivity, eddy_viscosity,
                 n2, m2, options=None):
        """
        :arg solver: FlowSolver object
        :arg uv_field: horizontal velocity field
        :type uv_field: :class:`Function`
        :arg rho_field: water density field
        :type rho_field: :class:`Function`
        :arg eddy_diffusivity: eddy diffusivity field
        :type eddy_diffusivity: :class:`Function`
        :arg eddy_viscosity: eddy viscosity field
        :type eddy_viscosity: :class:`Function`
        :arg n2: field for buoyancy frequency squared
        :type n2: :class:`Function`
        :arg m2: field for vertical shear frequency squared
        :type m2: :class:`Function`
        :kwarg options: model options
        """
        self.solver = solver
        # 3d model fields
        self.uv = uv_field
        self.rho = rho_field
        # diagnostic fields
        self.viscosity = eddy_viscosity
        self.diffusivity = eddy_diffusivity
        self.n2 = n2
        self.m2 = m2
        self.mu_tmp = Function(self.m2.function_space(),
                               name='tmp Shear frequency')
        self.mu = Function(self.m2.function_space(), name='Shear frequency X')
        self.mv = Function(self.m2.function_space(), name='Shear frequency Y')
        self.n2_tmp = Function(self.n2.function_space(),
                               name='tmp buoyancy frequency')
        self.n2_pos = Function(self.n2.function_space(),
                               name='positive buoyancy frequency')

        self.options = PacanowskiPhilanderModelOptions()
        if options is not None:
            self.options.update(options)

        self.shear_frequency_solver = ShearFrequencySolver(self.uv, self.m2,
                                                           self.mu, self.mv,
                                                           self.mu_tmp)
        if self.rho is not None:
            self.buoy_frequency_solver = BuoyFrequencySolver(self.rho, self.n2,
                                                             self.n2_tmp)

        self.initialize()
        print_output(self.options)

    def initialize(self):
        """Initializes fields"""
        self.n2.assign(1e-12)
        self.n2_pos.assign(1e-12)
        self.preprocess(init_solve=True)
        self.postprocess()

    def preprocess(self, init_solve=False):
        """
        Computes all diagnostic variables that depend on the mean flow model
        variables.

        To be called before updating the turbulence PDEs.
        """

        self.shear_frequency_solver.solve(init_solve=init_solve)

        if self.rho is not None:
            self.buoy_frequency_solver.solve(init_solve=init_solve)
            # split to positive and negative parts
            self.n2_pos.assign(1e-12)
            pos_ix = self.n2.dat.data[:] >= 0.0
            self.n2_pos.dat.data[pos_ix] = self.n2.dat.data[pos_ix]

    def postprocess(self):
        """
        Updates all diagnostic variables that depend on the turbulence state
        variables.

        To be called after evaluating the equations.
        """
        ri = self.n2_pos.dat.data[:]/self.m2.dat.data[:]
        denom = 1.0 + self.options.alpha*ri
        self.viscosity.dat.data[:] = self.options.max_viscosity/denom**self.options.exponent
        self.diffusivity.dat.data[:] = self.viscosity.dat.data[:]/denom
