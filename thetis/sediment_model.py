from .utility import *
from .log import warning


class CorrectiveVelocityFactor:
    def __init__(self, depth, ksp, settling_velocity, ustar, a):
        """
        Set up advective velocity factor `self.velocity_correction_factor`
        which accounts for mismatch between depth-averaged product of
        velocity with sediment and product of depth-averaged velocity
        with depth-averaged sediment.

        :arg depth: Depth of fluid
        :type depth: :class:`Function`
        :arg ksp: Grain roughness coefficient
        :type ksp: :class:`Constant`
        :arg settling_velocity: Settling velocity of the sediment particles
        :type settling_velocity: :class:`Constant`
        :arg ustar: Shear velocity
        :type ustar: :class:`Expression of functions`
        :arg a: Factor of bottom bed reference height
        :type a: :class:`Constant`
        """
        kappa = physical_constants['von_karman']

        # correction factor to advection velocity in sediment concentration equation
        Bconv = conditional(depth > Constant(1.1)*ksp, ksp/depth, Constant(1/1.1))
        Aconv = conditional(depth > Constant(1.1)*a, a/depth, Constant(1/1.1))

        # take max of value calculated either by ksp or depth
        Amax = conditional(Aconv > Bconv, Aconv, Bconv)

        r1conv = Constant(1) - (1/kappa)*conditional(settling_velocity/ustar < Constant(1),
                                                     settling_velocity/ustar, Constant(1))

        Ione = conditional(r1conv > Constant(1e-8), (Constant(1) - Amax**r1conv)/r1conv,
                           conditional(r1conv < Constant(- 1e-8), (Constant(1) - Amax**r1conv)/r1conv, ln(Amax)))

        Itwo = conditional(r1conv > Constant(1e-8), -(Ione + (ln(Amax)*(Amax**r1conv)))/r1conv,
                           conditional(r1conv < Constant(- 1e-8), -(Ione + (ln(Amax)*(Amax**r1conv)))/r1conv,
                                       Constant(-0.5)*ln(Amax)**2))

        self.alpha = -(Itwo - (ln(Amax) - ln(30))*Ione)/(Ione * ((ln(Amax) - ln(30)) + Constant(1)))

        # final correction factor
        self.velocity_correction_factor = Function(depth.function_space(), name='velocity correction factor')
        self.velocity_correction_factor_expr = max_value(min_value(self.alpha, Constant(1)), Constant(0.))
        self.update()

    def update(self):
        """
        Update `self.velocity_correction_factor` using the updated values for velocity
        """
        # final correction factor
        self.velocity_correction_factor.interpolate(self.velocity_correction_factor_expr)


class SedimentModel(object):
    def __init__(self, options, mesh2d, uv, elev, depth):

        """
        Set up a full morphological model simulation based on provided velocity and elevation functions.

        :arg options: Model options.
        :type options: :class:`.ModelOptions2d` instance
        :arg mesh2d: :class:`Mesh` object of the 2D mesh
        :arg uv: the velocity solution during the simulation
        :type uv: :class:`Function`
        :arg elev: the elevation solution during the simulation
        :type elev: :class:`Function`
        :arg depth: a :class:`DepthExpression` instance to evaluate the current depth

        The :class:`.SedimentModel` provides various expressions to be used in a suspended sediment and/or
        the Exner equation. NOTE that the functions used in these expressions need to be updated
        with the current values of the uv and elev fields by calling :func:`.SedimentModel.update`. This is not done
        in the initialisation of the sediment model, so that the :class:`.SedimentModel` can be created before
        initial conditions have been assigned to uv and elev. After the initial conditions have been
        assigned a call to update is required to ensure that the initial values are reflected in
        the :class:`.SedimentModel` terms.
        """

        self.uv = uv
        self.elev = elev
        self.depth = depth
        self.options = options
        self.solve_suspended_sediment = options.sediment_model_options.solve_suspended_sediment
        self.use_bedload = options.sediment_model_options.use_bedload
        self.use_sediment_slide = options.sediment_model_options.use_sediment_slide
        self.use_angle_correction = options.sediment_model_options.use_angle_correction
        self.use_slope_mag_correction = options.sediment_model_options.use_slope_mag_correction
        self.use_advective_velocity_correction = options.sediment_model_options.use_advective_velocity_correction
        self.use_secondary_current = options.sediment_model_options.use_secondary_current

        self.mesh2d = mesh2d

        if not self.use_bedload:
            if self.use_angle_correction:
                warning('Slope effect angle correction only applies to bedload transport which is not used in this simulation')
            if self.use_slope_mag_correction:
                warning('Slope effect magnitude correction only applies to bedload transport which is not used in this simulation')
            if self.use_secondary_current:
                warning('Secondary current only applies to bedload transport which is not used in this simulation')

        self.average_size = options.sediment_model_options.average_sediment_size
        self.bed_reference_height = options.sediment_model_options.bed_reference_height
        self.rhos = options.sediment_model_options.sediment_density

        # define function spaces
        self.P1DG_2d = get_functionspace(mesh2d, "DG", 1)
        self.P1_2d = get_functionspace(mesh2d, "CG", 1)
        self.R_1d = get_functionspace(mesh2d, "R", 0)
        self.P1v_2d = VectorFunctionSpace(mesh2d, "CG", 1)

        self.n = FacetNormal(mesh2d)

        # define parameters
        self.g = physical_constants['g_grav']
        self.rhow = physical_constants['rho0']
        kappa = physical_constants['von_karman']

        ksp = Function(self.P1_2d).interpolate(3*self.average_size)
        self.a = Function(self.P1_2d).interpolate(self.bed_reference_height/2)

        if self.options.sediment_model_options.morphological_viscosity is None:
            self.viscosity = self.options.horizontal_viscosity
        else:
            self.viscosity = self.options.sediment_model_options.morphological_viscosity

        # magnitude slope effect parameter
        self.beta = self.options.sediment_model_options.slope_effect_parameter
        # angle correction slope effect parameters
        self.surbeta2 = self.options.sediment_model_options.slope_effect_angle_parameter
        # secondary current parameter
        self.alpha_secc = self.options.sediment_model_options.secondary_current_parameter

        # calculate critical shields parameter thetacr
        self.R = self.rhos/self.rhow - Constant(1)

        self.dstar = Function(self.P1_2d).interpolate(self.average_size*((self.g*self.R)/(self.viscosity**2))**(1/3))
        if float(max(self.dstar.dat.data[:])) < 1:
            raise ValueError('dstar value less than 1')
        self.thetacr = Function(self.P1_2d).assign(conditional(self.dstar < 4, 0.24*(self.dstar**(-1)),
                                                   conditional(self.dstar < 10, 0.14*(self.dstar**(-0.64)),
                                                   conditional(self.dstar < 20, 0.04*(self.dstar**(-0.1)),
                                                               conditional(self.dstar < 150, 0.013*(self.dstar**(0.29)), 0.055)))))

        # critical bed shear stress
        self.taucr = Function(self.P1_2d).interpolate((self.rhos-self.rhow)*self.g*self.average_size*self.thetacr)

        # calculate settling velocity
        self.settling_velocity = Function(self.P1_2d).interpolate(conditional(self.average_size <= 1e-04,
                                                                              self.g*(self.average_size**2)*self.R/(18*self.viscosity),
                                                                              conditional(self.average_size <= 1e-03, (10*self.viscosity/self.average_size)
                                                                                          * (sqrt(1 + 0.01*((self.R*self.g*(self.average_size**3))
                                                                                             / (self.viscosity**2)))-1), 1.1*sqrt(self.g*self.average_size*self.R))))

        # first step: project velocity to CG
        self.uv_cg = Function(self.P1v_2d).project(self.uv)
        self.old_bathymetry_2d = Function(self.P1_2d).interpolate(self.depth.bathymetry_2d)
        self.depth_tot = Function(self.P1_2d).project(self.depth.get_total_depth(self.elev))

        self.u = self.uv_cg[0]
        self.v = self.uv_cg[1]

        # define bed friction
        hc = conditional(self.depth_tot > Constant(0.001), self.depth_tot, Constant(0.001))
        aux = conditional(11.036*hc/self.bed_reference_height > Constant(1.001),
                          11.036*hc/self.bed_reference_height, Constant(1.001))
        self.qfc = Constant(2)/(ln(aux)/kappa)**2
        # skin friction coefficient
        cfactor = conditional(self.depth_tot > ksp, Constant(2)
                              * (((1/kappa)*ln(11.036*self.depth_tot/ksp))**(-2)), Constant(0.0))
        # mu - ratio between skin friction and normal friction
        self.mu = conditional(self.qfc > Constant(0), cfactor/self.qfc, Constant(0))

        # calculate bed shear stress
        self.unorm = (self.u**2) + (self.v**2)

        self.bed_stress = Function(self.P1_2d).interpolate(self.rhow*Constant(0.5)*self.qfc*self.unorm)

        if self.solve_suspended_sediment:
            # deposition flux - calculating coefficient to account for stronger conc at bed
            self.B = conditional(self.a > self.depth_tot, Constant(1.0), self.a/self.depth_tot)
            ustar = sqrt(Constant(0.5)*self.qfc*self.unorm)
            self.rouse_number = (self.settling_velocity/(kappa*ustar)) - Constant(1)

            self.intermediate_step = conditional(abs(self.rouse_number) > Constant(1e-04),
                                                 self.B*(Constant(1)-self.B**min_value(self.rouse_number, Constant(3)))/min_value(self.rouse_number,
                                                 Constant(3)), -self.B*ln(self.B))

            self.integrated_rouse = max_value(conditional(self.intermediate_step > Constant(1e-12), Constant(1)/self.intermediate_step,
                                                          Constant(1e12)), Constant(1))

            # erosion flux - above critical velocity bed is eroded
            self.transport_stage_param = conditional(self.rhow*Constant(0.5)*self.qfc*self.unorm*self.mu > Constant(0),
                                                     (self.rhow*Constant(0.5)*self.qfc*self.unorm*self.mu - self.taucr)/self.taucr,
                                                     Constant(-1))

            self.erosion_concentration = Function(self.P1DG_2d).project(Constant(0.015)*(self.average_size/self.a)
                                                                        * ((max_value(self.transport_stage_param, Constant(0)))**1.5)
                                                                        / (self.dstar**0.3))

            if self.use_advective_velocity_correction:
                self.correction_factor_model = CorrectiveVelocityFactor(self.depth_tot, ksp,
                                                                        self.settling_velocity, ustar, self.a)
                self.velocity_correction_factor = self.correction_factor_model.velocity_correction_factor
            self.equilibrium_tracer = Function(self.P1DG_2d).interpolate(self.erosion_concentration/self.integrated_rouse)

            # get individual terms
            self._deposition = self.settling_velocity*self.integrated_rouse
            self._erosion = self.settling_velocity*self.erosion_concentration

        if self.use_bedload:
            # calculate angle of flow
            self.calfa = Function(self.P1_2d).interpolate(self.uv_cg[0]/sqrt(self.unorm))
            self.salfa = Function(self.P1_2d).interpolate(self.uv_cg[1]/sqrt(self.unorm))

            if self.use_angle_correction:
                # slope effect angle correction due to gravity
                self.stress = Function(self.P1DG_2d).interpolate(self.rhow*Constant(0.5)*self.qfc*self.unorm)

    def get_bedload_term(self, bathymetry):
        """
        Returns expression for bedload transport :math:`(qbx, qby)` to be used in the Exner equation.
        Note bathymetry is the function which is solved for in the exner equation.

        :arg bathymetry: Bathymetry of the domain. Bathymetry stands for
            the bedlevel (positive downwards).
        """

        # define bed gradient
        dzdx = self.old_bathymetry_2d.dx(0)
        dzdy = self.old_bathymetry_2d.dx(1)

        if self.use_slope_mag_correction:
            # slope effect magnitude correction due to gravity where beta is a parameter normally set to 1.3
            # we use z_n1 and equals so that we can use an implicit method in Exner
            slopecoef = Constant(1) + self.beta*(bathymetry.dx(0)*self.calfa + bathymetry.dx(1)*self.salfa)
        else:
            slopecoef = Constant(1.0)

        if self.use_angle_correction:
            # slope effect angle correction due to gravity
            cparam = Function(self.P1_2d).interpolate((self.rhos-self.rhow)*self.g*self.average_size*(self.surbeta2**2))
            tt1 = conditional(self.stress > Constant(1e-10), sqrt(cparam/self.stress), sqrt(cparam/Constant(1e-10)))

            # add on a factor of the bed gradient to the normal
            aa = self.salfa + tt1*dzdy
            bb = self.calfa + tt1*dzdx

            comb = sqrt(aa**2 + bb**2)
            angle_norm = conditional(comb > Constant(1e-10), comb, Constant(1e-10))

            # we use z_n1 and equals so that we can use an implicit method in Exner
            calfamod = (self.calfa + (tt1*bathymetry.dx(0)))/angle_norm
            salfamod = (self.salfa + (tt1*bathymetry.dx(1)))/angle_norm

        if self.use_secondary_current:
            # accounts for helical flow effect in a curver channel
            # use z_n1 and equals so can use an implicit method in Exner
            free_surface_dx = self.depth_tot.dx(0) - bathymetry.dx(0)
            free_surface_dy = self.depth_tot.dx(1) - bathymetry.dx(1)

            velocity_slide = (self.u*free_surface_dy)-(self.v*free_surface_dx)

            tandelta_factor = Constant(7)*self.g*self.rhow*self.depth_tot*self.qfc\
                / (Constant(2)*self.alpha_secc*((self.u**2) + (self.v**2)))

            # accounts for helical flow effect in a curver channel
            if self.use_angle_correction:
                # if angle has already been corrected we must alter the corrected angle to obtain the corrected secondary current angle
                t_1 = (self.bed_stress*slopecoef*calfamod) + (self.v*tandelta_factor*velocity_slide)
                t_2 = (self.bed_stress*slopecoef*salfamod) - (self.u*tandelta_factor*velocity_slide)
            else:
                t_1 = (self.bed_stress*slopecoef*self.calfa) + (self.v*tandelta_factor*velocity_slide)
                t_2 = ((self.bed_stress*slopecoef*self.salfa) - (self.u*tandelta_factor*velocity_slide))

            # calculated to normalise the new angles
            t4 = sqrt((t_1**2) + (t_2**2))

            # updated magnitude correction and angle corrections
            slopecoef_secc = t4/self.bed_stress

            calfanew = t_1/t4
            salfanew = t_2/t4

        # implement meyer-peter-muller bedload transport formula
        thetaprime = self.mu*(self.rhow*Constant(0.5)*self.qfc*self.unorm)/((self.rhos-self.rhow)*self.g*self.average_size)

        # if velocity above a certain critical value then transport occurs
        phi = conditional(thetaprime < self.thetacr, Constant(0), Constant(8)*(thetaprime-self.thetacr)**1.5)

        # bedload transport flux with magnitude correction
        if self.use_secondary_current:
            qb_total = slopecoef_secc*phi*sqrt(self.g*self.R*self.average_size**3)
        else:
            qb_total = slopecoef*phi*sqrt(self.g*self.R*self.average_size**3)

        # formulate bedload transport flux with correct angle depending on corrections implemented
        if self.use_angle_correction and self.use_secondary_current is False:
            qbx = qb_total*calfamod
            qby = qb_total*salfamod
        elif self.use_secondary_current:
            qbx = qb_total*calfanew
            qby = qb_total*salfanew
        else:
            qbx = qb_total*self.calfa
            qby = qb_total*self.salfa

        return qbx, qby

    def get_sediment_slide_term(self, bathymetry):
        # add component to bedload transport to ensure the slope angle does not exceed a certain value

        # maximum gradient allowed by sediment slide mechanism
        self.tanphi = tan(self.options.sediment_model_options.max_angle*pi/180)
        # approximate mesh step size for sediment slide mechanism
        L = self.options.sediment_model_options.meshgrid_size

        degree_h = self.P1_2d.ufl_element().degree()

        if degree_h == 0:
            self.sigma = 1.5 / CellSize(self.mesh2d)
        else:
            self.sigma = 5.0*degree_h*(degree_h + 1)/CellSize(self.mesh2d)

        # define bed gradient
        x, y = SpatialCoordinate(self.mesh2d)

        if self.options.sediment_model_options.slide_region is not None:
            dzdx = conditional(x < self.options.sediment_model_options.slide_region, bathymetry.dx(0), Constant(0.0))
            dzdy = conditional(x < self.options.sediment_model_options.slide_region, bathymetry.dx(1), Constant(0.0))
        else:
            dzdx = bathymetry.dx(0)
            dzdy = bathymetry.dx(1)

        # calculate normal to the bed
        nz = 1/sqrt(1 + (dzdx**2 + dzdy**2))

        self.betaangle = asin(sqrt(1 - (nz**2)))
        self.tanbeta = sqrt(1 - (nz**2))/nz

        # calculating magnitude of added component
        qaval = conditional(self.tanbeta - self.tanphi > 0, (1-self.options.sediment_model_options.porosity)
                            * 0.5*(L**2)*(self.tanbeta - self.tanphi)/(cos(self.betaangle*self.options.timestep
                                                                           * self.options.sediment_model_options.morphological_acceleration_factor)), 0)
        # multiplying by direction
        alphaconst = conditional(sqrt(1 - (nz**2)) > 0, - qaval*(nz**2)/sqrt(1 - (nz**2)), 0)

        diff_tensor = as_matrix([[alphaconst, 0, ], [0, alphaconst, ]])

        return diff_tensor

    def get_deposition_coefficient(self):
        """Returns coefficient :math:`C` such that :math:`C/H*sediment` is deposition term in sediment equation

        If sediment field is depth-averaged, :math:`C*sediment` is (total) deposition (over the column)
        as it appears in the Exner equation, but deposition term in sediment equation needs
        averaging: :math:`C*sediment/H`
        If sediment field is depth-integrated, :math:`C*sediment/H` is (total) deposition (over the column)
        as it appears in the Exner equation, and is the same in the sediment equation."""
        return self._deposition

    def get_erosion_term(self):
        """Returns expression for (depth-integrated) erosion."""
        return self._erosion

    def get_equilibrium_tracer(self):
        """Returns expression for (depth-averaged) equilibrium tracer."""
        return self.equilibrium_tracer

    def get_advective_velocity_correction_factor(self):
        """Returns correction factor for the advective velocity in the sediment equations

        With :attr:`.SedimentModelOptions.use_advective_velocity_correction`, this applies a correction to
        the supplied velocity solution `uv` to take into account the mismatch between
        depth-averaged product of velocity with sediment and product of depth-averaged
        velocity with depth-averaged sediment.
        """
        if self.use_advective_velocity_correction:
            return self.velocity_correction_factor
        else:
            return 1

    def update(self):
        """Update all functions used by :class:`.SedimentModel`

        This repeats all projection and interpolations steps based on the current values
        of the `uv` and `elev` functions, provided in __init__."""

        self.uv_cg.project(self.uv)

        self.old_bathymetry_2d.interpolate(self.depth.bathymetry_2d)
        self.depth_tot.project(self.depth.get_total_depth(self.elev))

        self.bed_stress.interpolate(self.rhow*Constant(0.5)*self.qfc*self.unorm)

        if self.solve_suspended_sediment:
            self.erosion_concentration.project(Constant(0.015)*(self.average_size/self.a)
                                               * ((max_value(self.transport_stage_param, Constant(0)))**1.5)
                                               / (self.dstar**0.3))

            self.equilibrium_tracer.interpolate(self.erosion_concentration/self.integrated_rouse)

        if self.use_bedload:
            # calculate angle of flow
            self.calfa.interpolate(self.uv_cg[0]/sqrt(self.unorm))
            self.salfa.interpolate(self.uv_cg[1]/sqrt(self.unorm))

            if self.use_angle_correction:
                # slope effect angle correction due to gravity
                self.stress.interpolate(self.rhow*Constant(0.5)*self.qfc*self.unorm)

        if self.use_advective_velocity_correction:
            self.correction_factor_model.update()
