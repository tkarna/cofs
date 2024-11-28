from thetis import *
from firedrake import VTKFile
from mpi4py import MPI

# Initialize MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


def generate_bathymetry(mesh, H, output_directory, exporting=True):
    """
        Generates a bathymetry function for a given mesh file by solving the Eikonal
        equation and projects the result onto a bathymetry function.

        Parameters:
        -----------
        mesh_file : str
            Path to the mesh file used to create the computational mesh.
        H : float
            The depth value used for the bathymetry, reduces to 5 at the shoreline (ID = 3).
        output_directory : str
            Directory where the output files (distance and bathymetry) will be saved.

        Returns:
        --------
        bathymetry : firedrake.Function
            The computed bathymetry function.
    """

    # create function space
    V = FunctionSpace(mesh, 'CG', 1)

    print_output("Calculating distance for bathymetry gradient")

    # Boundary
    bcs = [DirichletBC(V, 0.0, 3)]  # 3 = coasts PhysID

    L = 500  # distance to shore
    v = TestFunction(V)
    u = Function(V)

    solver_parameters = {
        'snes_type': 'ksponly',
        'ksp_rtol': 1e-4,
        'ksp_type': 'preonly',
        'pc_type': 'lu',
        'pc_factor_mat_solver_packages': 'mumps'
    }

    # Before we solve the Eikonal equation, let's solve a Laplace equation to
    # generate an initial guess
    F = L ** 2 * (inner(grad(u), grad(v))) * dx - v * dx
    solve(F == 0, u, bcs, solver_parameters=solver_parameters)
    # Relax the solver parameters for the Eikonal equation
    solver_parameters.update({
        'snes_type': 'newtonls',
        'ksp_rtol': 1e-4,
        'ksp_type': 'preonly',
        'pc_type': 'lu',
        'pc_factor_mat_solver_packages': 'mumps'
    })

    epss = [500., 200., 100., 50., 20.]

    for i, eps in enumerate(epss):
        print_output("Solving Eikonal with eps == " + str(float(eps)))
        F = inner(sqrt(inner(grad(u), grad(u))), v) * dx - v * dx + eps * inner(grad(u), grad(v)) * dx
        solve(F == 0, u, bcs, solver_parameters=solver_parameters)

    if exporting:
        VTKFile(output_directory + "/dist.pvd").write(u)

    """
    Adding viscocity sponge
    """

    bathymetry = Function(V, name="bathymetry")
    bathymetry.project(conditional(ge(u, L), H, (H - 5) * (u / L) + 5.))
    if exporting:
        VTKFile(output_directory + '/bathymetry.pvd').write(bathymetry)

    return bathymetry


def construct_solver(store_station_time_series=True, **model_options):
    mesh2d = Mesh('headland.msh')
    pwd = os.path.abspath(os.path.dirname(__file__))
    output_directory = model_options.get('output_directory', f'{pwd}/outputs_forward')
    exporting = not model_options.get('no_exports', False)

    tidal_amplitude = 1.
    tidal_period = 12.42 * 60 * 60
    t_end = tidal_period
    H = 40
    dt = 800.
    t_export = dt

    if os.getenv('THETIS_REGRESSION_TEST') is not None:
        t_end = 5*dt

    # bathymetry
    P1_2d = get_functionspace(mesh2d, 'CG', 1)
    bathymetry_2d = Function(P1_2d, name="bathymetry")
    bathy = generate_bathymetry(mesh2d, H, output_directory, exporting)
    bathymetry_2d.project(bathy)

    x, y = SpatialCoordinate(mesh2d)
    local_lx = np.max(mesh2d.coordinates.dat.data[:,
                      0])  # for parallel runs, the mesh is partioned so we need to get the maximum from each processor!
    local_ly = np.max(mesh2d.coordinates.dat.data[:, 1])

    all_lx = comm.gather(local_lx, root=0)
    all_ly = comm.gather(local_ly, root=0)
    if rank == 0:
        lx_ = np.max(all_lx)
        ly_ = np.max(all_ly)
    else:
        lx_ = None
        ly_ = None
    lx = comm.bcast(lx_, root=0)
    ly = comm.bcast(ly_, root=0)

    # friction Manning coefficient
    manning_2d = Function(P1_2d, name='Manning coefficient')
    manning_low = 0.02
    manning_high = 0.05
    manning_2d.interpolate(
        conditional(x < 11e3, manning_high, manning_low))
    if exporting:
        VTKFile(output_directory + '/manning_init.pvd').write(manning_2d)

    # viscosity
    h_viscosity = Function(P1_2d).interpolate(conditional(le(x, 1e3), 0.1*(1e3 - x), 1.0))
    if exporting:
        VTKFile(output_directory + '/h_viscosity.pvd').write(h_viscosity)

    # create solver
    solver_obj = solver2d.FlowSolver2d(mesh2d, bathymetry_2d)
    options = solver_obj.options

    options.simulation_export_time = t_export
    options.simulation_end_time = t_end
    options.output_directory = output_directory
    options.check_volume_conservation_2d = True
    options.fields_to_export = ['uv_2d', 'elev_2d']
    options.element_family = 'dg-cg'
    options.swe_timestepper_type = 'CrankNicolson'
    options.swe_timestepper_options.implicitness_theta = 0.6
    options.swe_timestepper_options.solver_parameters = {'snes_rtol': 1e-9,
                                                         'ksp_type': 'preonly',
                                                         'pc_type': 'lu',
                                                         'pc_factor_mat_solver_type': 'mumps',
                                                         'mat_type': 'aij'
                                                         }
    options.horizontal_viscosity = h_viscosity
    options.timestep = dt
    options.update(model_options)

    options.manning_drag_coefficient = manning_2d
    solver_obj.add_new_field(manning_2d, 'manning_2d',
                             manning_2d.name(),
                             'Manning2d', unit='s m-1/3')

    solver_obj.create_equations()

    if store_station_time_series:
        # store elevation time series at stations
        stations = [
            ('stationA', (lx/10, ly/2)),
            ('stationB', (lx/2, ly/2)),
            ('stationC', (3*lx/4, ly/4)),
            ('stationD', (3*lx/4, 3*ly/4)),
            ('stationE', (9*lx/10, ly/2)),
        ]
        for name, (sta_x, sta_y) in stations:
            cb = DetectorsCallback(solver_obj, [(sta_x, sta_y)], ['uv_2d'], name='timeseries_'+name+'_uv',
                                   detector_names=[name], append_to_log=False)
            solver_obj.add_callback(cb)

    left_tag = 1
    right_tag = 2
    coasts_tag = 3
    tidal_elev = Function(get_functionspace(mesh2d, "CG", 1), name='tidal_elev')
    tidal_elev_bc = {'elev': tidal_elev}
    # noslip currently doesn't work (vector Constants are broken in firedrake adjoint)
    freeslip_bc = {'un': Constant(0.0)}
    solver_obj.bnd_functions['shallow_water'] = {
        left_tag: tidal_elev_bc,
        right_tag: tidal_elev_bc,
        coasts_tag: freeslip_bc
    }

    # a function to update the tidal_elev bc value every timestep
    g = 9.81
    omega = 2 * pi / tidal_period

    def update_forcings(t):
        print_output("Updating tidal elevation at t = {}".format(t))
        tidal_elev.project(tidal_amplitude * sin(omega * t + omega / pow(g * H, 0.5) * x))

    # set initial condition for elevation, piecewise linear function
    solver_obj.assign_initial_conditions(uv=as_vector((1e-7, 0.0)), elev=tidal_elev)

    return solver_obj, update_forcings
