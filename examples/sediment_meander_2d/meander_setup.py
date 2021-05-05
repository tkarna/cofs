"""
Meander Test case
=======================
Solves the initial hydrodynamics simulation of flow around a 180 degree bend replicating
lab experiment 4 in Yen & Lee (1995).
Note this is not the main run-file and is just used to create an initial checkpoint for
the morphodynamic simulation.

For more details of the test case set-up see
[1] Clare et al. (2020). Hydro-morphodynamics 2D modelling using a discontinuous Galerkin discretisation.
    Computers & Geosciences, 104658. https://doi.org/10.1016/j.cageo.2020.104658
"""

from thetis import *

# define mesh
mesh2d = Mesh("meander.msh")
x, y = SpatialCoordinate(mesh2d)
# define function spaces
V = FunctionSpace(mesh2d, 'CG', 1)

# define underlying bathymetry
bathymetry_2d = Function(V, name='Bathymetry')
gradient = Constant(0.0035)
L_function = Function(V).interpolate(conditional(x > 5, pi*4*((pi/2)-acos((x-5)/(sqrt((x-5)**2+(y-2.5)**2))))/pi,
                                                 pi*4*((pi/2)-acos((-x+5)/(sqrt((x-5)**2+(y-2.5)**2))))/pi))
bathymetry_curve = Function(V).interpolate(conditional(y > 2.5,
                                           conditional(x < 5, (L_function*gradient) + 9.97072,
                                                       -(L_function*gradient) + 9.97072), 9.97072))
init = max(bathymetry_curve.dat.data[:])
final = min(bathymetry_curve.dat.data[:])
bathymetry_straight = Function(V).interpolate(conditional(x <= 5,
                                              conditional(y <= 2.5, -9.97072 + gradient*abs(y - 2.5) + init, 0),
                                              conditional(y <= 2.5, -9.97072 - gradient*abs(y - 2.5) + final, 0)))
bathymetry_2d = Function(V).interpolate(-bathymetry_curve - bathymetry_straight)
