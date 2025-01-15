import qmesh3 as qmesh


def generate_mesh(level):
    """
    Generate a mesh of the North Sea domain using qmesh.
    """
    name = "north_sea.{:s}"

    # Setup qmesh
    qmesh.setLogOutputFile("qmesh.log")

    # Read shapefile describing domain boundaries
    boundary = qmesh.vector.Shapes()
    boundary.fromFile("mesh_data/Boundary.shp")
    loop_shapes = qmesh.vector.identifyLoops(
        boundary,
        isGlobal=False,
        defaultPhysID=1000,
        fixOpenLoops=True,
    )
    polygon_shapes = qmesh.vector.identifyPolygons(
        loop_shapes,
        smallestNotMeshedArea=5.0e06,
        smallestMeshedArea=2.0e08,
        meshedAreaPhysID=1,
    )
    polygon_shapes.writeFile("mesh_data/Polygons.shp")

    # Create raster for mesh gradation towards coast
    gshhs_coast = qmesh.vector.Shapes()
    gshhs_coast.fromFile("mesh_data/Coastline.shp")
    gradation_raster = qmesh.raster.gradationToShapes()
    gradation_raster.setShapes(gshhs_coast)
    gradation_raster.setRasterBounds(-16.0, 12.0, 48.0, 64.0)
    gradation_raster.setRasterResolution(300, 300)
    gradation_raster.setGradationParameters(
        10.0e03 * 0.5**level,  # inner gradation (m)
        30.0e03 * 0.5**level,  # outer gradation (m)
        0.5,  # gradation distance (degrees)
    )
    gradation_raster.calculateLinearGradation()
    gradation_raster.writeNetCDF("mesh_data/Gradation.nc")

    # Create domain object and write GMSH files
    domain = qmesh.mesh.Domain()
    domain.setTargetCoordRefSystem("EPSG:32630", fldFillValue=1000.0)
    domain.setGeometry(loop_shapes, polygon_shapes)
    domain.setMeshMetricField(gradation_raster)

    # Generate mesh
    domain.gmsh(
        geoFilename=name.format("geo"),
        fldFilename=name.format("fld"),
        mshFilename=name.format("msh"),
    )

    # Write mesh in shapefile format for visualisation in GIS software
    north_sea_mesh = qmesh.mesh.Mesh()
    north_sea_mesh.readGmsh(name.format("msh"), "EPSG:3857")
    north_sea_mesh.writeShapefile(name.format("shp"))


if __name__ == "__main__":
    generate_mesh(0)
