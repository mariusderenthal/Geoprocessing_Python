# ############################################################################################################# #
# Raw-example for a basic script to be executed in python. The format is only a recommendation and everyone is  #
# encouraged to modify the scripts to her/his own needs.                                                        #
# (c) Matthias Baumann, Humboldt-Universität zu Berlin, 4/15/2019
#
# Assignment 07: Marius Derenthal                                                                                   #
# ####################################### LOAD REQUIRED LIBRARIES ############################################# #
import time
from osgeo import ogr
from osgeo import osr
import math
import sys
import numpy as np
import random
# ####################################### SET TIME-COUNT ###################################################### #
starttime = time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime())
print("--------------------------------------------------------")
print("Starting process, time: " + starttime)
print("")
# ####################################### FUNCTIONS ########################################################### #
def CS_Transform(LYRfrom, LYRto):
    outPR = LYRto.GetSpatialRef()
    inPR = LYRfrom.GetSpatialRef()
    transform = osr.CoordinateTransformation(inPR, outPR)
    return transform
def create_block(x, y):
    x_tl = x - 45
    y_tl = y + 45
    x_tr = x + 45
    y_tr = y + 45
    x_bl = x - 45
    y_bl = y - 45
    x_br = x + 45
    y_br = y - 45

    ring = ogr.Geometry(ogr.wkbLinearRing)

    ring.AddPoint(x_bl, y_bl)
    ring.AddPoint(x_br, y_br)
    ring.AddPoint(x_tl, y_tl)
    ring.AddPoint(x_tr, y_tr)
    ring.AddPoint(x_bl, y_bl)

    poly = ogr.Geometry(ogr.wkbPolygon)

    poly.AddGeometry(ring)

    return poly
def check_intersection(block, PA):
    interesection = PA.Intersection(block)
    return interesection.area == block.area
''' Potentially usefull functions I found on the internet
def create_grid(outputGridfn,xmin,xmax,ymin,ymax,gridHeight,gridWidth):
    #https://gis.stackexchange.com/questions/54119/creating-square-grid-polygon-shapefile-with-python/78030
    # convert sys.argv to float
    xmin = float(xmin)
    xmax = float(xmax)
    ymin = float(ymin)
    ymax = float(ymax)
    gridWidth = float(gridWidth)
    gridHeight = float(gridHeight)

    # get rows
    rows = ceil((ymax-ymin)/gridHeight)
    # get columns
    cols = ceil((xmax-xmin)/gridWidth)

    # start grid cell envelope
    ringXleftOrigin = xmin
    ringXrightOrigin = xmin + gridWidth
    ringYtopOrigin = ymax
    ringYbottomOrigin = ymax-gridHeight

    # create output file
    outDriver = ogr.GetDriverByName('ESRI Shapefile')
    if os.path.exists(outputGridfn):
        os.remove(outputGridfn)
    outDataSource = outDriver.CreateDataSource(outputGridfn)
    outLayer = outDataSource.CreateLayer(outputGridfn,geom_type=ogr.wkbPolygon )
    featureDefn = outLayer.GetLayerDefn()

    # create grid cells
    countcols = 0
    while countcols < cols:
        countcols += 1

        # reset envelope for rows
        ringYtop = ringYtopOrigin
        ringYbottom =ringYbottomOrigin
        countrows = 0

        while countrows < rows:
            countrows += 1
            ring = ogr.Geometry(ogr.wkbLinearRing)
            ring.AddPoint(ringXleftOrigin, ringYtop)
            ring.AddPoint(ringXrightOrigin, ringYtop)
            ring.AddPoint(ringXrightOrigin, ringYbottom)
            ring.AddPoint(ringXleftOrigin, ringYbottom)
            ring.AddPoint(ringXleftOrigin, ringYtop)
            poly = ogr.Geometry(ogr.wkbPolygon)
            poly.AddGeometry(ring)

            # add new geom to layer
            outFeature = ogr.Feature(featureDefn)
            outFeature.SetGeometry(poly)
            outLayer.CreateFeature(outFeature)
            outFeature.Destroy

            # new envelope for next poly
            ringYtop = ringYtop - gridHeight
            ringYbottom = ringYbottom - gridHeight

        # new envelope for next poly
        ringXleftOrigin = ringXleftOrigin + gridWidth
        ringXrightOrigin = ringXrightOrigin + gridWidth

    # Close DataSources
    outDataSource.Destroy()

def random_points_within(poly, num_points):
    #https://codereview.stackexchange.com/questions/69833/generate-sample-coordinates-inside-a-polygon
    min_x, min_y, max_x, max_y = poly.bounds

    points = []

    while len(points) < num_points:
        random_point = Point([random.uniform(min_x, max_x), random.uniform(min_y, max_y)])
        if (random_point.within(poly)):
            points.append(random_point)

    return points

def random_points_within_polygon_and_10m_sbuffer(poly, num_points):
    #https://codereview.stackexchange.com/questions/69833/generate-sample-coordinates-inside-a-polygon
    min_x, min_y, max_x, max_y = poly.bounds
    min_lon = min_x
    max_lon = max_x
    min_lat = min_y
    max_lat = max_y
    # obtain utm bounds and dimentions
    utmZone = int((np.floor((min_lon+ 180)/6)% 60) + 1)
    fieldProj = Proj("+proj=utm +zone="+str(utmZone)+", +north +ellps=WGS84 +datum=WGS84 +units=m +no_defs")
    UTMx_min, UTMy_max = fieldProj(min_lon, max_lat)
    UTMx_max, UTMy_max = fieldProj(max_lon, max_lat)
    UTMx_min, UTMy_min = fieldProj(min_lon, min_lat)
    UTMx_min = math.floor(UTMx_min*0.2)*5.0 + 2.5 # make it standard grid to match img
    UTMy_max = math.ceil(UTMy_max*0.2)*5.0 - 2.5  # make it standard grid to match img

    utm_npix_x = int(0.1*(math.ceil((UTMx_max - UTMx_min)*0.1)*10))
    utm_npix_y = int(0.1*(math.ceil((UTMy_max - UTMy_min)*0.1)*10))
    #set spacing raster
    spacing_utm_grid = np.arange(utm_npix_y*utm_npix_x).reshape((utm_npix_y, utm_npix_x))
    spacing_utm_grid[:,:] = 0

    points = []
    while len(points) < num_points:
        pair_coordinates = [random.uniform(min_x, max_x), random.uniform(min_y, max_y)]
        random_point = Point(pair_coordinates)
        if (random_point.within(poly)):
            utm_x, utm_y = fieldProj(pair_coordinates[0], pair_coordinates[1])
            utm_x_pix = math.floor((utm_x - UTMx_min)/10)
            utm_y_pix = math.floor((UTMy_max -utm_y)/10)
            if(spacing_utm_grid[utm_y_pix, utm_x_pix]==0):
                points.append(random_point)
                spacing_utm_grid[utm_y_pix,utm_x_pix]=1

    return points

def buffer(infile, outfile, buffdist):
    try:
        ds_in = ogr.Open(infile)
        lyr_in = ds_in.GetLayer(0)
        drv = ds_in.GetDriver()
        if os.path.exists(outfile):
            drv.DeleteDataSource(outfile)
        ds_out = drv.CreateDataSource(outfile)

        layer = ds_out.CreateLayer(lyr_in.GetLayerDefn().GetName(), lyr_in.GetSpatialRef(), ogr.wkbPolygon)
        n_fields = lyr_in.GetLayerDefn().GetFieldCount()

        for i in range(lyr_in.GetLayerDefn().GetFieldCount()):
            field_in = lyr_in.GetLayerDefn().GetFieldDefn(i)
            fielddef = ogr.FieldDefn(field_in.GetName(), field_in.GetType())
            layer.CreateField(fielddef)

        featuredefn = layer.GetLayerDefn()

        for feat in lyr_in:
            geom = feat.GetGeometryRef()
            # feature = ogr.Feature(featuredefn)
            # pdb.set_trace()
            feature = feat.Clone()
            feature.SetGeometry(geom.Buffer(float(buffdist)))
            # for in in xrange ( n_fields ):
            # feature.SetField (
            layer.CreateFeature(feature)
            del geom
        ds_out.Destroy()
    except:
        return False
    return True
'''
# ####################################### FOLDER PATHS & global variables ##################################### #
# Folder containing the working data
path_data_folder = '/Users/mariusderenthal/Google Drive/Global Change Geography/4.Semester/Geoprocessing_Python/Data/Assignment07 - data/'
prot_areas = "WDPA_May2018_polygons_GER_select10large.shp"
#country_shp = "gadm36_GERonly.shp"
center_point = "OnePoint.shp"
# ####################################### PROCESSING ########################################################## #
# Load data
prot = ogr.Open(path_data_folder + prot_areas, 0)
if prot is None:
    sys.exit('Could not open {0}.'.format(path_data_folder + prot_areas))
lyr_prot = prot.GetLayer()


cp = ogr.Open(path_data_folder + center_point, 0)
if cp is None:
    sys.exit('Could not open {0}.'.format(path_data_folder + center_point))
lyr_cp = cp.GetLayer()


cp_point = lyr_cp.GetNextFeature()
cp_geom = cp_point.GetGeometryRef()
cp_x = cp_geom.GetX()
cp_y = cp_geom.GetY()


''' Print out some useful information about datasource
# information protected areas
lyr_prot = prot.GetLayer()
num_prot = lyr_prot.GetFeatureCount()
spatialRef_prot = lyr_prot.GetSpatialRef()

# information center point
lyr_cp = cp.GetLayer()
num_cp = lyr_cp.GetFeatureCount()
spatialRef_cp = lyr_cp.GetSpatialRef()

#Layer information
print("Number of Features in Protected areas:", lyr_prot.GetFeatureCount())
print("Number of Features in OnePoint:", lyr_cp.GetFeatureCount())
print()

print("This is the extent of all my protected areas: ", lyr.GetExtent())
print()

print("This is the Spatial reference system of all my protected areas: ", lyr_prot.GetSpatialRef())
print()

print("This is the geometrical type of all my protected areas: ", lyr_prot.GetGeomType())
print(lyr_prot.GetGeomType() == ogr.wkbPoint)
print(lyr_prot.GetGeomType() == ogr.wkbPolygon)
print(lyr_prot.GetGeomType() == ogr.wkbMultiPolygon)
print()


for field in lyr.schema:
    print(field.name, field.GetTypeName())
'''
#create empty shape file to store stuff in
# set up the shapefile driver
driver = ogr.GetDriverByName("ESRI Shapefile")
driver_kml = ogr.GetDriverByName("KML")

# create the data source
data_source = driver.CreateDataSource("PA_Poly.shp")
data_source_kml = driver_kml.CreateDataSource("PA_Poly.kml")

# create the spatial reference, WGS84
srs = osr.SpatialReference()
srs.ImportFromEPSG(32633)

# create the layer
layer = data_source.CreateLayer("", srs, ogr.wkbMultiPolygon)
layer_kml = data_source_kml.CreateLayer("", srs, ogr.wkbMultiPolygon)

# Add the fields we're interested in
PA_Name = ogr.FieldDefn("PA_Name", ogr.OFTString)
layer.CreateField(PA_Name)

Plot_ID = ogr.FieldDefn("Plot_ID", ogr.OFTInteger)
layer.CreateField(Plot_ID)

Pixel_ID = ogr.FieldDefn("Pixel_ID", ogr.OFTInteger)
layer.CreateField(Pixel_ID)

PA_Name = ogr.FieldDefn("PA_Name", ogr.OFTString)
layer_kml.CreateField(PA_Name)

Plot_ID = ogr.FieldDefn("Plot_ID", ogr.OFTInteger)
layer_kml.CreateField(Plot_ID)

Pixel_ID = ogr.FieldDefn("Pixel_ID", ogr.OFTInteger)
layer_kml.CreateField(Pixel_ID)

# set up further id arrays
plot_id = np.arange(1,51)
pixel_id = np.arange(1,10)

# transform
transform = CS_Transform(lyr_prot, lyr_cp)


# perform the actual task
for feat in lyr_prot:
    # basic stuff
    #print(feat.GetField('NAME'))
    #print("This feature has the geometrical type of: ", lyr_prot.GetGeomType())
    #print("This feature has the spatial reference: ", lyr.GetSpatialRef())

    # get the envelope
    geom = feat.GetGeometryRef()
    geom.Transform(transform)
    env = geom.GetEnvelope()
    xmin, ymin, xmax, ymax = env[0], env[2], env[1], env[3]

    # generate random points inside the envelope
    num_points = 50
    counter = 0
    multipoint = ogr.Geometry(ogr.wkbMultiPoint)

    while counter < num_points:
        #generate random point inside the envelope
        point = ogr.Geometry(ogr.wkbPoint)
        point.AddPoint(random.uniform(xmin, xmax),
                       random.uniform(ymin, ymax))

        # create new point based on new x and y values
        # calculate the difference between both points, divide by 30, round, and multiply by 30 (idea from Max Wesemeyer)
        x_rp = point.GetX()
        y_rp = point.GetY()

        x_lan = ((math.ceil((x_rp - cp_x) / 30)) * 30) + cp_x
        y_lan = ((math.ceil((y_rp - cp_y) / 30)) * 30) + cp_y

        land_point = ogr.Geometry(ogr.wkbPoint)
        land_point.AddPoint(x_lan, y_lan)

        # check if point is within the polygon
        if land_point.Within(geom):

            # create ring around the random point
            ring = create_block(x=x_lan, y=y_lan)

            if ring.Within(geom):
                counter += 1

                # create a multipolygon
                multipolygon = ogr.Geometry(ogr.wkbMultiPolygon)

                # create the 3*3 grid
                id = 1
                for y_pos, bla in zip((30,0, -30), range(0,9)):
                    for x_pos in (-30, 0, 30):

                        # create the feature
                        feature = ogr.Feature(layer.GetLayerDefn())
                        feature_kml = ogr.Feature(layer_kml.GetLayerDefn())

                        # set the attributes using the values from the delimited text file
                        feature.SetField("PA_Name", feat.GetField('NAME'))
                        feature.SetField("Plot_ID", counter)
                        feature.SetField("Pixel_ID", id)

                        feature_kml.SetField("PA_Name", feat.GetField('NAME'))
                        feature_kml.SetField("Plot_ID", counter)
                        feature_kml.SetField("Pixel_ID", id)

                        ringy = ogr.Geometry(ogr.wkbLinearRing)

                        ringy.AddPoint(x_lan + x_pos + 15, y_lan + y_pos + 15)
                        ringy.AddPoint(x_lan + x_pos + 15, y_lan + y_pos - 15)

                        ringy.AddPoint(x_lan + x_pos - 15, y_lan + y_pos - 15)
                        ringy.AddPoint(x_lan + x_pos - 15, y_lan + y_pos + 15)
                        ringy.AddPoint(x_lan + x_pos + 15, y_lan + y_pos + 15)

                        # create polygon
                        poly = ogr.Geometry(ogr.wkbPolygon)
                        poly.AddGeometry(ringy)

                        # set the feature geometry
                        feature.SetGeometry(poly)
                        feature_kml.SetGeometry(poly)

                        # create the feature in the layer
                        layer.CreateFeature(feature)
                        layer_kml.CreateFeature(feature_kml)

                        multipolygon.AddGeometry(poly)

                        # INFO: id = 5 is the central polygon
                        id += 1

    print("Done sampling in PA: ", feat.GetField('NAME'))

# ####################################### END TIME-COUNT AND PRINT TIME STATS##################
print("")
endtime = time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime())
print("--------------------------------------------------------")
print("start: " + starttime)
print("end: " + endtime)
print("")