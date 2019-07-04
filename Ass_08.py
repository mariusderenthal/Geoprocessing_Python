# ############################################################################################################# #
# Raw-example for a basic script to be executed in python. The format is only a recommendation and everyone is  #
# encouraged to modify the scripts to her/his own needs.                                                        #
# (c) Matthias Baumann, Humboldt-Universität zu Berlin, 4/15/2019
#
# Assignment 08: Marius Derenthal                                                                                   #
# ####################################### LOAD REQUIRED LIBRARIES ############################################# #
import time
import gdal
from osgeo import ogr
from osgeo import osr
import sys
import pandas as pd
import numpy as np
# ####################################### SET TIME-COUNT ###################################################### #
starttime = time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime())
print("--------------------------------------------------------")
print("Starting process, time: " + starttime)
print("")
# ####################################### FUNCTIONS ########################################################### #
# ####################################### FOLDER PATHS & global variables ##################################### #
# Folder containing the working data
'''
path_data_folder = '/Users/mariusderenthal/Google Drive/Global Change Geography/4.Semester/Geoprocessing_Python/Data/Assignment08_data/'
dem_p = "DEM_Humboldt.tif"
mar_p = "Marihuana_Grows.shp"
par_p = "Parcels.shp"
land_p = "PublicLands.shp"
road_p = "Roads.shp"
timber_p = "TimberHarvestPlan.shp"
'''

bufferDistance = 10
# ####################################### PROJECTION ########################################################## #
'''
# 0. transform everything to
# EPSGS
# dem: 3310
# mar: 3310
# par: 26741
# land: 26741
# road: 26741
# timber: 26741

driver = ogr.GetDriverByName('ESRI Shapefile')

# input SpatialReference
inSpatialRef = osr.SpatialReference()
inSpatialRef.ImportFromEPSG(26741)

# output SpatialReference
outSpatialRef = osr.SpatialReference()
outSpatialRef.ImportFromEPSG(3310)

# create the CoordinateTransformation
coordTrans = osr.CoordinateTransformation(inSpatialRef, outSpatialRef)

# get the input layer
inDataSet = driver.Open(path_data_folder + timber_p)      #SET PATH
inLayer = inDataSet.GetLayer()

# create the spatial reference, 26741
srs = osr.SpatialReference()
srs.ImportFromEPSG(3310)

# create the output layer
outputShapefile = (path_data_folder + 'TimberHarvestPlan_transformed.shp') #SET NAME
if os.path.exists(outputShapefile):
    driver.DeleteDataSource(outputShapefile)
outDataSet = driver.CreateDataSource(outputShapefile)
outLayer = outDataSet.CreateLayer("TimberHarvestPlan_transformed", srs, geom_type=ogr.wkbPolygon)    #SELECT GEOMETRY

# add fields
inLayerDefn = inLayer.GetLayerDefn()
for i in range(0, inLayerDefn.GetFieldCount()):
    fieldDefn = inLayerDefn.GetFieldDefn(i)
    outLayer.CreateField(fieldDefn)

# get the output layer's feature definition
outLayerDefn = outLayer.GetLayerDefn()

# loop through the input features
inFeature = inLayer.GetNextFeature()
while inFeature:
    # get the input geometry
    geom = inFeature.GetGeometryRef()
    # reproject the geometry
    geom.Transform(coordTrans)
    # create a new feature
    outFeature = ogr.Feature(outLayerDefn)
    # set the geometry and attribute
    outFeature.SetGeometry(geom)
    for i in range(0, outLayerDefn.GetFieldCount()):
        outFeature.SetField(outLayerDefn.GetFieldDefn(i).GetNameRef(), inFeature.GetField(i))
    # add the feature to the shapefile
    outLayer.CreateFeature(outFeature)
    # dereference the features and get the next input feature
    outFeature = None
    inFeature = inLayer.GetNextFeature()

# Save and close the shapefiles
inDataSet = None
outDataSet = None
'''
# ####################################### PROCESSING ########################################################## #
# Update Path
path_data_folder = '/Users/mariusderenthal/Google Drive/Global Change Geography/4.Semester/Geoprocessing_Python/Data/Assignment08_data/'
dem_p = "DEM_Humboldt.tif"
mar_p = "Marihuana_Grows.shp"
par_p = "Parcels_transformed.shp"
land_p = "PublicLands_transformed.shp"
road_p = "Roads_transformed.shp"
timber_p = "copies/TimberHarvestPlan_transformed.shp"

# Load data
dem = gdal.Open(path_data_folder + dem_p)
dem_b = dem.GetRasterBand(1)
dem_arr = dem_b.ReadAsArray()
# [cols, rows] = dem_arr.shape
# dem_arr_mean = int(dem_arr.mean())

mar = ogr.Open(path_data_folder + mar_p, 0)
if mar is None:
    sys.exit('Could not open {0}.'.format(path_data_folder + mar_p))
lyr_mar = mar.GetLayer()

par = ogr.Open(path_data_folder + par_p, 0)
if par is None:
    sys.exit('Could not open {0}.'.format(path_data_folder + par_p))
lyr_par = par.GetLayer()

land = ogr.Open(path_data_folder + land_p, 0)
if land is None:
    sys.exit('Could not open {0}.'.format(path_data_folder + land_p))
lyr_land = land.GetLayer()

road = ogr.Open(path_data_folder + road_p, 0)
if road is None:
    sys.exit('Could not open {0}.'.format(path_data_folder + road_p))
lyr_road = road.GetLayer()

timber = ogr.Open(path_data_folder + timber_p, 1)
if timber is None:
    sys.exit('Could not open {0}.'.format(path_data_folder + timber_p))
lyr_timber = timber.GetLayer()



# 0. define final dataframe for collecting results and later export
df = pd.DataFrame(columns={'Parcel_APN': [],
                           'NR_GH_Plants': [],
                           'NR_OD_Plants': [],
                           'Dist_to_grow_m': [],
                           'Km_Priv.Road': [],
                           'Km_Loc.Road': [],
                           'Mean_Elevation': [],
                           'PublicLand_YN': [],
                           'Prop_in_THP': []
                           })



# 7. Create a union poly in order to account for overlaps of harvest area
union_poly = ogr.Geometry(ogr.wkbPolygon)
for tim in lyr_timber:
    geom = tim.GetGeometryRef()
    union_poly = union_poly.Union(geom)

lyr_timber.ResetReading()

# do everything for each parcel
for p in lyr_par:

    # 0. Some Infos on the parcel #####################################################################################
    Parcel_APN = p.GetField('APN')
    p_geom = p.geometry().Clone()
    area_parcel = p_geom.GetArea()
    print(Parcel_APN)


    # 1. Number of Cannabis Plants #####################################################################################
    lyr_mar.SetSpatialFilter(p.geometry())
    feat_count = lyr_mar.GetFeatureCount()

    g_plants_tot = 0
    o_plants_tot = 0

    for m in lyr_mar:
        g_plants = m.GetField('g_plants')
        o_plants = m.GetField('o_plants')

        g_plants_tot = g_plants_tot + g_plants
        o_plants_tot = o_plants_tot + o_plants

    lyr_mar.SetSpatialFilter(None)


    # 2. Distance of a parcel to the next grow #########################################################################
    buffer_geom = p_geom.Buffer(bufferDistance)
    lyr_mar.SetSpatialFilter(buffer_geom)
    feat_count_buff = lyr_mar.GetFeatureCount()

    while feat_count_buff == feat_count:
        buffer_geom = p_geom.Buffer(bufferDistance)
        lyr_mar.SetSpatialFilter(buffer_geom)
        feat_count_buff = lyr_mar.GetFeatureCount()

        bufferDistance += 10
        lyr_mar.SetSpatialFilter(None)

    distance = bufferDistance - 10
    distance2 = bufferDistance
    sum_dist = '{0}-{1}'.format(distance, distance2)

    bufferDistance = 10
    lyr_mar.SetSpatialFilter(None)


    # 3. km of private roads. ##########################################################################################
    '''
    lyr_road.SetAttributeFilter(None)
    lyr_road.SetAttributeFilter("FUNCTIONAL = 'Private'")
    lyr_road.ResetReading()
    road_p_feat = lyr_road.GetNextFeature()
    road_p_geom = road_p_feat.geometry()

    intersect_priv = road_p_geom.Intersection(p_geom)
    intersect_priv_len = intersect_priv.Length()
   #print(intersect_priv_len )

    lyr_road.SetAttributeFilter(None)
    lyr_road.ResetReading()
    '''
    lyr_road.SetAttributeFilter(None)
    lyr_road.SetAttributeFilter("FUNCTIONAL = 'Private'")

    for pr in lyr_road:
        pr_geom = pr.GetGeometryRef().Clone()
        intersect_priv = pr_geom.Intersection(p_geom)
        intersect_priv_len = intersect_priv.Length() / 1000

    lyr_road.ResetReading()
    lyr_road.SetAttributeFilter(None)

    # 4. km of local roads. ############################################################################################
    '''
    lyr_road.SetAttributeFilter("FUNCTIONAL = 'Local Roads'")
    road_l_feat = lyr_road.GetNextFeature()
    road_l_geom = road_l_feat.geometry()

    intersect_loc = road_l_geom.Intersection(p_geom)
    intersect_loc_len = intersect_loc.Length()

    #print(intersect_loc_len)
    lyr_road.SetAttributeFilter(None)
    lyr_road.ResetReading()
    '''
    lyr_road.SetAttributeFilter(None)
    lyr_road.SetAttributeFilter("FUNCTIONAL = 'Local Roads'")

    for lr in lyr_road:
        lr_geom = lr.GetGeometryRef().Clone()
        intersect_loc = lr_geom.Intersection(p_geom)
        intersect_loc_len = intersect_loc.Length() / 1000

    lyr_road.ResetReading()
    lyr_road.SetAttributeFilter(None)


    # 5. Mean elevation. ###############################################################################################
    #Several option appeared to useful, yet none has worked
    #Best Option: gdla.warp crop_to_cutline
    #https://gis.stackexchange.com/questions/262021/how-to-replicate-gdalwarp-cutline-in-python-to-cut-out-tif-image-based-on-the-e
    #https://gis.stackexchange.com/questions/45053/gdalwarp-cutline-along-with-shapefile
    #https://stackoverflow.com/questions/23769531/gtiff-mask-with-shapefile-in-python-with-gdal-ogr-etc
    #Issues: Invalid polygons


    # 6. Parcel on public land. ########################################################################################
    for pl in lyr_land:
        land_geom = pl.GetGeometryRef().Clone()
        if p_geom.Intersects(land_geom):
            public = 1
        else:
            public = 0

    lyr_land.ResetReading()


    # 7. Proportion of parcel in THP. ##################################################################################
    if union_poly.Intersects(p_geom):
        union_poly_c = union_poly.Buffer(0.0)   # this helps to deal with invalid polygons
        intersect_tim = union_poly_c.Intersection(p_geom)

        area_intersect_tim = intersect_tim.GetArea()
        prop_tim = area_intersect_tim / area_parcel

    else:
        prop_tim = 0

    lyr_timber.ResetReading()


    # 8. Write everything into the dataframe ###########################################################################
    if feat_count == 0:
        df = df.append({'Parcel_APN': p.GetField('APN'),
                        'NR_GH_Plants': g_plants_tot,
                        'NR_OD_Plants': o_plants_tot,
                        'Dist_to_grow_m': 0,
                        'Km_Priv.Road': intersect_priv_len,
                        'Km_Loc.Road': intersect_loc_len,
                        'Mean_Elevation': 'Not yet',
                        'PublicLand_YN': public,
                        'Prop_in_THP': prop_tim},
                       ignore_index=True)

    else:
        df = df.append({'Parcel_APN': p.GetField('APN'),
                        'NR_GH_Plants': g_plants_tot,
                        'NR_OD_Plants': o_plants_tot,
                        'Dist_to_grow_m': sum_dist,
                        'Km_Priv.Road': intersect_priv_len,
                        'Km_Loc.Road': intersect_loc_len,
                        'Mean_Elevation': 'Not yet',
                        'PublicLand_YN': public,
                        'Prop_in_THP': prop_tim},
                       ignore_index=True)

lyr_par.ResetReading()

df.to_csv(path_data_folder + "Classic_Weed_Analysis.csv", index=False, float_format='%.2f', encoding='utf-8-sig')

# ####################################### END TIME-COUNT AND PRINT TIME STATS##################
print("")
endtime = time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime())
print("--------------------------------------------------------")
print("start: " + starttime)
print("end: " + endtime)
print("")