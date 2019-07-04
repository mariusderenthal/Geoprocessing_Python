# ############################################################################################################# #
# Raw-example for a basic script to be executed in python. The format is only a recommendation and everyone is  #
# encouraged to modify the scripts to her/his own needs.                                                        #
# (c) Matthias Baumann, Humboldt-Universität zu Berlin, 4/15/2019
#
# Assignment 09: Marius Derenthal
#
# ####################################### WORKFLOW ############################################################ #
'''
First Excersice
1. Read as numpy array
  - get unique values in the land cover map
  - save unique values in an array
2. Based on landcovermap create startified random sample
 - apply spatial filter (every 20 pixel) to ensure 500m sample distance
 - landcover_class[x::17, y::17]   # pick random x and y between 1-16

 - use numpy.where function to subset classes (1-12)
    - for x in np.arange(12)+1
    - for x in np.unique(landcover_image)
        inds = np.where(landcover_map == x) # output are the indices, but in which format? add transpose ... then there should be arrays (containing the actual information) in a 1d arrays
        (row = y dimension; col = x dimension)
        i = np.random.choice( inds.shape[0] #length of our array, 1000, replace = False)      #we have to create a index vector ...
        indsselected = inds[i, :]




3. take indices from randomly sampled points
4. extract pixel values in raster
 - calculate mean for each band
5. plot values for each class

'''
# ####################################### LOAD REQUIRED LIBRARIES ############################################# #
import time
import os
import gdal
from osgeo import ogr
from osgeo import osr
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ####################################### SET TIME-COUNT ###################################################### #
starttime = time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime())
print("--------------------------------------------------------")
print("Starting process, time: " + starttime)
print("")
# ####################################### FUNCTIONS ########################################################### #
# ####################################### FOLDER PATHS & global variables ##################################### #
# Folder containing the working data
path_data_folder = '/Users/mariusderenthal/Google Drive/Global Change Geography/4.Semester/Geoprocessing_Python/Data/Assignment09_data/'
landcover_p = "landcover_lucas2015_15000_10000.tif"
landsat_p = "landsat_median1416_15000_10000.tif"
#LUCAS_p = "EU28_2015_20161028_lucas2015j.gpkg"
LUCAS_p = "LUCAS_Transformed.gpkg"

# ####################################### PROJECTION ########################################################## #
'''
# Transform LUCAS points projection

# EPSGS
# Landsat: 3035
# LUCAS: 4326

driver = ogr.GetDriverByName('GPKG')

# input SpatialReference
inSpatialRef = osr.SpatialReference()
inSpatialRef.ImportFromEPSG(4326)

# output SpatialReference
outSpatialRef = osr.SpatialReference()
outSpatialRef.ImportFromEPSG(3035)

# create the CoordinateTransformation
coordTrans = osr.CoordinateTransformation(inSpatialRef, outSpatialRef)

# get the input layer
inDataSet = driver.Open(path_data_folder + LUCAS_p)      #SET PATH
inLayer = inDataSet.GetLayer()

# create the spatial reference, 26741
srs = osr.SpatialReference()
srs.ImportFromEPSG(3035)

# create the output layer
outputShapefile = (path_data_folder + 'LUCAS_Transformed.gpkg') #SET NAME
if os.path.exists(outputShapefile):
    driver.DeleteDataSource(outputShapefile)
outDataSet = driver.CreateDataSource(outputShapefile)
outLayer = outDataSet.CreateLayer("LUCAS_Transformed", srs, geom_type=ogr.wkbPoint)    #SELECT GEOMETRY

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
# ####################################### Data ################################################################ #
# Landcover
lc = gdal.Open(path_data_folder + landcover_p)
lc_arr_all = lc.ReadAsArray()

# Landsat
land = gdal.Open(path_data_folder + landsat_p)
gt_land = land.GetGeoTransform()
land_rb=land.GetRasterBand(1)
land_arr_all = land.ReadAsArray()

# LUCAS
lucas = ogr.Open(path_data_folder + LUCAS_p, 0)
if lucas is None:
    sys.exit('Could not open {0}.'.format(path_data_folder + LUCAS_p))
lyr_lucas = lucas.GetLayer()

# ####################################### PROCESSING ########################################################## #
# dataframe for collecting results
df = pd.DataFrame(columns={#'LC_Class': [],
                           'Band02': [],
                           'Band03': [],
                           'Band04': [],
                           'Band05': [],
                           'Band06': [],
                           'Band07': []})

# set seed to make results replicable
np.random.seed(seed=8)

# implement 500 meter buffer by slicing into steps of 17 pixels (17*30 = 510)
# select random starting point
x = np.random.randint(low=0, high=16, size=1)
y = np.random.randint(low=0, high=16, size=1)

lc_arr = lc_arr_all[int(y)::16, int(x)::16]
#print(lc_arr.ndim)
#print(lc_arr.shape)

land_arr = land_arr_all[:,int(y)::16, int(x)::16]
#print(land_arr.ndim)
#print(land_arr.shape)

for classes in np.unique(lc_arr):
        inds = np.transpose(np.where(lc_arr == classes))

        if inds.shape[0] >= 1000:
            samplesize = 1000
        else:
            samplesize = inds.shape[0]

        i = np.random.choice(inds.shape[0], samplesize, replace=False)
        indsselected = inds[i, :]
        pixel_v = land_arr[:, indsselected[:,0], indsselected[:,1]]
        pixel_m = pixel_v.mean(axis=1)


        df = df.append({#'LC_Class': classes,
                           'Band02': pixel_m[0],
                           'Band03': pixel_m[1],
                           'Band04': pixel_m[2],
                           'Band05': pixel_m[3],
                           'Band06': pixel_m[4],
                           'Band07': pixel_m[5]},
                       ignore_index=True)

df_trans = df.T
#print(df.to_string())
#print(df_trans.to_string())
df_trans.plot(style='.-')
plt.show()

# ####################################### PROCESSING ########################################################## #
# define final dataframe for collecting results and later export
df2 = pd.DataFrame(columns={'class': [],
                            'x':     [],
                            'y':     [],
                           'Band02': [],
                           'Band03': [],
                           'Band04': [],
                           'Band05': [],
                           'Band06': [],
                           'Band07': []
                           })

#Bounding box around raster image
#create empty shape file to store stuff in
# set up the shapefile driver
driver = ogr.GetDriverByName('GPKG')
# create the data source
data_source = driver.CreateDataSource("rasterbox_Poly.gpkg")
# create the spatial reference, WGS84
srs = osr.SpatialReference()
srs.ImportFromEPSG(3035)

# create the layer
layer = data_source.CreateLayer("", srs, ogr.wkbPolygon)

# create the bounding box ring
ulx, xres, xskew, uly, yskew, yres  = land.GetGeoTransform()
lrx = ulx + (land.RasterXSize * xres)
lry = uly + (land.RasterYSize * yres)

# create ring geometry and fill with points
ringy = ogr.Geometry(ogr.wkbLinearRing)

ringy.AddPoint(ulx, uly)
ringy.AddPoint(lrx, uly)
ringy.AddPoint(lrx, lry)
ringy.AddPoint(ulx, lry)
ringy.AddPoint(ulx, uly)

# create polygon geometry and fill with ring
poly = ogr.Geometry(ogr.wkbPolygon)
poly.AddGeometry(ringy)

# create the feature
feature = ogr.Feature(layer.GetLayerDefn())
# set the feature geometry
feature.SetGeometry(poly)
# create the feature in the layer
layer.CreateFeature(feature)


# apply spatial filter to points
for poly in layer:
    lyr_lucas.SetSpatialFilter(poly.geometry())


#Extract pixel values at point location
#https://gis.stackexchange.com/questions/46893/getting-pixel-value-of-gdal-raster-under-ogr-point-without-numpy
for pts in lyr_lucas:
    geom = pts.GetGeometryRef()
    mx, my = geom.GetX(), geom.GetY()

    # Convert from map to pixel coordinates.
    # Only works for geotransforms with no rotation.
    px = int((mx - gt_land[0]) / gt_land[1])  # x pixel
    py = int((my - gt_land[3]) / gt_land[5])  # y pixel

    # extract classes
    for classes in range(lc.RasterCount):
        classes += 1
        class_rb = lc.GetRasterBand(classes)
        intval_lc = class_rb.ReadAsArray(py, px, 1, 1)


    # extract band values
    band_values = []

    for band in range(land.RasterCount):
        band += 1
        land_rb = land.GetRasterBand(band)
        intval = land_rb.ReadAsArray(py, px, 1, 1)
        band_values.append(int(intval[0]))


    df2 = df2.append({'class': int(intval_lc[0]),
                        'x': mx,
                        'y': my,
                        'Band02': band_values[0],
                        'Band03': band_values[1],
                        'Band04': band_values[2],
                        'Band05': band_values[3],
                        'Band06': band_values[4],
                        'Band07': band_values[5]},
                     ignore_index=True)

lyr_lucas.SetSpatialFilter(None)



#Save in dataframe
df2.to_csv(path_data_folder + "LUCAS_median.csv", index=False, float_format='%.2f', encoding='utf-8-sig')

# ####################################### END TIME-COUNT AND PRINT TIME STATS##################
print("")
endtime = time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime())
print("--------------------------------------------------------")
print("start: " + starttime)
print("end: " + endtime)
print("")