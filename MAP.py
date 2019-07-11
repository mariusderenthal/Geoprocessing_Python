# ############################################################################################################# #
# Raw-example for a basic script to be executed in python. The format is only a recommendation and everyone is  #
# encouraged to modify the scripts to her/his own needs.                                                        #
# (c) Matthias Baumann, Humboldt-Universität zu Berlin, 4/15/2019
#
# MAP: Marius Derenthal
#
# ####################################### WORKFLOW ############################################################ #
"""
1. Generate a stratified random sample of 500 points across the MODIS extent.
- reproject landsat into modis projection (instead of GeoTIFF we use Memory to do it on the fly)                $$$
- resample landsat to MODIS resolution (book chapter - mean )                                                   $$$
- reclassify landsat image into classes (strata) or continuous data stratify
- create mask using valid MODIS raster
- apply mask to Landsat image
- generate sample locations
- extract xy cooridnates and apply minimium distance filter


2. Extract the mean Landsat-based tree-cover for each MODIS pixel sample. This information will be
your reference dataset.
- extract tree cover from previously sampled random cooridnates

3. Extract the MODIS EVI values for each band (time step) and each sample.
- iterate through MODIS images

4. Parameterize a gradient boosting regression using the scikit-learn package, using the built-in
functions that help finding the best parameter combination. The model performance should be
printed to the console.
- Assignment 10

5. Make the tree-cover model prediction across all MODIS tiles by applying parallel processing.
- Last session
- working function


6. The output that your script generates should be stored in a new sub-folder called output.
"""
# ####################################### LOAD REQUIRED LIBRARIES ############################################# #
import time
import os
import gdal, gdalconst
#import gdal_merge as gm

import subprocess, glob
import math
import csv
from osgeo import ogr
from osgeo import osr
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

from matplotlib import colors, patches
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns

# ####################################### SET TIME-COUNT ###################################################### #
starttime = time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime())
print("--------------------------------------------------------")
print("Starting process, time: " + starttime)
print("")
# ####################################### FUNCTIONS ########################################################### #

# ####################################### FOLDER PATHS & global variables ##################################### #
# Folder containing the working data
path_data_folder = '/Users/mariusderenthal/Google Drive/Global Change Geography/4.Semester/Geoprocessing_Python/MAP/MAP_data/'
landsat = "Landsat_TC/CHACO_TreeCover2015.tif"
modis = 'MODIS_EVI/'

# create folder to save intermediate results
inter = (path_data_folder + 'inter')
if not os.path.exists(inter):
    os.makedirs(inter)

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
land = gdal.Open(path_data_folder + landsat)
#gt_land = land.GetGeoTransform()
#landsat_arr = land.ReadAsArray()
# ####################################### PROCESSING ########################################################## #
# mosaicing
# https://gis.stackexchange.com/questions/44003/python-equivalent-of-gdalbuildvrt#44048
tiles_list = glob.glob(path_data_folder + modis +"*.tif")
output = (path_data_folder + '/inter/mosaic.tif')
vrt_options = gdal.BuildVRTOptions(resampleAlg='cubic', addAlpha=True)
mosaic = gdal.BuildVRT(output, tiles_list, options=vrt_options)

# Resampling
# https://gis.stackexchange.com/questions/234022/resampling-a-raster-from-python-without-using-gdalwarp
inputfile = (path_data_folder + landsat)
land = gdal.Open(inputfile)
inputProj = land.GetProjection()
inputTrans = land.GetGeoTransform()

referenceProj = mosaic.GetProjection()
referenceTrans = mosaic.GetGeoTransform()
bandreference = mosaic.GetRasterBand(1)
x = mosaic.RasterXSize
y = mosaic.RasterYSize

resampled = (path_data_folder + 'inter/resampled.tif')
driver= gdal.GetDriverByName('GTiff')
resamp = driver.Create(resampled,x,y,1,bandreference.DataType)
resamp.SetGeoTransform(referenceTrans)
resamp.SetProjection(referenceProj)

gdal.ReprojectImage(land,resamp,inputProj,referenceProj,gdalconst.GRA_Average) # use mean for resampling approach
#del resampled


# create mask
modis_arr  = mosaic.ReadAsArray()
print(modis_arr.shape)
#modis_arr = np.delete(modis_arr, 47, axis=0) # mosaicing adds a extra band which has to be removed beforehand
modis_arr = modis_arr[0:46,:,:]
print("MODIS SHAPE: ",modis_arr.shape)
#print(modis_arr.ndim)
print("MODIS MIN: ",modis_arr.min())
print("MODIS MAX: ",modis_arr.max())

nodata = -9999


# apply mask
resamp_arr = resamp.ReadAsArray()
print("LANDSAT SHAPE: ",resamp_arr.shape)
#print(modis_arr.ndim)
print("LANDSAT MIN: ",resamp_arr.min())
print("LANDSAT MAX: ",resamp_arr.max())

nodatamask = modis_arr == nodata
#print(nodatamask.shape)
#print(nodatamask)



#landsat_arr_m = resamp_arr[nodatamask] = nodata

print("----------------------------------------------")

# reclassify
#print(resamp_arr)

#resamp_arr[resamp_arr < 0] = 0
resamp_arr[(resamp_arr >= 0) &(resamp_arr < 1000)] = 1
resamp_arr[(resamp_arr >= 1000) & (resamp_arr < 2000)] = 2
resamp_arr[(resamp_arr >= 2000) & (resamp_arr < 3000)] = 3
resamp_arr[(resamp_arr >= 3000) & (resamp_arr < 4000)] = 4
resamp_arr[(resamp_arr >= 4000) & (resamp_arr < 5000)] = 5
resamp_arr[(resamp_arr >= 5000) & (resamp_arr < 6000)] = 6
resamp_arr[(resamp_arr >= 6000) & (resamp_arr < 7000)] = 7
resamp_arr[(resamp_arr >= 7000) & (resamp_arr < 8000)] = 8
resamp_arr[(resamp_arr >= 8000) & (resamp_arr < 9000)] = 9
resamp_arr[(resamp_arr >= 9000) & (resamp_arr < 10000)] = 10
#resamp_arr[resamp_arr >= 10000 ] = 11

print(resamp_arr)
'''
driver = gdal.GetDriverByName('GTiff')
file = gdal.Open(path_data_folder + landsat)
band = file.GetRasterBand(1)
lista = band.ReadAsArray()
resamp_arr = lista

# reclassification
for j in  range(file.RasterXSize):
    for i in  range(file.RasterYSize):
        if lista[i,j] >= 0 and lista[i,j] < 1000:
            lista[i,j] = 1
        elif lista[i,j] >= 1000 and lista[i,j] < 2000:
            lista[i,j] = 2
        elif lista[i, j] >= 2000 and lista[i, j] < 3000:
            lista[i,j] = 3
        elif lista[i,j] >= 3000 and lista[i,j] < 4000:
            lista[i,j] = 4
        elif lista[i,j] >= 4000 and lista[i,j] < 5000:
            lista[i,j] = 5
        elif lista[i,j] >= 5000 and lista[i,j] < 6000:
            lista[i,j] = 6
        elif lista[i,j] >= 6000 and lista[i,j] < 7000:
            lista[i,j] = 7
        elif lista[i,j] >= 7000 and lista[i,j] < 8000:
            lista[i,j] = 8
        elif lista[i,j] >= 8000 and lista[i,j] < 9000:
            lista[i,j] = 9
        elif lista[i,j] >= 9000 and lista[i,j] <= 10000:
            lista[i,j] = 10
        elif lista[i,j] >= 10001:
            lista[i,j] = 11
    print(j)

lista[np.where(resamp_arr < 0)] = 0
lista[np.where(resamp_arr >= 0 & resamp_arr < 1000)] = 1
lista[np.where(resamp_arr >= 1000 & resamp_arr < 2000)] = 2
lista[np.where(resamp_arr >= 2000 & resamp_arr < 3000)] = 3
lista[np.where(resamp_arr >= 3000 & resamp_arr < 4000)] = 4
lista[np.where(resamp_arr >= 4000 & resamp_arr < 5000)] = 5
lista[np.where(resamp_arr >= 5000 & resamp_arr < 6000)] = 6
lista[np.where(resamp_arr >= 6000 & resamp_arr < 7000)] = 7
lista[np.where(resamp_arr >= 7000 & resamp_arr < 8000)] = 8
lista[np.where(resamp_arr >= 8000 & resamp_arr < 9000)] = 9
lista[np.where(resamp_arr >= 9000 & resamp_arr <= 10000)] = 10
'''

# create new file
file2 = driver.Create(path_data_folder + 'inter/reclassified.tif', resamp.RasterXSize , resamp.RasterYSize , 1)
file2.GetRasterBand(1).WriteArray(resamp_arr)

# spatial ref system
proj = resamp.GetProjection()
georef = resamp.GetGeoTransform()
file2.SetProjection(proj)
file2.SetGeoTransform(georef)
file2.FlushCache()



# ####################################### END TIME-COUNT AND PRINT TIME STATS##################
print("")
endtime = time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime())
print("--------------------------------------------------------")
print("start: " + starttime)
print("end: " + endtime)
print("")
