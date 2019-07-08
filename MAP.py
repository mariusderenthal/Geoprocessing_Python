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
- reproject landsat into modis projection (instead of GeoTIFF we use Memory to do it on the fly)
- resample landsat to MODIS resolution (book chapter - mean )
- reclassify landsat image into classes (strata) or continious data stratify
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
import gdal
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
path_data_folder = '/Users/mariusderenthal/Google Drive/Global Change Geography/4.Semester/Geoprocessing_Python/Data/Assignment10_data/'
landsat_p = "landsat8_metrics1416.tif"
samples = 'landsat8_metrics1416_samples.csv'

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


# ####################################### PROCESSING ########################################################## #


# ####################################### END TIME-COUNT AND PRINT TIME STATS##################
print("")
endtime = time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime())
print("--------------------------------------------------------")
print("start: " + starttime)
print("end: " + endtime)
print("")
