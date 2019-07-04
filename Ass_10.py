# ############################################################################################################# #
# Raw-example for a basic script to be executed in python. The format is only a recommendation and everyone is  #
# encouraged to modify the scripts to her/his own needs.                                                        #
# (c) Matthias Baumann, Humboldt-Universität zu Berlin, 4/15/2019
#
# Assignment 10: Marius Derenthal
#
# ####################################### WORKFLOW ############################################################ #
"""
0. load Data
1. separate dependent from independent
2. standardize predictors (z-transform)
3. split data set into training and validation data
4. hyper parameter selection (grid search)
5. select best model parameters and fit
6. classification report
7. apply model to raster
8. plot classified map
9.
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

sns.set()

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
# 0. load Data
# landsat_metrics
land = gdal.Open(path_data_folder + landsat_p)
gt_land = land.GetGeoTransform()
land_arr_all = land.ReadAsArray()

# samples
samp_df = pd.read_csv(path_data_folder + samples)

# ####################################### PROCESSING ########################################################## #
# 1. separate dependent from independent-------------------------------------------------------------------------
y_df = x_np = samp_df.iloc[:, 0]
x_df = samp_df.iloc[:, 1:]


# 2. standardize predictors (z-transform)------------------------------------------------------------------------
x_df_scaled = StandardScaler().fit(x_df.astype('float64')).transform(x_df.astype('float64'))


# 3. split data set into training and validation data------------------------------------------------------------
Xtrain, Xtest, ytrain, ytest = train_test_split(x_df_scaled, y_df, test_size=0.5, random_state=42)


# 4. hyper parameter selection (grid search)---------------------------------------------------------------------
svc = SVC(kernel='rbf', class_weight='balanced')
svc.get_params()

# grid search cross-validation to explore combinations of parameters
param_grid = {'C': [1, 5, 10, 50, 100, 1000],
              'gamma': [0.0001, 0.0005, 0.001, 0.005]}

grid = GridSearchCV(svc, param_grid, cv=5, iid=False)
grid.fit(Xtrain, ytrain)


# 5. select best model parameters and fit------------------------------------------------------------------------
model = grid.best_estimator_
yfit = model.predict(Xtest)


# 6. classification report---------------------------------------------------------------------------------------
print(classification_report(ytest, yfit,
                            #target_names=faces.target_names
                            ))

mat = confusion_matrix(ytest, yfit)
sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False
            #,xticklabels=faces.target_names,
            #yticklabels=faces.target_names
            )
plt.xlabel('true label')
plt.ylabel('predicted label')
plt.show()

# 7. apply to raster---------------------------------------------------------------------------------------------
# reduce (reshape and reorder axis from 30|1500|1500 to 1000000|30)
land_rs = land_arr_all.reshape((land_arr_all.shape[0], -1))
land_rs = land_rs.transpose()

# standardize (z-transform)
land_rs = StandardScaler().fit(land_rs.astype('float64')).transform(land_rs.astype('float64'))

# apply model to raster
land_fit = model.predict(land_rs)
land_fit = land_fit.transpose()

# back transform into 1500|1500
land_fit = land_fit.reshape(int(math.sqrt(land_fit.shape[0])), -1)

# 8. plot classified map-----------------------------------------------------------------------------------------
'''
# plot land cover map
lcColors = {1: [1.0, 0.0, 0.0, 1.0],  # Artificial
            2: [1.0, 1.0, 0.0, 1.0],  # Cropland
            3: [0.78, 0.78, 0.39, 1.0],  # Grassland
            4: [0.0, 0.78, 0.0, 1.0],  # Forest, broadleaf
            5: [0.0, 0.39, 0.39, 1.0],  # Forest, conifer
            6: [0.0, 0.0, 1.0, 1.0]}  # Water

index_colors = [lcColors[key] if key in lcColors else (1, 1, 1, 0) for key in range(1, land_fit.max()+1)]

cmap = plt.matplotlib.colors.ListedColormap(index_colors, 'Classification', land_fit.max())



# prepare labels and patches for the legend
#labels = ['artificial land', 'cropland', 'grassland', 'forest broadleaved', 'forest coniferous', 'water']
#patches = [patches.Patch(color=index_colors[i], label=labels[i]) for i in range(len(labels))]

# put those patched as legend-handles into the legend
#plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., ncol=1, frameon=False)

#plt.imshow(land_fit, cmap=cmap, interpolation='none')
#plt.show()
'''


# 9. export land cover map as GeoTiff----------------------------------------------------------------------------
ds = gdal.Open(path_data_folder + landsat_p)

drv = gdal.GetDriverByName('GTiff')
dst_ds = drv.Create(path_data_folder + "classification_output.tif", 1500, 1500, 1, gdal.GDT_Byte, [])

dst_band = dst_ds.GetRasterBand(1)

# write the data
dst_band.WriteArray(land_fit, 0, 0)

colors = dict((
    (1, (255, 0, 0)),  # Artificial
    (2, (255, 255, 0)),  # Cropland
    (3, (200, 200, 100)),  # Grassland
    (4, (0, 200, 0)),  # Forest broadleaf
    (5, (0, 100, 100)),  # Forest conifer
    (6, (0, 0, 255))  # Water
))

ct = gdal.ColorTable()
for key in colors:
    ct.SetColorEntry(key, colors[key])

dst_band.SetColorTable(ct)

# flush data to disk, set the NoData value and calculate stats
dst_band.FlushCache()

dst_ds.SetGeoTransform(ds.GetGeoTransform())
dst_ds.SetProjection(ds.GetProjectionRef())

dst_ds = None

# ####################################### END TIME-COUNT AND PRINT TIME STATS##################
print("")
endtime = time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime())
print("--------------------------------------------------------")
print("start: " + starttime)
print("end: " + endtime)
print("")
