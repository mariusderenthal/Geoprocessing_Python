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
- reclassify landsat image into classes (strata) or continuous data stratify                                    $$$
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
# import gdal_merge as gm

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
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error
from sklearn.metrics import confusion_matrix
from sklearn import metrics

from matplotlib import colors, patches
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns
from joblib import Parallel, delayed

# ####################################### SET TIME-COUNT ###################################################### #
starttime = time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime())
print("--------------------------------------------------------")
print("Starting process, time: " + starttime)
print("")


# ####################################### FUNCTIONS ########################################################### #
# https://stackoverflow.com/questions/53218931/how-to-unnest-explode-a-column-in-a-pandas-dataframe
def unnesting(df, explode):
    idx = df.index.repeat(df[explode[0]].str.len())
    df1 = pd.concat([
        pd.DataFrame({x: np.concatenate(df[x].values)}) for x in explode], axis=1)
    df1.index = idx

    return df1.join(df.drop(explode, 1), how='left')

#https://stackoverflow.com/questions/52443906/pixel-array-position-to-lat-long-gdal-python
def pixel2coord(x, y, raster):
    xoff, a, b, yoff, d, e = raster.GetGeoTransform()

    xp = a * x + b * y + a * 0.5 + b * 0.5 + xoff
    yp = d * x + e * y + d * 0.5 + e * 0.5 + yoff
    return(xp, yp)


def compositingF(raster_path):

    ras_file = gdal.Open(raster_path)
    ras_arr = ras_file.ReadAsArray()
    #ras_arr[(ras_arr == -9999)] = np.nan

    meani = np.nanmean(ras_arr, axis=0)
    meadi = np.nanmedian(ras_arr, axis=0)
    stdi = np.nanstd(ras_arr, axis=0)
    # vari = np.nanvar(modis_arr, axis=0)
    maxi = np.nanmax(ras_arr, axis=0)
    mini = np.nanmin(ras_arr, axis=0)
    # percenti25 = np.nanpercentile(modis_arr,25, axis=0)
    # percenti50 = np.nanpercentile(modis_arr,50, axis=0)
    # percenti75 = np.nanpercentile(modis_arr,75, axis=0)

    compi_arr = np.stack((meani,
                          #meadi,
                          stdi,
                          # ,vari
                          maxi,
                          mini
                          # ,percenti25
                          # ,percenti50
                          # ,percenti75
    ))
    return compi_arr

def classifyF (raster, raster_path):

    ras_file = gdal.Open(raster_path)
    x_size = ras_file.RasterXSize
    y_size = ras_file.RasterYSize


    # reduce (reshape and reorder axis from 30|1500|1500 to 1000000|30)
    # print("composite Shape: ", compi_arr.shape)
    land_rs = raster.reshape((raster.shape[0], -1))
    land_rs = land_rs.transpose()

    # standardize (z-transform)
    land_rs = StandardScaler().fit(land_rs.astype('float64')).transform(land_rs.astype('float64'))

    #fill nas with mean
    imp = SimpleImputer(missing_values=np.nan, strategy='constant')
    imp = imp.fit(land_rs)
    land_rs_imp = imp.transform(land_rs)
    # land_rs.fillna(land_rs.mean(), inplace=True)

    # apply model to raster
    land_fit = model.predict(land_rs_imp)
    land_fit = land_fit.transpose()

    # back transform into 1500|1500
    land_fit = land_fit.reshape(y_size, x_size)
    # print("classification Shape: ", land_fit.shape)
    return land_fit



# ####################################### FOLDER PATHS & global variables ##################################### #
# Folder containing the working data
path_data_folder = '/Users/mariusderenthal/Google Drive/Global Change Geography/4.Semester/Geoprocessing_Python/MAP/MAP_data/'
landsat = 'Landsat_TC/CHACO_TreeCover2015.tif'
modis = 'MODIS_EVI/'
modis_tile = 'MODIS_EVI_h0_v2.tif'
inter = 'inter/'
output = 'output/'

# create folder to save intermediate results
inter_folder = (path_data_folder + 'inter')
if not os.path.exists(inter_folder):
    os.makedirs(inter_folder)

output_folder = (path_data_folder + 'output')
if not os.path.exists(output_folder):
    os.makedirs(output_folder)



# set seed to make results replicable
np.random.seed(seed=8)

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
# gt_land = land.GetGeoTransform()
# landsat_arr = land.ReadAsArray()
###################modis_tile_test = gdal.Open(path_data_folder + modis + modis_tile)
####################modis_tile_test_array = modis_tile_test.ReadAsArray()


# ####################################### PROCESSING ########################################################## #

# Step 1: mosaicing ----------------------------------------------------------------------------------------------------
# https://gis.stackexchange.com/questions/44003/python-equivalent-of-gdalbuildvrt#44048
tiles_list = glob.glob(path_data_folder + modis +"*.tif")
output_mosaic = path_data_folder + '/inter/mosaic.tif'
vrt_options = gdal.BuildVRTOptions(resampleAlg='cubic', addAlpha=True)     #resampling Algorithm does not matter, as tiles do not overlap
mosaic = gdal.BuildVRT(output_mosaic, tiles_list, options=vrt_options)

#mosaic = gdal.Open(path_data_folder + inter +"/" + "mosaic.tif")     #shortcut after first successful run
modis_arr = mosaic.ReadAsArray()
print("Step 1: mosaicing DONE", time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime()))


#########################pat_out_test = compositingF(path_data_folder + modis + modis_tile)



# Step 2: resampling ---------------------------------------------------------------------------------------------------
# https://gis.stackexchange.com/questions/234022/resampling-a-raster-from-python-without-using-gdalwarp
land = gdal.Open(path_data_folder + landsat)
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

gdal.ReprojectImage(land,resamp,inputProj,referenceProj,gdalconst.GRA_Average)     #use mean for resampling approach
del resampled

#resamp = gdal.Open(path_data_folder + inter +"/" + "resampled.tif")     #shortcut after first successful run
resamp_arr = resamp.ReadAsArray()
print("Step 2: resampling DONE", time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime()))





# Step 3: reclassification ---------------------------------------------------------------------------------------------
reclass_arr = resamp_arr.copy()

reclass_arr[(reclass_arr >= 0) & (reclass_arr < 2000)] = 1
reclass_arr[(reclass_arr >= 2000) & (reclass_arr < 4000)] = 2
reclass_arr[(reclass_arr >= 4000) & (reclass_arr < 6000)] = 3
reclass_arr[(reclass_arr >= 6000) & (reclass_arr < 8000)] = 4
reclass_arr[(reclass_arr >= 8000) & (reclass_arr < 10000)] = 5
""" Optional export of results
resamp_arr[(resamp_arr >= 10000) & (resamp_arr < 6000)] = 6
resamp_arr[(resamp_arr >= 6000) & (resamp_arr < 7000)] = 7
resamp_arr[(resamp_arr >= 7000) & (resamp_arr < 8000)] = 8
resamp_arr[(resamp_arr >= 8000) & (resamp_arr < 9000)] = 9
resamp_arr[(resamp_arr >= 9000) & (resamp_arr < 10000)] = 10

# create new file
driver= gdal.GetDriverByName('GTiff')
file2 = driver.Create(path_data_folder + 'inter/reclassified.tif', resamp.RasterXSize , resamp.RasterYSize , 1)
file2.GetRasterBand(1).WriteArray(resamp_arr)

# spatial ref system
proj = resamp.GetProjection()
georef = resamp.GetGeoTransform()
file2.SetProjection(proj)
file2.SetGeoTransform(georef)
file2.FlushCache()
print("Done reclassifying the landsat image at:", time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime()))
"""
# reclass = gdal.Open(path_data_folder + inter +"/" + "reclassified.tif")     #shortcut after first successful run
# reclass_arr = reclass.ReadAsArray()
print("Step 3: reclassification DONE", time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime()))





# Step 4: compositing --------------------------------------------------------------------------------------------------
compi_output = Parallel(n_jobs=3)(delayed(compositingF)(i) for i in tiles_list)

#manual steps
nodata = -9999
# get rid of NAs -create composites
modis_arr = modis_arr[0:46,:,:]
modis_arr[(modis_arr == nodata)] = np.nan

meani = np.nanmean(modis_arr, axis=0)
#print("Done calculating mean:", time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime()))
meadi = np.nanmedian(modis_arr, axis=0)
stdi = np.nanstd(modis_arr, axis=0)
#print("Done calculating std:", time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime()))
#vari = np.nanvar(modis_arr, axis=0)
#print("Done calculating variance:", time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime()))
maxi = np.nanmax(modis_arr, axis=0)
#print("Done calculating maximum:", time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime()))
mini = np.nanmin(modis_arr, axis=0)
#print("Done calculating minimum:", time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime()))
#percenti25 = np.nanpercentile(modis_arr,25, axis=0)
#percenti50 = np.nanpercentile(modis_arr,25, axis=0)
#percenti75 = np.nanpercentile(modis_arr,25, axis=0)
#print("Done calculating percentiles:", time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime()))

compi_arr = np.stack((meani
                   ,stdi
                   ,meadi
                   #,vari
                   ,maxi
                   ,mini
                   #,percenti25
                   #,percenti50
                   #,percenti75
                   ))

""" Optional export of results
# create new file
driver= gdal.GetDriverByName('GTiff')
file2 = driver.Create(path_data_folder + 'inter/composite.tif', resamp.RasterXSize , resamp.RasterYSize , 5, gdal.GDT_Float32)
file2.GetRasterBand(1).WriteArray(compi_arr[0,:,:])
file2.GetRasterBand(2).WriteArray(compi_arr[1,:,:])
file2.GetRasterBand(3).WriteArray(compi_arr[2,:,:])
file2.GetRasterBand(4).WriteArray(compi_arr[3,:,:])
file2.GetRasterBand(4).WriteArray(compi_arr[4,:,:])

# spatial ref system
proj = resamp.GetProjection()
georef = resamp.GetGeoTransform()
file2.SetProjection(proj)
file2.SetGeoTransform(georef)
file2.FlushCache()
#print("Done reclassifying the landsat image at:", time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime()))
"""
#compi = gdal.Open(path_data_folder + inter +"/" + "composite.tif")     #shortcut after first successful run
#compi_arr = compi.ReadAsArray()
print("Step 4: compositing DONE", time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime()))





# Step 5: masking ------------------------------------------------------------------------------------------------------
mask1 = np.logical_not(np.isnan(compi_arr[0, :, :]))     # based on entire mosaic
reclass_arr_m = np.where(mask1 == True, reclass_arr, np.nan)     # apply mask on reclassified array
resamp_arr_m = np.where(mask1 == True, resamp_arr, np.nan)     # apply mask on original tree cover array
"""
#print(modis_arr.shape)
print("MODIS SHAPE: ",modis_arr.shape)
print("MODIS MIN: ",modis_arr.min())
print("MODIS MAX: ",modis_arr.max())
#print(modis_arr)

#modis_arr[modis_arr == np.nan] = nodata
#modis_arr = np.putmask(modis_arr, modis_arr == np.nan, nodata)
#print(modis_arr)
#np.min(modis_arr, axis=1)

minimum = np.min(modis_arr, axis=0)
print("MIN SHAPE: ",minimum.shape)
print("MIN MIN: ",minimum.min())
print("MIN MAX: ",minimum.max())
#print(minimum)

maximum = np.max(modis_arr, axis=0)
print("MAX SHAPE: ",maximum.shape)
print("MAX MIN: ",maximum.min())
print("MAX MAX: ",maximum.max())
#print(maximum)

#nodatamask = test == nodata
#print("Mask SHAPE: ",nodatamask.shape)
#print("Mask MIN: ",nodatamask.min())
#print("Mask MAX: ",nodatamask.max())

mask = maximum == nodata
#print(mask)

# apply mask
print("LANDSAT SHAPE: ",reclass_arr.shape)
print("LANDSAT MIN: ",reclass_arr.min())
print("LANDSAT MAX: ",reclass_arr.max())

#landsat_arr_m = resamp_arr[nodatamask] = nodata
landsat_arr_m = np.where(mask == True, nodata, reclass_arr)
print("Masked Landsat SHAPE: ",landsat_arr_m.shape)
print("Masked Landsat MIN: ",landsat_arr_m.min())
print("Masked Landsat MAX: ",landsat_arr_m.max())
#print(landsat_arr_m)
"""
print("Step 5: masking DONE:", time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime()))





# Step 6: sampling stratified random sampling (Ass.09)
df = pd.DataFrame(columns={'TC': [],
                           'Band01': [],
                           'Band02': [],
                           'Band03': [],
                           'Band04': [],
                           'Band05': []})

sample_p = pd.DataFrame(columns={ 'x': [],
                                  'y': [],
                                  'class': []})

for classes in np.unique(reclass_arr_m[~np.isnan(reclass_arr_m)]):
    inds = np.transpose(np.where(reclass_arr_m == classes))

    if inds.shape[0] >= 100:
        samplesize = 100
    else:
        samplesize = inds.shape[0]

    i = np.random.choice(inds.shape[0], samplesize, replace=False)
    indsselected = inds[i, :]
    pixel_v = compi_arr[:, indsselected[:, 0], indsselected[:, 1]]
    tc = resamp_arr_m[indsselected[:, 0], indsselected[:, 1]]
    # print("TreeCover: ", tc)
    # print(tc.shape)
    # print("Pixel_value",pixel_v)
    # print("Pixel_value 0", pixel_v[0])
    # print(pixel_v.shape)
    # pixel_m = pixel_v.mean(axis=1)

    df = df.append({'TC': tc,
                    'Band01': pixel_v[0],
                    'Band02': pixel_v[1],
                    'Band03': pixel_v[2],
                    'Band04': pixel_v[3]},
                   ignore_index=True)

    sample_p= sample_p.append({'x': indsselected[:, 0],
                                    'y': indsselected[:, 1],
                                    'class': classes},
                                    ignore_index=True)

    # print("Done with class: ",  classes)

df_samples = unnesting(df, ['Band01', 'Band02', 'Band03', 'Band04', 'TC'])

sample_points = unnesting(sample_p,['x','y'])
#print(sample_points.to_string())

#for index, row in sample_points.iterrows():
#    print('THIS IS MY SAMPLE POINT COLLECTION: ', row)

# df_trans = df.T
# print(df.to_string())
print("Step 6: sampling DONE:", time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime()))





# Step 7: Classification
# Step7.1: data seperation ---------------------------------------------------------------------------------------------
y_df = df_samples.iloc[:, 4]
x_df = df_samples.iloc[:, 0:4]
print("Step 7.1: data seperation DONE", time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime()))


# Step 7.2: z-transformation -------------------------------------------------------------------------------------------
x_df_scaled = StandardScaler().fit(x_df.astype('float64')).transform(x_df.astype('float64'))
print("Step 7.2: z-transformation DONE", time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime()))


# Step 7.3: data split (training, validation ---------------------------------------------------------------------------
Xtrain, Xtest, ytrain, ytest = train_test_split(x_df_scaled, y_df, test_size=0.5, random_state=42)
print("Step 7.3: data split DONE", time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime()))


# Step 7.4: hyper parameter selection (grid search)---------------------------------------------------------------------
est = GradientBoostingRegressor(n_estimators=100)
# grid search cross-validation to explore combinations of parameters
# https://www.slideshare.net/PyData/gradient-boosted-regression-trees-in-scikit-learn-gilles-louppe
param_grid = {'learning_rate': [0.1, 0.05, 0.02, 0.01],
              'max_depth': [4, 6],
              'min_samples_leaf': [1, 3, 9, 5, 17],
              'max_features': [1.0, 0.3, 0.1]}

grid = GridSearchCV(est, param_grid, cv=5, iid=False, n_jobs=3).fit(Xtrain, ytrain)
grid.get_params()
print("Step 7.4: hyper parameter selection DONE", time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime()))


# Step 7.5: fitting best model -----------------------------------------------------------------------------------------
model = grid.best_estimator_
yfit = model.predict(Xtest)
print("Step 7.5: fitting best model DONE", time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime()))





# Step 8: classification report ----------------------------------------------------------------------------------------
# print(classification_report(ytest, yfit, target_names=faces.target_names))
# print(metrics.classification_report(expected, predicted))
mse = mean_squared_error(ytest, yfit)
print("MSE: %.4f" % mse)
print("Step 8: classification report DONE", time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime()))





# Step 9: apply to raster ----------------------------------------------------------------------------------------------
class_raster = Parallel(n_jobs=3)(delayed(classifyF)(i,j) for i,j in zip(compi_output,tiles_list))
"""
# reduce (reshape and reorder axis from 30|1500|1500 to 1000000|30)
#print("composite Shape: ", compi_arr.shape)
land_rs = compi_arr.reshape((compi_arr.shape[0], -1))
land_rs = land_rs.transpose()

# standardize (z-transform)
land_rs = StandardScaler().fit(land_rs.astype('float64')).transform(land_rs.astype('float64'))

# apply model to raster
imp = SimpleImputer(missing_values=np.nan, strategy='mean')
imp = imp.fit(land_rs)
land_rs_imp = imp.transform(land_rs)
# land_rs.fillna(land_rs.mean(), inplace=True)
land_fit = model.predict(land_rs_imp)
land_fit = land_fit.transpose()

# back transform into 1500|1500
land_fit = land_fit.reshape(4800, 3000)
#print("classification Shape: ", land_fit.shape)
"""
print("Step 9: apply to raster DONE", time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime()))





# Step 10: plot classified map -----------------------------------------------------------------------------------------
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
print("Step 10: plotting NOT DONE", time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime()))





# Step 11: export land cover map as GeoTiff ----------------------------------------------------------------------------
#ds = gdal.Open(path_data_folder + inter + '/composite.tif')
drv = gdal.GetDriverByName('GTiff')

# modify file names
files_no_ext = [".".join(f.split(".")[:-1]) for f in os.listdir('../MAP/MAP_data/MODIS_EVI')]
files_no_ext.sort()
files_no_ext = files_no_ext[1:]
counter = 0
for j, i in zip(class_raster, files_no_ext):

    ras_file = gdal.Open(tiles_list[counter])
    x_size = ras_file.RasterXSize
    y_size = ras_file.RasterYSize

    dst_ds = drv.Create(path_data_folder + output + i + "_RF-results.tif", x_size, y_size, 1, gdal.GDT_Float64, [])
    dst_band = dst_ds.GetRasterBand(1)
    # write the data
    dst_band.WriteArray(j, 0, 0)

    # flush data to disk, set the NoData value and calculate stats
    dst_band.FlushCache()

    dst_ds.SetGeoTransform(ras_file.GetGeoTransform())
    dst_ds.SetProjection(ras_file.GetProjectionRef())
    counter += 1

    dst_ds = None


"""
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
"""

print("Step 11: exporting results DONE", time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime()))



# 12. export sample points as Shape------------------------------------------------------------------------------
# get the envelope
geoTransform = ds.GetGeoTransform()
xmin = geoTransform[0]
ymax = geoTransform[3]
xmax = xmin + geoTransform[1] * ds.RasterXSize
ymin = ymax + geoTransform[5] * ds.RasterYSize
#print(xmin, ymin, xmax, ymax)

#create empty shape file to store stuff in
# set up the shapefile driver
driver = ogr.GetDriverByName("ESRI Shapefile")

# create the data source
data_source = driver.CreateDataSource(path_data_folder + output + "/sample_points.shp")

# create the spatial reference, WGS84
lyr = ds.GetLayer

#sourceSR = lyr.GetSpatialRef()
#wkty = ds.GetProjectionRef.ExportToWkt()
proj = osr.SpatialReference(wkt=ds.GetProjection())
epsg = proj.GetAttrValue('AUTHORITY',1)
#print(proj.GetAttrValue('AUTHORITY',1))
srs = osr.SpatialReference()
#srs.ImportFromEPSG(4326)
srs.ImportFromWkt(ds.GetProjectionRef())
#dst_ds.SetGeoTransform(ds.GetGeoTransform())
#dst_ds.SetProjection(ds.GetProjectionRef())

# create the layer
layer = data_source.CreateLayer("", srs, ogr.wkbPoint)

# Add the fields we're interested in
Class = ogr.FieldDefn("Class", ogr.OFTString)
layer.CreateField(Class)

Sample_ID = ogr.FieldDefn("Sample_ID", ogr.OFTInteger)
layer.CreateField(Sample_ID)
counter = 0

for index, row in sample_points.iterrows():
    #print(row)
    counter += 1

    # create the feature
    feature = ogr.Feature(layer.GetLayerDefn())

    # set the attributes using the values from the delimited text file
    feature.SetField("Class", row['class'])
    feature.SetField("Sample_ID", counter)

    x_lan, y_lan = pixel2coord(row['y'], row['x'], ds)

    #x_lan = xmin + row['x'] * ds.RasterXSize
    #y_lan = ymax + row['y'] * ds.RasterXSize

    land_point = ogr.Geometry(ogr.wkbPoint)

    land_point.AddPoint(x_lan, y_lan)

    # set the feature geometry
    feature.SetGeometry(land_point)

    # create the feature in the layer
    layer.CreateFeature(feature)

print("Step 12: exporting sample points as .shp DONE", time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime()))


# ####################################### END TIME-COUNT AND PRINT TIME STATS##################
print("")
endtime = time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime())
print("--------------------------------------------------------")
print("start: " + starttime)
print("end: " + endtime)
print("")