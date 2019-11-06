# ####################################### INFORMATION ################################################################ #
# The basic script structure was taken from Matthias Baumann, Humboldt-Universität zu Berlin and has been
# modified to specific requirements.
# Autor: Marius Derenthal

# ####################################### TASK ####################################################################### #
# 1. Generate a stratified random sample of 500 points across the MODIS extent.
# 2. Extract the mean Landsat-based tree-cover for each MODIS pixel sample.
# 3. Extract the MODIS EVI values for each band (time step) and each sample.
# 4. Parameterize a gradient boosting regression using the scikit-learn package.
# 5. Make the tree-cover model prediction across all MODIS tiles by applying parallel processing.
# 6. The output that your script generates should be stored in a new sub-folder called output.

# ####################################### WORKFLOW ################################################################### #
"""
Step 1: mosaicing ------------------------------------------------------------------------------------------------------
Step 2: resampling -----------------------------------------------------------------------------------------------------
Step 3: reclassification of Landsat image ------------------------------------------------------------------------------
Step 4: compositing ----------------------------------------------------------------------------------------------------
Step 5: masking --------------------------------------------------------------------------------------------------------
Step 6: sampling stratified random sampling ----------------------------------------------------------------------------
Step 7: Classification -------------------------------------------------------------------------------------------------
Step 8: classification report ------------------------------------------------------------------------------------------
Step 9: apply to raster composites and export --------------------------------------------------------------------------
Step 10: plot classified map -------------------------------------------------------------------------------------------
Step 11: export sample points as Shape----------------------------------------------------------------------------------
"""
# ####################################### LOAD REQUIRED LIBRARIES #################################################### #
import time
import os
from pathlib import Path
import glob

from osgeo import ogr
import gdal, gdalconst
from osgeo import osr
import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error

from joblib import Parallel, delayed

# ####################################### SET TIME-COUNT ############################################################# #
starttime = time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime())
print("--------------------------------------------------------")
print("Starting process, time: " + starttime)
print("")


# ####################################### FUNCTIONS ################################################################## #
# https://stackoverflow.com/questions/53218931/how-to-unnest-explode-a-column-in-a-pandas-dataframe
def unnesting(df, explode):
    idx = df.index.repeat(df[explode[0]].str.len())
    df1 = pd.concat([
        pd.DataFrame({x: np.concatenate(df[x].values)}) for x in explode], axis=1)
    df1.index = idx

    return df1.join(df.drop(explode, 1), how='left')


# https://stackoverflow.com/questions/52443906/pixel-array-position-to-lat-long-gdal-python
def pixel2coord(x, y, raster):
    xoff, a, b, yoff, d, e = raster.GetGeoTransform()

    xp = a * x + b * y + a * 0.5 + b * 0.5 + xoff
    yp = d * x + e * y + d * 0.5 + e * 0.5 + yoff

    return (xp, yp)


def compositing_func(raster_path):
    """
    :param raster_path: entire input file path
    :return: return raster file and composite
    """
    # read raster file and create array
    ras_file = gdal.Open(raster_path)
    ras_arr = ras_file.ReadAsArray()
    # ras_arr[(ras_arr == -9999)] = np.nan

    # calculate spectral-temporal indices
    meani = np.nanmean(ras_arr, axis=0)
    meadi = np.nanmedian(ras_arr, axis=0)
    stdi = np.nanstd(ras_arr, axis=0)
    maxi = np.nanmax(ras_arr, axis=0)
    mini = np.nanmin(ras_arr, axis=0)
    #vari = np.nanvar(modis_arr, axis=0)
    #percenti25 = np.nanpercentile(modis_arr,25, axis=0)
    #percenti50 = np.nanpercentile(modis_arr,50, axis=0)
    #percenti75 = np.nanpercentile(modis_arr,75, axis=0)

    compi_arr = np.stack((meani,
                          meadi,
                          stdi,
                          maxi,
                          mini,
                          #vari
                          # percenti25,
                          # percenti50,
                          # percenti75
                          ))
    return ras_file, compi_arr


def classify_func(raster, ras_file):
    """
    :param raster: compositing output as array
    :param ras_file: original file
    :return: array size and classification output
    """

    # get raster size (rows and columns)
    x_size = ras_file.RasterXSize
    y_size = ras_file.RasterYSize

    # reduce (reshape and reorder axis)
    land_rs = raster.reshape((raster.shape[0], -1))
    land_rs = land_rs.transpose()

    # standardize (z-transform)
    land_rs = StandardScaler().fit(land_rs.astype('float64')).transform(land_rs.astype('float64'))

    # fill nas with mean value
    imp = SimpleImputer(missing_values=np.nan, strategy='mean')  # default value for 'constant' is 0
    imp = imp.fit(land_rs)
    land_rs_imp = imp.transform(land_rs)

    # apply model to raster
    land_fit = model.predict(land_rs_imp)
    land_fit = land_fit.transpose()

    # replace the most common output with NA (it is assumed that the most common output is the result of the mean replacement)
    land_fit = land_fit.astype(np.int)
    count = np.bincount(land_fit)

    mode = np.argmax(count)

    mode = mode.astype(np.float)
    land_fit = land_fit.astype(np.float)

    land_fit[land_fit == mode] = np.nan

    # back-transform to original form
    land_fit = land_fit.reshape(y_size, x_size)

    return x_size, y_size, land_fit


def export_func(raster, raster_path, x_size, y_size, ras_file):
    """
    :param raster: classification output
    :param raster_path: entire input file path
    :param x_size: array size
    :param y_size: array size
    :param ras_file: original file
    :return:
    """
    name = Path(raster_path).stem
    drv = gdal.GetDriverByName('GTiff')
    dst_ds = drv.Create(path_data_folder + output + name + "_RF-results.tif", x_size, y_size, 1, gdal.GDT_Int32, [])
    dst_band = dst_ds.GetRasterBand(1)

    # write the data
    dst_band.WriteArray(raster, 0, 0)

    # flush data to disk, set the NoData value and calculate stats
    dst_band.FlushCache()

    dst_ds.SetGeoTransform(ras_file.GetGeoTransform())
    dst_ds.SetProjection(ras_file.GetProjectionRef())
    dst_ds = None


def map_func(raster_path):
    """
    :param raster_path: entire input file path
    :return: None
    """
    ras_file, comp = compositing_func(raster_path)
    x_size, y_size, raster = classify_func(comp, ras_file)
    export_func(raster, raster_path, x_size, y_size, ras_file)


# ####################################### FOLDER PATHS & global variables ############################################ #
# Folder containing the working data
path_data_folder = ''
landsat = 'Landsat_TC/CHACO_TreeCover2015.tif'
modis = 'MODIS_EVI/'
modis_tile = 'MODIS_EVI_h0_v2.tif'
inter = 'inter/'
output = 'output/'

# create folder to save intermediate results
inter_folder = (path_data_folder + 'inter')
if not os.path.exists(inter_folder):
    os.makedirs(inter_folder)

# create folder to save final results
output_folder = (path_data_folder + 'output')
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# set seed to make results replicable
np.random.seed(seed=8)

# ####################################### PROJECTION ################################################################# #
# No re-projection required
# ####################################### Data ####################################################################### #
# landsat data
land = gdal.Open(path_data_folder + landsat)

# all files in folder
tiles_list = glob.glob(path_data_folder + modis + "*.tif")
# ####################################### PROCESSING ################################################################# #
# Step 1: mosaicing ----------------------------------------------------------------------------------------------------
print("Step 1: mosaicing STARTED at:", time.strftime("%H:%M:%S", time.localtime()))
# https://gis.stackexchange.com/questions/44003/python-equivalent-of-gdalbuildvrt#44048
# set output path
output_mosaic = path_data_folder + '/inter/mosaic.tif'

# set options
vrt_options = gdal.BuildVRTOptions(resampleAlg='cubic', addAlpha=True)  # resampling Algorithm does not matter, as tiles do not overlap

# apply to files and export
mosaic = gdal.BuildVRT(output_mosaic, tiles_list, options=vrt_options)

# mosaic = gdal.Open(path_data_folder + inter +"/" + "mosaic.tif")     #shortcut after first successful run
modis_arr = mosaic.ReadAsArray()
print("Step 1: mosaicing DONE", time.strftime("%H:%M:%S", time.localtime()))
print()



# Step 2: resampling ---------------------------------------------------------------------------------------------------
print("Step 2: resampling STARTED at:", time.strftime("%H:%M:%S", time.localtime()))
# https://gis.stackexchange.com/questions/234022/resampling-a-raster-from-python-without-using-gdalwarp
# open landsat file and retrieve basic information
land = gdal.Open(path_data_folder + landsat)
inputProj = land.GetProjection()
inputTrans = land.GetGeoTransform()
referenceProj = mosaic.GetProjection()
referenceTrans = mosaic.GetGeoTransform()
bandreference = mosaic.GetRasterBand(1)
x = mosaic.RasterXSize
y = mosaic.RasterYSize

# set export information
resampled = (path_data_folder + 'inter/resampled.tif')
driver = gdal.GetDriverByName('GTiff')
resamp = driver.Create(resampled, x, y, 1, bandreference.DataType)
resamp.SetGeoTransform(referenceTrans)
resamp.SetProjection(referenceProj)

# apply resampler/reprojector
gdal.ReprojectImage(land, resamp, inputProj, referenceProj, gdalconst.GRA_Average)  # use mean for resampling approach
del resampled

# resamp = gdal.Open(path_data_folder + inter +"/" + "resampled.tif")     #shortcut after first successful run
resamp_arr = resamp.ReadAsArray()
print("Step 2: resampling DONE", time.strftime("%H:%M:%S", time.localtime()))
print()



# Step 3: reclassification of Landsat image ----------------------------------------------------------------------------
print("Step 3: reclassification of Landsat image STARTED at:", time.strftime("%H:%M:%S", time.localtime()))
# classify resampled landsat image in order to perform stratified sampling (5 classes)
reclass_arr = resamp_arr.copy()
reclass_arr[(reclass_arr >= 0) & (reclass_arr < 2000)] = 1
reclass_arr[(reclass_arr >= 2000) & (reclass_arr < 4000)] = 2
reclass_arr[(reclass_arr >= 4000) & (reclass_arr < 6000)] = 3
reclass_arr[(reclass_arr >= 6000) & (reclass_arr < 8000)] = 4
reclass_arr[(reclass_arr >= 8000) & (reclass_arr < 10000)] = 5

""" Optional export of results
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
print("Step 3: reclassification of Landsat image DONE", time.strftime("%H:%M:%S", time.localtime()))
print()



# Step 4: compositing --------------------------------------------------------------------------------------------------
print("Step 4: compositing STARTED at:", time.strftime("%H:%M:%S", time.localtime()))
# handle no data pixels
nodata = -9999
modis_arr = modis_arr[0:46, :, :]
modis_arr[(modis_arr == nodata)] = np.nan

# calculate spectral temporal indices
meani = np.nanmean(modis_arr, axis=0)
meadi = np.nanmedian(modis_arr, axis=0)
stdi = np.nanstd(modis_arr, axis=0)
maxi = np.nanmax(modis_arr, axis=0)
mini = np.nanmin(modis_arr, axis=0)
#vari = np.nanvar(modis_arr, axis=0)
#percenti25 = np.nanpercentile(modis_arr,25, axis=0)
#percenti50 = np.nanpercentile(modis_arr,25, axis=0)
#percenti75 = np.nanpercentile(modis_arr,25, axis=0)

# stack spectral temporal indices into array
compi_arr = np.stack((meani
                      , meadi
                      , stdi
                      , maxi
                      , mini
                      #, vari
                      #, percenti25
                      #, percenti50
                      #, percenti75
                      ))

""" Optional export of results
# create new file
driver= gdal.GetDriverByName('GTiff')
file2 = driver.Create(path_data_folder + 'inter/composite.tif', resamp.RasterXSize , resamp.RasterYSize , 5, gdal.GDT_Float64)
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
"""
# compi = gdal.Open(path_data_folder + inter + "/" + "composite.tif")  # shortcut after first successful run
# compi_arr = compi.ReadAsArray()
print("Step 4: compositing DONE", time.strftime("%H:%M:%S", time.localtime()))
print()



# Step 5: masking ------------------------------------------------------------------------------------------------------
print("Step 5: masking STARTED at:", time.strftime("%H:%M:%S", time.localtime()))
# based on entire mosaic
mask1 = np.logical_not(np.isnan(compi_arr[0, :, :]))

# apply mask on reclassified array
reclass_arr_m = np.where(mask1 == True, reclass_arr, np.nan)

# apply mask on original tree cover array
resamp_arr_m = np.where(mask1 == True, resamp_arr, np.nan)

print("Step 5: masking DONE:", time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime()))
print()



# Step 6: sampling stratified random sampling (Ass.09) -----------------------------------------------------------------
print("Step 6: sampling STARTED at:", time.strftime("%H:%M:%S", time.localtime()))
# create empty data frame to store spectral information
df = pd.DataFrame(columns={'TC': [],
                           'Band01': [],
                           'Band02': [],
                           'Band03': [],
                           'Band04': [],
                           'Band05': []
                           #'Band06': []
                           #'Band07': [],
                           #'Band08': [],
                           #'Band09': [],
                           })

# create empty data frame to store sample point information
sample_p = pd.DataFrame(columns={'x': [],
                                 'y': [],
                                 'class': []})

# perform stratified random sampling (100/class)
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

    # store spectral information in data frame
    df = df.append({'TC': tc,
                    'Band01': pixel_v[0],
                    'Band02': pixel_v[1],
                    'Band03': pixel_v[2],
                    'Band04': pixel_v[3],
                    'Band05': pixel_v[4]
                    #'Band06': pixel_v[5]
                    #'Band07': pixel_v[6],
                    #'Band08': pixel_v[7],
                    #'Band09': pixel_v[8],
                    },
                   ignore_index=True)

    # store sample point information in data frame
    sample_p = sample_p.append({'x': indsselected[:, 0],
                                'y': indsselected[:, 1],
                                'class': classes},
                               ignore_index=True)

# unnest/explode spectral information within data frame
df_samples = unnesting(df, ['Band01', 'Band02', 'Band03', 'Band04','Band05', 'TC'])

# unnest/explode sample point information within data frame
sample_points = unnesting(sample_p, ['x', 'y'])
print("Step 6: sampling DONE:", time.strftime("%H:%M:%S", time.localtime()))
print()



# Step 7: Classification -----------------------------------------------------------------------------------------------
print("Step 7: classification STARTED at:", time.strftime("%H:%M:%S", time.localtime()))
# Step7.1: data separation
y_df = df_samples.iloc[:,5]
x_df = df_samples.iloc[:, 0:5]
print("Step 7.1: data separation DONE", time.strftime("%H:%M:%S", time.localtime()))

# Step 7.2: z-transformation
x_df_scaled = StandardScaler().fit(x_df.astype('float64')).transform(x_df.astype('float64'))
print("Step 7.2: z-transformation DONE", time.strftime("%H:%M:%S", time.localtime()))

# Step 7.3: data split (training, validation
x_train, x_test, y_train, y_test = train_test_split(x_df_scaled, y_df, test_size=0.5, random_state=42)
print("Step 7.3: data split DONE", time.strftime("%H:%M:%S", time.localtime()))

# Step 7.4: hyper parameter selection (grid search)
est = GradientBoostingRegressor(n_estimators=100)
# grid search cross-validation to explore combinations of parameters
# https://www.slideshare.net/PyData/gradient-boosted-regression-trees-in-scikit-learn-gilles-louppe
param_grid = {'learning_rate': [0.1, 0.05, 0.02, 0.01],
              'max_depth': [4, 6],
              'min_samples_leaf': [1, 3, 9, 5, 17],
              'max_features': [1.0, 0.3, 0.1]}

grid = GridSearchCV(est, param_grid, cv=5, iid=False, n_jobs=3).fit(x_train, y_train)
grid.get_params()
print("Step 7.4: hyper parameter selection DONE", time.strftime("%H:%M:%S", time.localtime()))

# Step 7.5: fitting best model
model = grid.best_estimator_
y_fit = model.predict(x_test)
print("Step 7.5: fitting best model DONE", time.strftime("%H:%M:%S", time.localtime()))
print("Step 7: classification DONE", time.strftime("%H:%M:%S", time.localtime()))
print()



# Step 8: classification report ----------------------------------------------------------------------------------------
print("Step 8: classification report STARTED at:", time.strftime("%H:%M:%S", time.localtime()))
mse = mean_squared_error(y_test, y_fit)
print("Model Perfomance - mean squared error (MSE): %.4f" % mse)
print("Step 8: classification report DONE", time.strftime("%H:%M:%S", time.localtime()))
print()



# Step 9: apply to raster composites and export ------------------------------------------------------------------------
print("Step 9: apply to raster STARTED at:", time.strftime("%H:%M:%S", time.localtime()))
Parallel(n_jobs=3)(delayed(map_func)(i) for i in tiles_list)
print("Step 9: apply to raster DONE", time.strftime("%H:%M:%S", time.localtime()))
print()



# Step 10: plot classified map -----------------------------------------------------------------------------------------
print("Step 10: plotting NOT DONE", time.strftime("%H:%M:%S", time.localtime()))
print()



# Step 11: export sample points as Shape--------------------------------------------------------------------------------
print("Step 11: exporting sample points as .shp STARTED at:", time.strftime("%H:%M:%S", time.localtime()))
# get the envelope
ds = gdal.Open(path_data_folder + inter + 'mosaic.tif')
geoTransform = ds.GetGeoTransform()
xmin = geoTransform[0]
ymax = geoTransform[3]
xmax = xmin + geoTransform[1] * ds.RasterXSize
ymin = ymax + geoTransform[5] * ds.RasterYSize

# create empty shape file to store stuff in
# set up the shapefile driver
driver = ogr.GetDriverByName("ESRI Shapefile")

# create the data source
data_source = driver.CreateDataSource(path_data_folder + output + "/sample_points.shp")

# create the spatial reference
lyr = ds.GetLayer
proj = osr.SpatialReference(wkt=ds.GetProjection())
epsg = proj.GetAttrValue('AUTHORITY', 1)
srs = osr.SpatialReference()
srs.ImportFromWkt(ds.GetProjectionRef())

# create the layer
layer = data_source.CreateLayer("", srs, ogr.wkbPoint)

# Add the fields we're interested in
Class = ogr.FieldDefn("Class", ogr.OFTString)
layer.CreateField(Class)

Sample_ID = ogr.FieldDefn("Sample_ID", ogr.OFTInteger)
layer.CreateField(Sample_ID)
counter = 0

for index, row in sample_points.iterrows():
    counter += 1

    # create the feature
    feature = ogr.Feature(layer.GetLayerDefn())

    # set the attributes using the values from the delimited text file
    feature.SetField("Class", row['class'])
    feature.SetField("Sample_ID", counter)

    x_lan, y_lan = pixel2coord(row['y'], row['x'], ds)
    land_point = ogr.Geometry(ogr.wkbPoint)
    land_point.AddPoint(x_lan, y_lan)

    # set the feature geometry
    feature.SetGeometry(land_point)

    # create the feature in the layer
    layer.CreateFeature(feature)

print("Step 11: exporting sample points as .shp DONE", time.strftime("%H:%M:%S", time.localtime()))
print()
# ####################################### END TIME-COUNT AND PRINT TIME STATS ######################################## #
print("")
endtime = time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime())
print("--------------------------------------------------------")
print("start: " + starttime)
print("end: " + endtime)
print("")
