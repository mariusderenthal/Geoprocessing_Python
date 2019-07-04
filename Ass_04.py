# ############################################################################################################# #
# Raw-example for a basic script to be executed in python. The format is only a recommendation and everyone is  #
# encouraged to modify the scripts to her/his own needs.                                                        #
# (c) Matthias Baumann, Humboldt-Universität zu Berlin, 4/15/2019
#
# Assignment 04: Marius Derenthal
# ####################################### LOAD REQUIRED LIBRARIES ############################################# #
import time
import gdal
import os
import numpy as np
import re
# ####################################### SET TIME-COUNT ###################################################### #
starttime = time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime())
print("--------------------------------------------------------")
print("Starting process, time: " + starttime)
print("")
# ####################################### FUNCTIONS ########################################################### #
#01 Mask function
def mask_bool(qa_path):
    #load qa file
    qa_file = gdal.Open(qa_path)
    #read qa file
    qa_array = qa_file.GetRasterBand(1).ReadAsArray()
    #set conditions
    mask = np.logical_or(qa_array == 0, qa_array == 1, qa_array == 3)
    return mask

#02 Masking function
def masking(sr_path):
    # find the according qa band
    qa_path = sr_path.replace('sr', 'qa')
    # create mask
    qa_mask = mask_bool(qa_path)
    # load image
    sr_file = gdal.Open(sr_path)
    # create empty 3d array to fill with bands
    image_stack = np.empty([sr_file.RasterCount, sr_file.RasterYSize, sr_file.RasterXSize])
    # iterate through number of bands
    for b in range(1, sr_file.RasterCount):
        ras = sr_file.GetRasterBand(b).ReadAsArray()
        ras_masked = np.where(qa_mask == False, np.nan, ras)
        image_stack[b, :, :] = ras_masked

    return image_stack

#03 Mean Image function
def mean_image(list_files):
    # create empty 4d array to fill with bands
    ex = gdal.Open(list_files[0])
    all_images = np.empty([len(list_files), ex.RasterCount, ex.RasterYSize, ex.RasterXSize])

    # fill 4d array
    for f, i in zip(list_files, range(len(list_files))):
        all_images[i, :, :, :] = masking(f)

    # extract mean along each image
    mean_bands = np.nanmean(all_images, axis=0)

    return mean_bands

#04 Max ndvi composite function
def max_ndvi_image(list_files):
    # create empty 4d array to fill with bands
    ex = gdal.Open(list_files[0])
    all_images = np.empty([len(list_files), ex.RasterCount, ex.RasterYSize, ex.RasterXSize])

    # fill 4d array
    for f, i in zip(list_files, range(len(list_files))):
        all_images[i, :, :, :] = masking(f)

    # select bands need for NDVI
    vis = all_images[:, 2, :, :]
    nir = all_images[:, 3, :, :]

    # calculate NDVI
    ndvi = np.divide((nir - vis), (nir + vis))

    # taken from Max Wesemeyer
    ndvi[np.isnan(ndvi)] = 0
    NDVI_index = np.argmax(ndvi, axis=0)
    NDVI_index = np.expand_dims(NDVI_index, axis=0)
    NDVI_index = np.reshape(NDVI_index, newshape=(1, 1, 1000, 1000))
    NDVI_max = np.take_along_axis(all_images, NDVI_index, axis=0)
    NDVI_max = NDVI_max.squeeze()

    return NDVI_max

# Create GeoTIFF function (based on code taken from the Geoprocessing-with-Python_Chapter11.pdf)
def make_raster(in_ds, fn, data, data_type, nodata=None):
    driver = gdal.GetDriverByName('GTiff')
    out_ds = driver.Create(fn, in_ds.RasterXSize, in_ds.RasterYSize, 6, data_type, ['COMPRESS=DEFLATE','PREDICTOR=2', 'TILED=YES'])
    out_ds.SetProjection(in_ds.GetProjection())
    out_ds.SetGeoTransform(in_ds.GetGeoTransform())

    for i in range(len(data)):
        out_ds.GetRasterBand(i+1).WriteArray(data[i])

    out_ds.FlushCache()
    out_ds = None

    return out_ds

# ####################################### FOLDER PATHS & global variables ##################################### #
home = '/Users/mariusderenthal/Google Drive/Global Change Geography/4.Semester/Geoprocessing_Python/'  # home folder
data = 'Data/'
folder = "landsat8_2015/"  # folder containing files
# ####################################### PROCESSING ########################################################## #
# List of all images
images = [f for f in os.listdir(home+data+folder) if re.findall("LC", f)]

sr_files = list()
qa_files = list()

# Identify for each image its path and save it in a list
for i in images:
    files = [f for f in os.listdir(home+data+folder+i) if re.findall("LC", f)]

    for f in files:
        if re.findall("sr", f):
            sr_files.append(home+data+folder + i + "/" + f)
        if re.findall("qa", f):
            qa_files.append(home+data+folder + i + "/" + f)

# Apply functions
mean_all = mean_image(sr_files)
maxNDVI = max_ndvi_image(sr_files)

# Export results as GeoTIFF
make_raster(in_ds=gdal.Open(sr_files[0]),
            fn='mean_all.tiff',
            data = mean_all,
            data_type=gdal.GDT_Int16,
            nodata=None)

make_raster(in_ds=gdal.Open(sr_files[0]),
            fn='ndvi_max.tiff',
            data = maxNDVI,
            data_type=gdal.GDT_Int16,
            nodata=None)

'''
# Check output
mean_test = gdal.Open("/Users/mariusderenthal/Google Drive/Global Change Geography/4.Semester/Geoprocessing_Python/coding/mean_all.tiff")
mean_test_a = mean_test.ReadAsArray()
mean_test_a.shape

ndvi_test = gdal.Open("/Users/mariusderenthal/Google Drive/Global Change Geography/4.Semester/Geoprocessing_Python/coding/ndvi_max.tiff")
ndvi_test_a = ndvi_test.ReadAsArray()
ndvi_test_a.shape
'''
# ####################################### END TIME-COUNT AND PRINT TIME STATS##################
print("")
endtime = time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime())
print("--------------------------------------------------------")
print("start: " + starttime)
print("end: " + endtime)
print("")