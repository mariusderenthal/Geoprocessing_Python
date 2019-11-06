# ####################################### INFORMATION ################################################################ #
# The basic script structure was taken from Matthias Baumann, Humboldt-Universität zu Berlin and has been
# modified to specific requirements.
# Autor: Marius Derenthal

# ####################################### TASK ####################################################################### #
# Create greatest disturbance image:
# Create a single layer raster where each pixel has the year of the greatest disturbance
# (largest positive change magnitude) if it was disturbed or 0 if it was undisturbed or non-forest.
# Undisturbed are all pixels with a change magnitude less than 500.

# Create a histogram that shows the annual forest disturbance frequency between 1987 and 2018 for the study area.
# ####################################### WORKFLOW ################################################################### #
"""
1. create subsets of years and fitted values ---------------------------------------------------------------------------
2. create 3d difference array ------------------------------------------------------------------------------------------
3. extract maximum values ----------------------------------------------------------------------------------------------
4. apply minimum disturbance conditions and the forest mask ------------------------------------------------------------
5. plot map ------------------------------------------------------------------------------------------------------------
6. plot histogram ------------------------------------------------------------------------------------------------------
7. export results as GeoTIFF -------------------------------------------------------------------------------------------
"""
# ####################################### LOAD REQUIRED LIBRARIES #################################################### #
import time
import os
import gdal
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import patches as mpatches
from matplotlib import colors

# ####################################### SET TIME-COUNT ############################################################# #
starttime = time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime())
print("--------------------------------------------------------")
print("Starting process, time: " + starttime)
print("")
# ####################################### FUNCTIONS ################################################################## #
def make_raster(in_ds, fn, data, data_type, nr_bands, nodata=None):
    driver = gdal.GetDriverByName('GTiff')
    out_ds = driver.Create(fn, in_ds.RasterXSize, in_ds.RasterYSize, nr_bands, data_type,
                           ['COMPRESS=DEFLATE',
                            'PREDICTOR=2',
                            'TILED=YES'])

    out_ds.SetProjection(in_ds.GetProjection())
    out_ds.SetGeoTransform(in_ds.GetGeoTransform())


    out_ds.GetRasterBand(1).WriteArray(data[0])

    out_ds.FlushCache()
    out_ds = None

    return out_ds


def plotDisturbanceYearMap(img):

    values = np.unique(img.ravel())
    values = values[values > 0]

    cmap = plt.get_cmap('jet', values.size)
    cmap.set_under('grey')
    norm = colors.BoundaryNorm(list(values), cmap.N)

    plt.figure(figsize=(8, 4))
    im = plt.imshow(img, interpolation='none', cmap=cmap, norm=norm)

    dist_colors = [im.cmap(im.norm(value)) for value in values]
    # create a patch (proxy artist) for every color
    patches = [mpatches.Patch(color=dist_colors[i], label=values[i]) for i in range(len(values))]
    # put those patched as legend-handles into the legend
    plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2,
               borderaxespad=0., ncol=2, frameon=False)

    plt.grid(True)
    plt.show()

# ####################################### FOLDER PATHS & global variables ############################################ #
home = ''  # home folder
data = 'Data/'
#folder = "landsat8_2015/"  # folder containing files
inter = 'inter/'
output = 'output/'

# create folder to save intermediate results
inter_folder = (home + data + 'inter')
if not os.path.exists(inter_folder):
    os.makedirs(inter_folder)

# create folder to save final results
output_folder = (home + data + 'output')
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# ####################################### Data ####################################################################### #
# Step 0. load data
# read forest mask
for_ds = gdal.Open(home + data + 'gpy_poland_forestmask.tif')
formask = for_ds.ReadAsArray()
#for_ds = None
#formask.shape

# read vertex file
src_ds = gdal.Open(home + data + 'gpy_poland_landtrendr_tcw_vertex_8618.tif')
img = src_ds.ReadAsArray()
#src_ds = None
#img.shape

# ####################################### PROCESSING ################################################################# #
'''
# exploratory stuff
p1 = img[:, 370, 291]

vertex_year = img[0:6, 370, 291]
vertex_raw = img[7:13, 370, 291]
vertex_fit = img[14:20, 370, 291]
plt.plot(vertex_year, vertex_fit)
plt.plot(vertex_year, vertex_raw)

np.unique(img[0:6,:,:])
vertex_raw_all = img[7:13, :, :]

# try some shit out
test = np.subtract(img[0, :, :], img[1, :, :])
np.max(min)
'''
# Step 1. create subsets of years and fitted values --------------------------------------------------------------------
# create subset for years
years = img[0:7,:,:].copy()
years.shape
#one = years[5,:,:]
np.amax(years)
np.amin(years)

# create subset for fitted values
fitted_stack = img[14:21,:,:].copy()
fitted_stack.shape

''' First attempt: Apply mask first, then continue (did not work out well)
ex = gdal.Open(home + data + 'gpy_poland_landtrendr_tcw_vertex_8618.tif')
image_stack = np.empty([7, ex.RasterYSize, ex.RasterXSize])
image_stack.shape

# iterate through number of bands (fitted values)
for b, i in zip(range(15, 22), range(8)):
    #print(b)
    ras = src_ds.GetRasterBand(b).ReadAsArray()
    #ras_masked = np.where(formask == 0, np.nan, ras)
    #ras_masked = np.where(formask == 0, 0, ras)
    #image_stack[i, :, :] = ras_masked
    image_stack[i, :, :] = ras

image_stack.shape
'''

# Step 2. create 3d difference array -----------------------------------------------------------------------------------
# create empty 3d array to fill with difference values
ex = gdal.Open(home + data + 'gpy_poland_landtrendr_tcw_vertex_8618.tif')
all_diff = np.empty([6 , ex.RasterYSize, ex.RasterXSize])

# fill 3d difference array
for i in range(6):
    all_diff[i,:,:] = np.subtract(fitted_stack[i+1, :, :], fitted_stack[i, :, :])

# Step 3. extract maximum values ---------------------------------------------------------------------------------------
# extract max values and their index
max_value = np.amax(all_diff, axis=0)
max_index = np.argmax(all_diff, axis=0)

# modify index array in order to make it compatible with years array
max_index = np.reshape(max_index, newshape=(1, ex.RasterYSize, ex.RasterXSize))

# extract values along axis
dist_year = np.take_along_axis(img[0:7,:,:], max_index+1, axis=0)
dist_year = dist_year.squeeze()

# Step 4. apply minimum disturbance conditions and the forest mask -----------------------------------------------------
dist_year[max_value < 500] = 0
dist_year = np.where(formask == 0, np.nan, dist_year)

# Step 5. plot map -----------------------------------------------------------------------------------------------------
plotDisturbanceYearMap(dist_year)

# Step 6. plot histogram -----------------------------------------------------------------------------------------------
dist_year_hist = dist_year.flatten()
dist_year_hist = np.delete(dist_year_hist, np.where(dist_year_hist == 0))

plt.grid(axis='y', alpha=0.75)
plt.xlabel('Year')
plt.ylabel('Count')
plt.title('Disturbance Histogram')
plt.hist(dist_year_hist , bins= 32, range=(1987,2018))
plt.show()

# Step 7. export results as GeoTIFF ------------------------------------------------------------------------------------
test_result = np.expand_dims(dist_year, axis=0) #create a band

make_raster(in_ds=gdal.Open(home + data + output + 'gpy_poland_landtrendr_tcw_vertex_8618.tif'),
            fn='tiff_disturbance.tiff',
            data=dist_year,
            data_type=gdal.GDT_Int16,
            nr_bands=1,
            nodata=None)

# ####################################### END TIME-COUNT AND PRINT TIME STATS ######################################## #
print("")
endtime = time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime())
print("--------------------------------------------------------")
print("start: " + starttime)
print("end: " + endtime)
print("")
