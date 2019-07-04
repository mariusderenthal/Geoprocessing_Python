# ############################################################################################################# #
# Raw-example for a basic script to be executed in python. The format is only a recommendation and everyone is  #
# encouraged to modify the scripts to her/his own needs.                                                        #
# (c) Matthias Baumann, Humboldt-Universität zu Berlin, 4/15/2019
#
# Assignment 03: Marius Derenthal
# ####################################### LOAD REQUIRED LIBRARIES ############################################# #
import time
import os
import re
from osgeo import gdal
import numpy as np

# ####################################### SET TIME-COUNT ###################################################### #
starttime = time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime())
print("--------------------------------------------------------")
print("Starting process, time: " + starttime)
print("")
# ####################################### FOLDER PATHS & global variables ##################################### #
#os.chdir('.')                       # go up one level
home = '/Users/mariusderenthal/Google Drive/Global Change Geography/4.Semester/Geoprocessing_Python'  # home folder
folder = "/Data/"  # folger containing files

files = os.listdir(home+folder)
files = [f for f in os.listdir(home+folder) if re.match(".*\\.[tT][iI][fF]{1,2}$", f)]
files = [files[1],files[3],files[4],files[2],files[0]] #Order from 2000 - 2018

# ####################################### FUNCTIONS ########################################################### #
def findintersection(raster1, raster2):
    inter = [max(raster1[0], raster2[0]), min(raster1[1], raster2[1]), min(raster1[2], raster2[2]), max(raster1[3], raster2[3])]
    return inter
# ####################################### PROCESSING 1 ########################################################## #
# extract and list raster, bands, geotrans info and bounding box information
rasters = []
bands = []
geos = []
bbs = []
Å
for f in files:
    # only .tif files
    if re.match(".*\\.[tT][iI][fF]{1,2}$", f):
        # open raster
        raster = gdal.Open(home + folder + f)

        band = raster.GetRasterBand(1)
        gt = raster.GetGeoTransform()
        bb = [gt[0], gt[3], gt[0] + (gt[1] * raster.RasterXSize), gt[3] + (gt[5] * raster.RasterYSize)]

        rasters.append(raster)
        bands.append(band)
        geos.append(gt)
        bbs.append(bb)

i = 0

for b in range(len(bbs)-1):
    if i == 0:
        ibox = findintersection(bbs[b],bbs[b+1])
    else:
        ibox = findintersection(ibox, bbs[b + 1])
    i += 1

lefts = []
tops = []
cols = []
rows = []

for t in range(len(bbs)):
    left = int(round((ibox[0] - bbs[t][0]) / geos[t][1]))
    top = int(round((ibox[1] - bbs[t][1]) / geos[t][5]))
    col = int(round((ibox[2] - bbs[t][0]) / geos[t][1])) - left
    row = int(round((ibox[3] - bbs[t][1]) / geos[t][5])) - top

    lefts.append(left)
    tops.append(top)
    cols.append(col)
    rows.append(row)


arrays = []

for i in range(len(bands)):
    array = bands[i].ReadAsArray(lefts[i], tops[i], cols[i], rows[i])
    arrays.append(array)

#Results for 2000.tif
print(np.max(arrays[1]))
print(np.mean(arrays[1]))
print(np.min(arrays[1]))
print(np.std(arrays[1]))

# ####################################### END TIME-COUNT AND PRINT TIME STATS################################## #
print("")
endtime = time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime())
print("--------------------------------------------------------")
print("start: " + starttime)
print("end: " + endtime)
print("")