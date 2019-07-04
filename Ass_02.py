# ############################################################################################################# #
# Raw-example for a basic script to be executed in python. The format is only a recommendation and everyone is  #
# encouraged to modify the scripts to her/his own needs.                                                        #
# (c) Matthias Baumann, Humboldt-Universität zu Berlin, 4/15/2019                                               #
# ####################################### LOAD REQUIRED LIBRARIES ############################################# #

import pandas as pd
from geopandas import GeoDataFrame
from shapely.geometry import Point
from shapely.geometry import MultiPoint
import time

# ####################################### SET TIME-COUNT ###################################################### #
starttime = time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime())
print("--------------------------------------------------------")
print("Starting process, time: " + starttime)
print("")
# ####################################### FOLDER PATHS & global variables ##################################### #
path = "/Users/mariusderenthal/Google Drive/Global Change Geography/4.Semester/Geoprocessing_Python/"
land_path = '/Users/mariusderenthal/Google Drive/Global Change Geography/4.Semester/Geoprocessing_Python/Data/LANDSAT_8_C1_313804.csv'

# ####################################### FUNCTIONS ########################################################### #

# ####################################### PROCESSING 1 ########################################################## #
# load data
ds_file = '/Users/mariusderenthal/Google Drive/Global Change Geography/4.Semester/Geoprocessing_Python/Data/DE_2015_20180724.csv'
grid_file = '/Users/mariusderenthal/Google Drive/Global Change Geography/4.Semester/Geoprocessing_Python/Data/GRID_CSVEXP_20171113.csv'

ds = pd.read_csv(ds_file)
grid = pd.read_csv(grid_file)

# filter data
ds_fil = ds[(ds['OBS_TYPE'] == 1) &
            (ds['OBS_DIR'] == 1) &
            (ds['OBS_RADIUS'] <= 2) &
            (ds['AREA_SIZE'] >= 2) &
            (ds['FEATURE_WIDTH'] > 1) &
            (ds['LC1_PCT'] >= 5)]

# 1) How many observations are left?
print('1.How many observations are there?', len(ds))
print('1.How many observations are left?', len(ds_fil))

# 2) How many land cover classes (LC1) were recorded?
print('2.How many land cover classes (LC1) were recorded?', len(ds_fil.LC1.unique()))

# 3) What is the average observer distance (meters)?
print('3.What is the average observer distance (meters)?', ds_fil.OBS_DIST.mean())

# 4) What is the average observer distance of class A21?
means = ds_fil.groupby('LC1').mean()
print('4.What is the average observer distance of class A21?', means.loc['A21', 'OBS_DIST'])

# 5) How many samples are in the land cover class (LC1) with the most samples?
sizes = ds_fil.groupby(['LC1']).size()
print('5.How many samples are in the land cover class (LC1) with the most samples?', 'Class:', sizes.idxmax(), ',',
      'Samples:', sizes.max())

# create shapefile
df = pd.merge(grid, ds_fil)

geometry = [Point(xy) for xy in zip(df.GPS_LONG, df.GPS_LAT)]

crs = "+init=epsg:4326"

df = GeoDataFrame(df, crs=crs, geometry=geometry)
df = df.drop(columns=['ASSOC18'])
df.to_file(driver='ESRI Shapefile', filename='lucas_de.shp')

# ####################################### PROCESSING 2 ########################################################## #
land = pd.read_csv(land_path)

# filter data
land_fil = land[(land['Day/Night Indicator'] == 'DAY') &
                (land['Data Type Level-1'] == 'OLI_TIRS_L1TP') &
                (land['Land Cloud Cover'] < 70)]

print('0.How many observations are there?', len(land))
print('0.How many observations are left?', len(land_fil))

land_fil.columns = [c.replace(' ', '_') for c in land_fil.columns]

# 1) What is the average geometric accuracy in the X and Y direction: 'Geometric RMSE Model X' and 'Geometric RMSE Model Y'
print(
    '1.What is the average geometric accuracy in the X and Y direction: Geometric RMSE Model X and Geometric RMSE Model Y?',
    land_fil['Geometric_RMSE_Model_X'].mean(),
    'for x and',
    land_fil['Geometric_RMSE_Model_Y'].mean(),
    'for y')

# 2) What is the average land cloud cover
print('2.What is the average land cloud cover', land_fil['Land_Cloud_Cover'].mean())

# 3) How many Landsat footprints (unique Path/Row combination) are there?
subset = land_fil[['WRS_Path', 'WRS_Row']]
tuples = [tuple(x) for x in subset.values]
len(set(tuples))
print('3.How many Landsat footprints (unique Path/Row combination) are there?', len(set(tuples)))

UL = [Point(xy) for xy in zip(land_fil.UL_Corner_Lat_dec, land_fil.UL_Corner_Long_dec)]
UR = [Point(xy) for xy in zip(land_fil.UR_Corner_Lat_dec, land_fil.UR_Corner_Long_dec)]
LL = [Point(xy) for xy in zip(land_fil.LL_Corner_Lat_dec, land_fil.LL_Corner_Long_dec)]
LR = [Point(xy) for xy in zip(land_fil.LR_Corner_Lat_dec, land_fil.LR_Corner_Long_dec)]

data_tuples = list(zip(UL, UR, LL, LR))
data_df = pd.DataFrame(data_tuples, columns=['UL', 'UR', 'LL', 'LR'])

geometry = []
for i in range(0, len(data_df)):
    mu = MultiPoint(data_df.iloc[i])
    po = mu.convex_hull
    geometry.append(po)

crs = "+init=epsg:4326"

df = GeoDataFrame(land_fil, crs=crs, geometry=geometry)
df.to_file(driver='ESRI Shapefile', filename='landsat_de.shp')

# ####################################### END TIME-COUNT AND PRINT TIME STATS################################## #
print("")
endtime = time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime())
print("--------------------------------------------------------")
print("start: " + starttime)
print("end: " + endtime)
print("")
