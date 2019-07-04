# ############################################################################################################# #
# (c) Matthias Baumann, Humboldt-Universität zu Berlin, 4/15/2019                                               #
# Assignment VI – Protected area summary                                                                          #
# Group: Marius Derenthal, Lara Schmitt, Max Wesenmeyer, Lukas Blickensdoerfer                                                                                         #
# ####################################### LOAD REQUIRED LIBRARIES ############################################# #

import time
from osgeo import ogr
import numpy as np
import pandas as pd

# ####################################### SET TIME-COUNT ###################################################### #
starttime = time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime())
print("--------------------------------------------------------")
print("Starting process, time: " + starttime)
print("")
# ####################################### FUNCTIONS ########################################################### #



# ####################################### FOLDER PATHS & global variables ##################################### #

# Folder containing the working data
path_data_folder = '/Users/mariusderenthal/Google Drive/Global Change Geography/4.Semester/Geoprocessing_Python/Data'
prot_areas = "WDPA_May2019-shapefile-polygons.shp"
country_shp = "gadm36_dissolve.shp"
# ####################################### PROCESSING ########################################################## #

prot = ogr.Open(path_data_folder + "/" + prot_areas, 1)
countries = ogr.Open(path_data_folder + "/" + country_shp, 0)

# if you need to exclude marine PA´s
'''
if prot is None:
    sys.exit('Could not open {0}.'.format(prot_areas))
layer = prot.GetLayer(0)

for feat in layer:
    # print(feat.GetField('MARINE'))
    if feat.GetField('MARINE') == '2':
        print("delete")
        layer.DeleteFeature(feat.GetFID())
prot.SyncToDisk()
# loop over country and set spatial filter
# calculate needed values
'''

# define final dataframe for collecting results and later export
df = pd.DataFrame(columns ={'Country ID':[],
                            'Country Name':[],
                            'PA Category':[],
                            '# PAs':[],
                            'Mean area of PAs':[],
                            'Area of largest PA':[],
                            'Name of largest PA':[],
                            'Year of establ. Of largest PA':[]
                            })

i = 0
pro = prot.GetLayer()

# do everything for each country
for c in countries.GetLayer(0):
    country_name = c.GetField('NAME_0')
    print(country_name)
    # get the PA´s for the current county
    pro.SetSpatialFilter(c.geometry())
    feat_count = pro.GetFeatureCount()

    print("done setting spatial filter: " + time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime()))

    # if no protected areas are found
    if feat_count == 0:
        df = df.append({'Country ID': i,
                        'Country Name': country_name,
                        'PA Category': "all",
                        '# PAs': 0,
                        'Mean area of PAs': np.nan,
                        'Area of largest PA': np.nan,
                        'Name of largest PA': np.nan,
                        'Year of establ. Of largest PA': np.nan}, ignore_index=True)
        pro.SetSpatialFilter(None)
        print("done getting features: " + time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime()))
        print("no protected areas found")
        i += 1
    else:
        # create np.arrays to collect and summarize PA-information on country level
        PA_DEF = np.empty(feat_count, dtype='U25')
        area = np.empty(feat_count)
        names = np.empty(feat_count, dtype='U25')
        years = np.empty(feat_count, int)
        j = 0
        # collect info
        for feat in pro:
            PA_DEF[j] = feat.GetField('IUCN_CAT')
            area[j] = feat.GetField('REP_AREA')
            names[j] = feat.GetField('NAME')
            years[j] = feat.GetField('STATUS_YR')
            j += 1
        pro.SetSpatialFilter(None)
        print("done getting features: " + time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime()))
        # calculate summary and save for "all" PA´s in current county
        df = df.append({'Country ID': i,
                        'Country Name': country_name,
                        'PA Category': "all",
                        '# PAs': feat_count,
                        'Mean area of PAs': np.mean(area),
                        'Area of largest PA': max(area),
                        'Name of largest PA': names[np.argmax(area)],
                        'Year of establ. Of largest PA': years[np.argmax(area)]}, ignore_index=True)

        # do the same for each PA category in the country
        for type in np.unique(PA_DEF):
            bool_sub = PA_DEF == type
            area_tmp = area[bool_sub]

            df = df.append({'Country ID': i,
                            'Country Name': country_name,
                            'PA Category': type,
                            '# PAs': len(area_tmp),
                            'Mean area of PAs': np.mean(area_tmp),
                            'Area of largest PA': max(area_tmp),
                            'Name of largest PA': names[bool_sub][np.argmax(area_tmp)],
                            'Year of establ. Of largest PA':  years[bool_sub][np.argmax(area_tmp)]}, ignore_index=True)

        print("done calculating metrics: " + time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime()))
        print("")
        i += 1

    # for testing set number of loops
    #if i == 5:
    #    break

# save to file
df.to_csv(path_data_folder + "/" + "df_output_sig.csv", index=False, float_format='%.2f', encoding='utf-8-sig')



# ####################################### END TIME-COUNT AND PRINT TIME STATS##################
print("")
endtime = time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime())
print("--------------------------------------------------------")
print("start: " + starttime)
print("end: " + endtime)
print("")
