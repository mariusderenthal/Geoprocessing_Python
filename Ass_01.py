import os
import re


path = "/Users/mariusderenthal/Desktop/Assignment01_data/Part01_Landsat"



#######################################PART 1.1
# Define string patterns for each satellite
pattern08 = '^LC08'
pattern07 = '^LE07'
pattern05 = '^LT05'
pattern04 = '^LT04'

for foot in os.listdir(path):
    footy = os.path.join(path, foot)

    if os.path.isdir(footy):

        # Create empty count variables
        count01 = 0
        count08 = 0
        count07 = 0
        count05 = 0
        count04 = 0

        for scenes in os.listdir(footy):
            if re.match(pattern08, scenes):
                count08 += 1
            if re.match(pattern07, scenes):
                count07 += 1
            if re.match(pattern05, scenes):
                count05 += 1
            if re.match(pattern04, scenes):
                count04 += 1
        print("For the",foot,"footpring we count ", count08, "Landsat 8 scenes")
        print("For the",foot,"footpring we count ", count07, "Landsat 7 scenes")
        print("For the",foot,"footpring we count ", count05, "Landsat 5 scenes")
        print("For the",foot,"footpring we count ", count04, "Landsat 4 scenes","\n")


#######################################PART 1.2
# Define average number of files for each Satellite
num08 = 19
num07 = 19
num05 = 21
num04 = 21

# Create empty count variables
missings = 0

f = open("Corrupt_Scenes.txt", "w+")

for foot in os.listdir(path):
    footy = os.path.join(path, foot)

    if os.path.isdir(footy):

        # Create empty count variables
        count01 = 0
        count08 = 0
        count07 = 0
        count05 = 0
        count04 = 0

        for scenes in os.listdir(footy):
            scenesy = os.path.join(footy, scenes)

            if re.match(pattern08, scenes):
                list = os.listdir(scenesy)
                number_files = len(list)
                if num08 != number_files:
                    missings += 1
                    a = os.path.join(path, scenes)
                    f.write(scenes + os.linesep)

            if re.match(pattern07, scenes):
                list = os.listdir(scenesy)
                number_files = len(list)
                if num07 != number_files:
                    missings += 1
                    a = os.path.join(path, scenes)
                    f.write(scenes + os.linesep)

            if re.match(pattern05, scenes):
                list = os.listdir(scenesy)
                number_files = len(list)
                if num05 != number_files:
                    missings += 1
                    a = os.path.join(path, scenes)
                    f.write(scenes + os.linesep)

            if re.match(pattern04, scenes):
                list = os.listdir(scenesy)
                number_files = len(list)
                if num04 != number_files:
                    missings += 1
                    a = os.path.join(path, scenes)
                    f.write(scenes + os.linesep)

f.close()
print("In total",missings,"scenes are incomplete")



#######################################PART 2.1
path = "/Users/mariusderenthal/Desktop/Assignment01_data/Part02_GIS-Files"

ras_len = len([file for file in os.listdir(path)
               if file.endswith('.tif') and os.path.isfile(os.path.join(path, file))])

shp_len = len([file for file in os.listdir(path)
               if file.endswith('.shp') and os.path.isfile(os.path.join(path, file))])

print("The folder contains:", ras_len, "raster layers")
print("The folder contains:", shp_len, "vector layers")


#######################################PART 2.2
list_dir = os.listdir(path)
incom_vec = []
#incom_ras = []

f = open("Corrupt_Shapes.txt", "w+")

for name in list_dir:

    if name.endswith('.shp'):
        dbf_files = re.sub('.shp', ".dbf", name)
        prj_files = re.sub('.shp', ".prj", name)

        if dbf_files not in list_dir:
            incom_vec.append(dbf_files)
            f.write(dbf_files  + os.linesep)

        if prj_files not in list_dir:
            incom_vec.append(prj_files)
            f.write(prj_files + os.linesep)

    #if name.endswith('.tif'):
     #   dbf_files = re.sub('.tif', ".dbf", name)

     #   if dbf_files not in list_dir:
      #      incom_ras.append(dbf_files)

f.close()


print(len(incom_vec), "shapefiles are not complete")

