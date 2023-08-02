import os
import glob
import pandas as pd
from PIL import Image

def get_image_paths_and_labels(root_folder):
    image_paths = []
    labels = []
    
    # iterate over all direct child directories of the root folder
    for dirpath, dirnames, filenames in os.walk(root_folder):
        # only process directories that contain files (i.e., skip the root)
        if filenames:
            # for every file in directory
            for f in filenames:
                # if it is a TIF file
                if f.endswith('.tif'):
                    # record its full path and the name of its parent directory
                    full_path = os.path.join(dirpath, f)
                    label = os.path.basename(dirpath)
                    image_paths.append(full_path)
                    labels.append(label)
    return image_paths, labels

root_folder = "HR"  # replace with your actual folder

# get the paths and labels
image_paths, labels = get_image_paths_and_labels(root_folder)

# create a dataframe
df = pd.DataFrame({
    'image_path': image_paths,
    'label': labels
})

# check the dataframe
print(df.head())

# save the dataframe to a csv file
df.to_csv('hr.csv', index=False)

