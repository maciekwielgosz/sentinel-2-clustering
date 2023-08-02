import pandas as pd
import numpy as np
from PIL import Image
from sklearn.decomposition import TruncatedSVD

# read the dataframe from the csv file
df = pd.read_csv('hr.csv')

# define a function to compute SVD and return the first 16 components
def compute_svd(image_path):
    img = Image.open(image_path).convert('L')  # convert image to grayscale
    img = np.array(img)  # convert image data to numpy array
    svd = TruncatedSVD(n_components=16)
    result = svd.fit_transform(img)
    result = result.flatten()[:16]  # take the first 16 components
    return result

# create new columns in the dataframe for the SVD components
for i in range(16):
    df[f'svd_{i}'] = None

# compute the SVD for each image and store the components in the dataframe
for idx, row in df.iterrows():
    svd_result = compute_svd(row['image_path'])
    for i in range(16):
        df.loc[idx, f'svd_{i}'] = svd_result[i]

# check the dataframe
print(df.head())

# save the dataframe 
df.to_csv('hr_svd.csv', index=False)

