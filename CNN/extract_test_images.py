'''
subject_id: patient, one subject can have multiple studies
study_id: examination, one study can have multiple xray images
dicom_id: xray image, the file name of image
'''

import pandas as pd
import shutil
import os

# joint dataframes
df1 = pd.read_csv('cxr-record-list.csv')
df2 = pd.read_csv('mimic-cxr-2.0.0-chexpert.csv')
df3 = pd.read_csv('mimic-cxr-2.0.0-metadata.csv')

merge_df = df1.merge(df2,on=['subject_id','study_id'],how='inner')
merge_df = merge_df.merge(df3[['dicom_id', 'ViewPosition']],on='dicom_id',how='inner')
merge_df['jpg_id'] = merge_df['dicom_id'] + '.jpg'

test_images = os.listdir('Image_Testing') #all test images downloaded

#we select posterior to anterior view
for disease in ['Pleural Dffusion', 'Pneumonia', 'Pneumothorax', 'Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Enlarged Cardiomediastinum', #
       'Fracture', 'Lung Lesion', 'Lung Opacity', 'No Finding',  'Pleural Other',   'Support Devices']:
    df = merge_df[(merge_df['ViewPosition'] =='PA') & (merge_df[disease].isin([0,1]))]
    df = df[df['jpg_id'].isin(test_images)]
    
    if not os.path.exists(f'Testing/{disease}'):
        os.makedirs(f'Testing/{disease}') 
    for file in df['jpg_id']:
        file = 'Image_Testing/' + file
        shutil.copy(file, f'Testing/{disease}')

#some images are mislabeled as "PA" and we manually removed them
