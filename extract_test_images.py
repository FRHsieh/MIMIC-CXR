'''
subject_id: patient, one subject can have multiple studies
study_id: examination, one study can have multiple xray images
dicom_id: xray image, the file name of image
'''

import pandas as pd
import shutil
import os

# joint dataframes
df2 = pd.read_csv('cxr-record-list.csv')
df4 = pd.read_csv('mimic-cxr-2.0.0-chexpert.csv')
df6 = pd.read_csv('mimic-cxr-2.0.0-metadata.csv')

merge_df = df2.merge(df4,on=['subject_id','study_id'],how='inner')
merge_df = merge_df.merge(df6[['dicom_id', 'ViewPosition']],on='dicom_id',how='inner')
merge_df['jpg_id'] = merge_df['dicom_id'] + '.jpg'

#we select posterior to anterior view
df_pneu_pa = merge_df[(merge_df['ViewPosition'] =='PA') & (merge_df['Pneumonia'].isin([0,1]))]
df_pneu_pa = df_pneu_pa[df_pneu_pa['jpg_id'].isin(test_images)]
df_PE_pa = merge_df[(merge_df['ViewPosition'] =='PA') & (merge_df['Pleural Effusion'].isin([0,1]))]
df_PE_pa = df_PE_pa[df_PE_pa['jpg_id'].isin(test_images)]
df_Pneumothorax_pa = merge_df[(merge_df['ViewPosition'] =='PA') & (merge_df['Pneumothorax'].isin([0,1]))]
df_Pneumothorax_pa = df_Pneumothorax_pa[df_Pneumothorax_pa['jpg_id'].isin(test_images)]

test_images = os.listdir('Image_Testing_2') #all test images downloaded

for file in df_pneu_pa['jpg_id']:
    file = 'Image_Testing_2/' + file
    shutil.copy(file, 'Testing/Pneumonia')

for file in df_PE_pa['jpg_id']:
    file = 'Image_Testing_2/' + file
    shutil.copy(file, 'Testing/Pleural Effusion')

for file in df_Pneumothorax_pa['jpg_id']:
    file = 'Image_Testing_2/' + file
    shutil.copy(file, 'Testing/Pneumothorax')

#some images are mislabeled as "PA" and we manually removed them
