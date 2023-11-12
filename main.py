import os
import pandas as pd
import numpy as np
import shutil

df = pd.read_csv('cv-other-train.csv')

pwd = os.getcwd()
print(pwd)
directory = os.path.join(pwd, 'cv-other-train')
files = os.listdir(directory)

# print("valid train before: ", len(os.listdir('cv-other-train')))
#
# # print(len(os.listdir(os.path.join(pwd,'dataset/training/male'))))
# count = 0
# for index, row in df.iterrows():
#     count+=1
#     print(count)
#     if pd.isnull(row['gender']) or pd.isnull(row['accent']):
#         filename = row['filename'].split('/')[1]
#         rm_dir = os.path.join(pwd,'cv-other-train')
#         if filename in os.listdir(rm_dir):
#             os.remove(os.path.join(rm_dir,filename))

# x=91
# count = 0
# dir = os.path.join(pwd,'images/validation/male')
# items = os.listdir(dir)
# for item in items:
#     # print(x)
#     if x==0:
#         break
#     move_dir = os.path.join(pwd,'images/training/female')
#     if item in os.listdir(dir):
#         os.remove(f"{dir}/{item}")
#         # shutil.move(f"{dir}/{item}", move_dir)
#         print(x)
#         x-=1


# CV-TRAIN

def move_files(type, csv):#, size):
    count = 0
    for index, row in df.iterrows():
        count+=1
        print(count)
        if pd.isnull(row['gender']) or pd.isnull(row['accent']):
            pass
        else:
            filename = row['filename'].split('/')[1]
            if filename in files:
                # print(f"Moving {filename} to {type}/{row['gender']}")
                # os.rename(filename,f"{filename}-{row['gender']}")
                move_dir = ''
                if row['gender'] == 'male' or row['gender'] == 'female':
                    move_dir = f"{pwd}/images/{type}/{row['gender']}"
                if not os.path.exists(move_dir):
                    if move_dir != '':
                        os.makedirs(move_dir)
                if move_dir != '' and row['gender'] == 'male':# and \
                        # len([f for f in os.listdir(f"{pwd}/images/{type}/male")]) < size:
                    shutil.copy(f"{pwd}/{csv}/{filename}", move_dir)
                if move_dir != '' and row['gender'] == 'female': # and \
                        # len([f for f in os.listdir(f"{pwd}/images/{type}/female")]) < size:
                    shutil.copy(f"{pwd}/{csv}/{filename}", move_dir)


# move_files('training','cv-valid-train')
# move_files('validation','cv-other-train')

print("training male: ", len(os.listdir('dataset/training/male')))
print("training female: ", len(os.listdir('dataset/training/female')))
print("validation male: ", len(os.listdir('dataset/validation/male')))
print("validation female: ", len(os.listdir('dataset/validation/female')))
print(len(os.listdir('images/training/female')))
# print(len([f for f in os.listdir(f"{pwd}/dataset/validation/female")]))
# print(len(os.listdir('cv-other-train')))
# print(len(os.listdir('cv-other-test')))
# print(len(os.listdir('cv-other-train')))
# print(len(os.listdir('cv-valid-train')))
# print(len(os.listdir('cv-valid-test')))

# for index,row in other_train.iterrows():
#     if pd.isnull(row['gender']) or pd.isnull(row['accent']):
#         pass
#     else:
#         filename = row['filename'].split('/')[1]
#         if filename in files:
#             print('yes')
#             # os.rename(filename,f"{filename}-{row['gender']}")
#             move_dir = f"{pwd}/dataset/training/{row['gender']}"
#             if not os.path.exists(move_dir):
#                 os.makedirs(move_dir)
#             shutil.copy(f"{pwd}/cv-other-train/{filename}", move_dir)





