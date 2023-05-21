import tarfile
from ..pose_models import mediapipe_pose as mp_pose
import pandas as pd
import csv

tarfile_path = '/Users/hediehpourghasem/Downloads/part_0.tar.gz'
videos_path = './videos'
csv_path = '/Users/hediehpourghasem/Downloads/train.csv'
mediapipe = mp_pose()
df = pd.read_csv(csv_path)
csv_columns = ['file_name','label','keypoints']
dataset = []

def tarfile_extractor(tarfile_path):
    tarfile_name = tarfile_path.split('/')[-1]
    file = tarfile.open(tarfile_path)
    video_names = file.getnames()
    file.extractall(videos_path)
    file.close()
    print(f'File {tarfile_name} sucessfully extracted')
    return video_names

def pose_extractor(video_name):
    video_path = videos_path + video_name
    keypoints_list = mediapipe.extract_pose_keypoints(video_path)
    print(keypoints_list)
    print(len(keypoints_list))
    return keypoints_list

def save_dataset(path, dict_data):
    try:
        with open(path, 'w') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
            writer.writeheader()
            for data in dict_data:
                writer.writerow(data)
    except IOError:
        print("I/O error during saving .csv skeleton dataset")


def make_skeleton_dataset():
    video_names = tarfile_extractor(tarfile_path)
    
    for name in video_names:
        keypoints = pose_extractor(name)

        row = df[df['youtube_id'] == '-EkAGBzhWe4']
        video_label = row['label'].iloc[0]

        file_name = name.split('.')[0].split('_')[0]
        dataset.append({'file_name': file_name,'label': video_label,'keypoints': keypoints})

    save_dataset('./train/part_0.csv', dataset)





    
    
    
    








