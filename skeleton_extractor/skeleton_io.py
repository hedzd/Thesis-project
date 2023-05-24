import tarfile
from pose_models.mediapipe_pose import mediapipe_pose
import pandas as pd
import csv
import os
import re

class make_skeleton_dataset():
    def __init__(self) -> None:
        self.tarfile_path = '/Users/hediehpourghasem/Downloads/part_0.tar.gz'
        self.videos_path = './videos'
        self.csv_path = '/Users/hediehpourghasem/Downloads/train.csv'
        self.mediapipe = mediapipe_pose()
        self.df = pd.read_csv(self.csv_path)
        self.csv_columns = ['file_name','label','keypoints', 'num_nan_frames', 'num_frames']
        self.dataset = []
        new_csv_path = './train'
        if not os.path.exists(new_csv_path):    
            os.mkdir(new_csv_path)
        self.corrupted_files = []
        self.num_processed = 0

    def tarfile_extractor(self, tarfile_path):
        tarfile_name = tarfile_path.split('/')[-1]
        file = tarfile.open(tarfile_path)
        video_names = file.getnames()
        file.extractall(self.videos_path)
        file.close()
        print(f'File {tarfile_name} sucessfully extracted')
        return video_names

    def pose_extractor(self, video_name):
        video_path = self.videos_path + '/' + video_name
        print(video_path)
        keypoints_list, num_none = self.mediapipe.extract_pose_keypoints(video_path)
        return keypoints_list, num_none

    def save_dataset(self, path, dict_data):
        print(path)
        try:
            with open(path, 'w') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=self.csv_columns)
                writer.writeheader()
                for data in dict_data:
                    writer.writerow(data)
        except IOError:
            print("I/O error during saving .csv skeleton dataset")


    def make_dataset(self):
        video_names = self.tarfile_extractor(self.tarfile_path)
        print('extraction complete')
        
        self.num_processed = 0
        for video_name in video_names[1:]:
            name = video_name.split('/')[-1]
            print(f'video name: {name}')
            keypoints, num_none = self.pose_extractor(name)
            if keypoints == None:
                print(f'skip file {name}')
                self.corrupted_files.append(name)
                continue

            file_name = re.sub('_\d{6}_\d{6}.mp4$', '', name)
            # print(file_name)
            row = self.df[self.df['youtube_id'] == file_name]
            print(row)
            video_label = row['label'].iloc[0]
            
            new_row = {'file_name': file_name,'label': video_label,'keypoints': keypoints,
                                 'num_nan_frames': num_none, 'num_frames': len(keypoints)}
            self.dataset.append(new_row)
            self.num_processed += 1
            print(f'Progress = {self.num_processed/len(video_names[1:])}%')

        #TODO: filename
        self.save_dataset('./train/part_0.csv', self.dataset)