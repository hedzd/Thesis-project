import tarfile
from pose_models.mediapipe_pose import mediapipe_pose
import pandas as pd
import csv
import re
import pickle

class make_skeleton_dataset():
    def __init__(self, videos_path) -> None:
        self.videos_path = videos_path
        self.mediapipe = mediapipe_pose()
        # self.csv_columns = ['file_name','label','keypoints', 'num_nan_frames', 'num_frames']
        self.corrupted_files = []
        self.num_processed = 0
        self.num_eliminated = 0
        self.keep_labels =  ['taking a shower', 'tying bow tie', 'using segway', 'yawning',
       'sniffing', 'shredding paper', 'cooking on campfire', 'digging',
       'drinking shots', 'blowing leaves', 'unloading truck',
       'springboard diving', 'swinging legs', 'drumming fingers',
       'bending metal', 'applauding', 'climbing a rope',
       'recording music', 'grinding meat', 'whistling', 'exercising arm',
       'jogging', 'news anchoring', 'water sliding', 'reading newspaper',
       'rock scissors paper', 'sharpening knives', 'moving furniture',
       'making tea', 'washing hair', 'building shed', 'plastering',
       'running on treadmill', 'holding snake', 'waiting in line',
       'bee keeping', 'building cabinet', 'gargling', 'laying bricks',
       'making sushi', 'exercising with an exercise ball',
       'making a sandwich', 'faceplanting', 'garbage collecting',
       'stomping grapes', 'shooting goal (soccer)', 'drawing',
       'cleaning pool', 'cracking neck', 'breading or breadcrumbing',
       'sign language interpreting', 'peeling potatoes', 'changing wheel',
       'doing laundry', 'doing aerobics', 'tossing coin', 'making a cake',
       'tossing salad', 'pushing wheelchair', 'slapping',
       'cooking sausages', 'hockey stop', 'playing kickball', 'spraying',
       'strumming guitar', 'sword fighting', 'playing flute',
       'riding mule', 'skiing crosscountry', 'massaging feet']

    def tarfile_extractor(self, tarfile_path):
        tarfile_name = tarfile_path.split('/')[-1]
        file = tarfile.open(tarfile_path)
        video_names = file.getnames()
        file.extractall(self.videos_path)
        file.close()
        print(f'File {tarfile_name} sucessfully extracted')
        return video_names, tarfile_name

    def pose_extractor(self, video_name):
        video_path = self.videos_path + '/' + video_name
        print(video_path)
        is_corrupted, keypoints_list, num_none = self.mediapipe.extract_pose_keypoints(video_path)
        return is_corrupted, keypoints_list, num_none

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
    
    def save_pickle(self, path, dict_data):
        pickle.dump((dict_data), open(path + '.pkl', 'ab'))

    def make_dataset(self, tarfile_path, csv_path, new_csv_path):
        video_names, tarfile_name = self.tarfile_extractor(tarfile_path)
        print('extraction complete')
        df = pd.read_csv(csv_path)
        
        filename_list = []
        videolabel_list = []
        keypoints_list = []
        numnan_list = []
        numframes_list = []

        self.num_processed = 0
        self.num_eliminated = 0
        for video_name in video_names[1:]:
            name = video_name.split('/')[-1]
            print(f'video name: {name}')

            file_name = re.sub('_\d{6}_\d{6}.mp4$', '', name)
            row = df[df['youtube_id'] == file_name]
            video_label = row['label'].iloc[0]

            if video_label in self.keep_labels:

                is_corrupted, keypoints, num_none = self.pose_extractor(name)
                if is_corrupted:
                    print(f'skip file {name}')
                    self.corrupted_files.append(name)
                    self.num_processed += 1
                    continue

                print(row)
                
                print(f'type keypoints {type(keypoints)}')

                filename_list.append(file_name)
                videolabel_list.append(video_label)
                keypoints_list.append(keypoints)
                numnan_list.append(num_none)
                numframes_list.append(len(keypoints))

                self.num_processed += 1

            # Uncomment for saving csv
            # new_row = {'file_name': file_name,'label': video_label,'keypoints': keypoints,
            #                      'num_nan_frames': num_none, 'num_frames': len(keypoints)}
            # dataset.append(new_row)
            else:
                self.num_eliminated +=1
            
            print(f'File = {tarfile_name} , Progress = {(self.num_eliminated+self.num_processed)*100/len(video_names[1:])}%')
        
        # Uncomment for saving csv
        # new_csv_addr = new_csv_path + '/' + tarfile_path.split('/')[-1].split('.')[0] + '.csv'
        pickle_addr = new_csv_path + '/' + tarfile_path.split('/')[-1].split('.')[0]

        # Uncomment for saving csv
        # self.save_dataset(new_csv_addr, dataset)

        dataset_df = pd.DataFrame({'file_name':filename_list,
                   'label':videolabel_list,
                   'keypoints':keypoints_list,
                   'num_nan_frames': numnan_list,
                   'num_frames': numframes_list
                   })
        dataset_df.to_pickle(pickle_addr+'.pkl')  