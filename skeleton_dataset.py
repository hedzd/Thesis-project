import os
import pandas as pd
import shutil

# class SkeletonConfig:
#     load_dir: str = './train'
#     save_dir: str = './final'
#     filename: str = 'train_ds'
#     nan_frame_tresh: float = 0.5

def unzip_pkls(zip_path: str, extract_to: str):
    shutil.unpack_archive(zip_path, extract_to)
    print(f'Pickle files unzipped in {extract_to}')

def proc_pkls(load_dir, filename, save_dir='./final', nan_frame_tresh = None):
    result_df = pd.DataFrame()

    for filename in os.listdir(load_dir):
        if filename.endswith('.pkl'):
            file_path = os.path.join(load_dir, filename)  
            with open(file_path, "rb") as f:
                df = pd.read_pickle(f)
            # print(f"Processing file: {file_path}")

            # Eliminate files with too many nan frames
            df = df[df['num_nan_frames']/df['num_frames'] < nan_frame_tresh].reset_index(drop=True)
            # print('Eliminated files with too many nan frames')
            df.drop(columns=['num_nan_frames', 'num_frames'], inplace=True)

            result_df = pd.concat([result_df, df])
    
    result_df = result_df.reset_index(drop=True)

    print("Number of samples per labels in final dataset:")
    print(result_df['label'].value_counts(ascending=True))
    if not os.path.exists(save_dir):    
            os.mkdir(save_dir)
    result_df.to_pickle(save_dir + '/' + filename + '.pkl')  

