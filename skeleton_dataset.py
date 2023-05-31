import os
import pandas as pd
import shutil

class SkeletonConfig:
    mode: str = 'train'
    load_dir: str = ''
    save_dir: str = ''
    filename: str = ''
    nan_frame_tresh: float = 0.5

def unzip_pkls():
    zip_path = ""
    extract_to = ""
    shutil.unpack_archive(zip_path, extract_to)

def proc_gait_data(config: SkeletonConfig = SkeletonConfig()):
    result_df = pd.DataFrame()

    for filename in os.listdir(config.load_dir):
        if filename.endswith('.pkl'): 
            file_path = os.path.join(config.load_dir, filename)  
            with open(load_dir, "rb") as f:
                df = pd.read_pickle(f)
            print(f"Processing file: {file_path}")

            # Eliminate files with too many nan frames
            df = df[df['num_nan_frames']/df['num_frames'] < config.nan_frame_tresh].reset_index(drop=True)
            print('Eliminated files with too many nan frames')
            df.drop(columns=['num_nan_frames', 'num_frames'], inplace=True)

            result_df = result_df.append(df1)
    
    print("Number of samples per labels in final dataset:")
    print(result_df['label'].value_counts())
    result_df.to_pickle(config.save_dir + config.filename + '.pkl')  

