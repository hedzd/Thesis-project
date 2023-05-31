import numpy as np
import pandas as pd

class ProcessingConfig:
    num_per_class: int = None
    min_sample_thresh: int = None
    filter_nan_frames: bool = True
    max_frame: int = 300
    filter_visibility: bool = False

def repeat_array_to_length(arr, target_length, axis=None):
    repeats = int(np.ceil(target_length / arr.shape[axis]))
    repeated_arr = np.tile(arr, (repeats, 1))
    result = repeated_arr[:target_length,:]
    return result

def proc_gait_data(load_dir: str, save_dir: str, filename: str="processed.pkl", 
        config: ProcessingConfig = ProcessingConfig()) -> None:
    """ Processes Raw dataset (pickle file) provided by MediaPipe """
    
    num_features = 3
    num_nodes = 33

    with open(load_dir, "rb") as f:
        df = pd.read_pickle(f)

    # Eliminate classes with too few sample data
    if config.min_sample_thresh != None:
        v = df.label.value_counts()
        df = df[df.label.isin(v.index[v.gt(config.min_sample_thresh)])].reset_index(drop=True)

    # Set num data per class
    if config.num_per_class != None:
        new_df = pd.DataFrame(columns=list(df.columns))
        label_ls = df['label'].unique()
        for l in label_ls:
            df_l = df[df['label'] == l]
            num_rows = df_l.shape[0]
            if config.num_per_class > num_rows:
                new_df = pd.concat([new_df, df_l])
            else:
                new_df = pd.concat([new_df, df_l.sample(n = config.num_per_class, random_state=12345)])
        df = new_df

    
    raw_data = df['keypoints'].values    
    labels = df['label'].values
    names = df['file_name'].values

    # Change shape to N, T, V, C
    num_frames = [r.shape[0] for r in raw_data]
    max_frame = config.max_frame
    num_samples = raw_data.shape[0]   
    data = np.zeros((num_samples, max_frame, num_nodes, num_features)) # N, T, V, C

    for idx, r in enumerate(raw_data):
        # Eliminate completely nan frames
        if filter_nan_frames:
            r = r[~np.isnan(r).any(axis=1), :]

        # Padding frames
        r = repeat_array_to_length(r, config.max_frame, 1)
        
        sample_feature = np.stack(np.split(r, num_nodes, axis=1), axis=1) # T, V, C

        data[idx, :] = sample_feature

    # Eliminating visibility
    if config.filter_visibility:
        data = data[:,:,:,:2]

    with open(os.path.join(save_dir, filename), 'wb') as f:
        pickle.dump((data, labels, names), f)


    