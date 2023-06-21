import torch
import numpy as np
from .models.st_gcn.st_gcn import Model as stgcn_model

classes = ['archery', 'bench pressing', 'bouncing on trampoline', 'bowling' 'clapping',
 'climbing a rope', 'cracking neck', 'crawling baby', 'dancing macarena',
 'disc golfing', 'doing aerobics', 'dribbling basketball',
 'dunking basketball', 'grinding meat', 'hammer throw', 'high jump',
 'high kick', 'hockey stop', 'hurdling', 'jogging', 'jumping into pool',
 'kicking soccer ball', 'playing drums', 'playing tennis', 'playing ukulele',
 'playing violin', 'pole vault', 'presenting weather forecast', 'pull ups',
 'recording music', 'riding mechanical bull', 'riding or walking with horse',
 'robot dancing', 'running on treadmill', 'shearing sheep', 'skiing slalom',
 'sword fighting', 'tying bow tie']

def stgcn_eval(input_data):
    # define model
    num_classes = 38
    model = stgcn_model(3, num_classes, True, ('mediapose', 'uniform',))

    # load weights
    weight_path = 'action_recognition/weights/29'
    model.load_state_dict(torch.load(weight_path, map_location='cpu'))

    # set model to evaluation mode
    model.to("cpu", dtype=float)
    model.eval()

    # preprocess data
    print('starting preprocess')
    input_data = preprocess(input_data)
    x = torch.tensor(input_data)
    # print(x.size()) # B=batch size, T, V, C=3
    x = torch.permute(x, (0, 3, 1, 2)) # B=batch size, C=2, T, V
    x = torch.unsqueeze(x, dim=-1)
    print(f'input tensor size {x.size()}')

    # call model
    output = model(x)

    # apply log softmax
    m = torch.nn.LogSoftmax(dim=1)
    output = m(output)
    y_pred = output.argmax(-1)
    y_pred_class = classes[y_pred]
    return y_pred, y_pred_class

def repeat_array_to_length(arr, target_length):
    repeats = int(np.ceil(target_length / arr.shape[0]))
    repeated_arr = np.tile(arr, (repeats, 1))
    result = repeated_arr[:target_length,:]
    return result

def preprocess(raw_data):
    num_features = 3
    num_nodes = 33

    raw_data = np.expand_dims(raw_data, axis=0)
    print(raw_data.shape)

    num_frames = [r.shape[0] for r in raw_data]
    max_frame = 300
    num_samples = 1   
    data = np.zeros((num_samples, max_frame, num_nodes, num_features)) # N, T, V, C
    print('Make N, T, V, C with zeros', flush=True)

    for idx, r in enumerate(raw_data):
        # print(f'start processing data with index {idx}')
        # Eliminate completely nan frames
        # if config.filter_nan_frames:
        r = r[~np.isnan(r).any(axis=1), :]

        # Padding frames
        r = repeat_array_to_length(r, 300)
        
        sample_feature = np.stack(np.split(r, num_nodes, axis=1), axis=1) # T, V, C
        # print(sample_feature.shape)

        data[idx, :] = sample_feature
        # print(f'finish processing data with index {idx}')
    
    print(f'Shape change to N,T,V,C format, shape: {data.shape}', flush=True)

    return data

