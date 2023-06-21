from skeleton_extractor import make_skeleton_dataset
from skeleton_extractor import file_io
from skeleton_dataset import unzip_pkls, proc_pkls
from preprocess import proc_data
from pose_models.mediapipe_pose import mediapipe_pose
from action_recognition.eval_models import stgcn_eval
import sys

if __name__ == '__main__':
    # Change this dir if needed
    txt_files_dir = 'text_dataset'

    # f = file_io(txt_files_dir)
    # f.test_dataset()
    # print(f'number of corrupted files: {len(f.msd.corrupted_files)}')
    # print(f'number of processed files: {f.msd.num_processed}')

    # Unzip
    # unzip_pkls(zip_path = "/Users/hediehpourghasem/Downloads/train.zip", extract_to = ".")
    
    # log = open("dfcontent.log", "a")
    # sys.stdout = log
    # Append all dfs
    # proc_pkls(load_dir='./train', final_filename='train_ds', save_dir='./final', nan_frame_tresh = 0.5)
    # sys.stdout = sys.__stdout__

    # Preprocess
    # print('preprocess started')
    # proc_data(load_dir='/content/drive/MyDrive/ds1/final/train_ds.pkl', save_dir='/content/drive/MyDrive/ds1', filename = 'train_processed.pkl')
    # print('preprocess ended')
    # fake_test_val()

    #Eval model
    mp = mediapipe_pose()
    video_file_path = '/Users/hediehpourghasem/Desktop/-5NN5hdIwTc_000036_000046.mp4'
    _, frames_keypoints, _ = mp.extract_pose_keypoints(video_file_path)
    print(frames_keypoints.shape)
    print('start evaluating model')
    y_pred, y_pred_class = stgcn_eval(frames_keypoints)
    print(y_pred_class)
    print(y_pred)