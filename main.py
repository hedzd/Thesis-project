from skeleton_extractor import make_skeleton_dataset
from skeleton_extractor import file_io
from skeleton_dataset import unzip_pkls, proc_pkls
from preprocess import proc_data
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
    
    log = open("dfcontent.log", "a")
    sys.stdout = log
    # Append all dfs
    proc_pkls(load_dir='./train', final_filename='train_ds', save_dir='./final', nan_frame_tresh = 0.5)
    sys.stdout = sys.__stdout__
    
    # Preprocess
    proc_data(load_dir='final/train_ds.pkl', save_dir='./final', filename = 'train_processed.pkl')

    # fake_test_val()