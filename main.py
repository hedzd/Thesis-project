from skeleton_extractor import make_skeleton_dataset
from skeleton_extractor import file_io
from skeleton_dataset import unzip_pkls, proc_pkls
from preprocess import proc_data, fake_test_val

if __name__ == '__main__':
    # Change this dir if needed
    txt_files_dir = 'text_dataset'

    # f = file_io(txt_files_dir)
    # f.test_dataset()
    # print(f'number of corrupted files: {len(f.msd.corrupted_files)}')
    # print(f'number of processed files: {f.msd.num_processed}')

    # Unzip
    # unzip_pkls(zip_path = "/Users/hediehpourghasem/Downloads/train.zip", extract_to = ".")
    
    # Change config if needed
    # proc_pkls()

    # Preprocess
    # proc_data(load_dir='final/train_ds.pkl', save_dir='final', filename = 'processed.pkl')

    fake_test_val()