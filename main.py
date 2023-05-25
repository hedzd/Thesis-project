from skeleton_extractor import make_skeleton_dataset
from skeleton_extractor import file_io

if __name__ == '__main__':
    # change this dir if needed
    txt_files_dir = 'text_dataset'

    f = file_io(txt_files_dir)
    f.train_dataset()
    print(f'number of corrupted files: {len(f.msd.corrupted_files)}')
    print(f'number of processed files: {f.msd.num_processed}')
