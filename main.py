from skeleton_extractor import make_skeleton_dataset

if __name__ == '__main__':
    msd = make_skeleton_dataset()
    msd.make_dataset()
    print(f'number of corrupted files: {len(msd.corrupted_files)}')
    print(f'number of processed files: {msd.num_processed}')
