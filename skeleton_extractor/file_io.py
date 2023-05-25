from . import make_skeleton_dataset
import requests
import os
import shutil

class file_io():
    def __init__(self, txt_files_dir) -> None:
        self.txt_files_dir = txt_files_dir
        self.train_addr = self.txt_files_dir + '/k400_train_path.txt'
        self.test_addr = self.txt_files_dir + '/k400_test_path.txt'
        self.val_addr = self.txt_files_dir + '/k400_val_path.txt'
        self.videos_path = './videos'
        self.tar_path = './downloads'
        if not os.path.exists(self.tar_path):    
            os.mkdir(self.tar_path)

        self.msd = make_skeleton_dataset(self.videos_path)
    
    def download_file(self, url, tarfile_addr):
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            with open(tarfile_addr, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192): 
                    # If you have chunk encoded response uncomment if
                    # and set chunk_size parameter to None.
                    #if chunk: 
                    f.write(chunk)
    
    def make_datasets(self, txt_addr, csv_path, new_csv_path):
        file = open(txt_addr, 'r')
        Lines = file.readlines()

        for line in Lines:
            tarfile_url = line.strip() 
            print(f'downloading {tarfile_url}')
            tarfile_name = tarfile_url.split('/')[-1]
            tarfile_addr = self.tar_path + '/' + tarfile_name
            print(tarfile_addr)
            print('starting download')
            self.download_file(tarfile_url, tarfile_addr)

        # For test
        # tarfile_addr = '/Users/hediehpourghasem/Downloads/part_0.tar.gz'

        # make csv dataset with keypoints
            self.msd.make_dataset(tarfile_addr, csv_path, new_csv_path)

            #remove videos
            os.remove(tarfile_addr)
            try:
                shutil.rmtree(self.videos_path)
                print("videos directory is removed successfully")
            except OSError as x:
                print("Error occured: %s : %s" % (self.videos_path, x.strerror))

    def train_dataset(self):
        csv_path = 'text_dataset/train.csv'
        new_csv_path = './train'
        if not os.path.exists(new_csv_path):    
            os.mkdir(new_csv_path)
        
        self.make_datasets(self.train_addr, csv_path, new_csv_path)


    def test_dataset(self):
        csv_path = 'text_dataset/test.csv'
        new_csv_path = './test'
        if not os.path.exists(new_csv_path):    
            os.mkdir(new_csv_path)

        self.make_datasets(self.test_addr, csv_path, new_csv_path)
        

    def val_dataset(self):
        csv_path = 'text_dataset/val.csv'
        new_csv_path = './val'
        if not os.path.exists(new_csv_path):    
            os.mkdir(new_csv_path)
        
        self.make_datasets(self.val_addr, csv_path, new_csv_path)
        
