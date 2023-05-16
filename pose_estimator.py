from pose_io import uploader
from pose_io import downloader
import time

input_url = ''

def pose_video_extractor(method, input_url):
    out_dir = '/tmp/'
    file_name = time.time()
    file_format = '.mp4'
    dl_file_addr = out_dir + file_name + file_format

    dl = downloader(input_url, dl_file_addr)
    dl.download()

    if method == 'mediapipe':
        from mediapipe import mp_pose
        mediapipe = mp_pose(dl_file_addr)
        annotated_addr = mediapipe.extract_pose()

    up = uploader(annotated_addr)
    up.upload()


    
