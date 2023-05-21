from pose_io import uploader
from pose_io import downloader
import time

input_url = ''

def pose_video_extractor(method, input_url):
    out_dir = '/tmp/'
    file_name = time.time()
    file_format = '.mp4'
    dl_file_addr = out_dir + file_name + file_format
    annotated_addr = f'{out_dir}{file_name}_annotated_mp.{file_format}'

    dl = downloader(input_url, dl_file_addr)
    dl.download()

    if method == 'mediapipe':
        import pose_models.mediapipe_pose as mediapipe_pose
        annotated_addr = f'{out_dir}{file_name}_annotated_mp.{file_format}'
        mediapipe = mediapipe_pose()
        mediapipe.extrsave_extract_poseact_pose(dl_file_addr, annotated_addr)

    up = uploader(annotated_addr)
    up.upload()


    
