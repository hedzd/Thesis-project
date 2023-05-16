import cv2
import mediapipe as mp
import numpy as np

class mp_pose:
    def __init__(self, file_addr):
        _, self.outdir, name = file_addr.split('/')
        self.outdir = '/' + self.outdir + '/'
        self.inflnm, self.inflext = name.split('.')
        self.video_path = file_addr

    def extract_pose(self): 
        mp_drawing = mp.solutions.drawing_utils
        mp_pose = mp.solutions.pose
        pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

        cap = cv2.VideoCapture(self.video_path)

        if cap.isOpened() == False:
            print("Error opening video stream or file")
            raise TypeError

        frame_width = int(cap.get(3))
        frame_height = int(cap.get(4))
        fps = round(cap.get(cv2.CAP_PROP_FPS))

        # outdir = '/content/'
        # inflnm = 'video1'
        # inflext = 'mp4'

        out_filename = f'{self.outdir}{self.inflnm}_annotated_mp.{self.inflext}'
        out = cv2.VideoWriter(out_filename, cv2.VideoWriter_fourcc(
            'M', 'J', 'P', 'G'), fps, (frame_width, frame_height))

        while cap.isOpened():
            ret, image = cap.read()
            if not ret:
                break

            image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = pose.process(image)
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            mp_drawing.draw_landmarks(
                image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            out.write(image)

        pose.close()
        cap.release()
        out.release()
        return out_filename