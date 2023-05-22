import cv2
import mediapipe as mp
import numpy as np

class mediapipe_pose:
    def __init__(self):
        return

    def landmarks_list_to_array(self, landmark_list):
        # print(landmark_list)
        keypoints = []
        if landmark_list is None:
            new_row = {
                np.nan
            }
            keypoints.append(new_row)
        else:
            for data_point in landmark_list.landmark:
                keypoints.append({
                                    'X': data_point.x,
                                    'Y': data_point.y,
                                    'Z': data_point.z,
                                    'Visibility': data_point.visibility,
                                    })
        return keypoints

    def save_extract_pose(self, in_path, out_path): 
        mp_drawing = mp.solutions.drawing_utils
        mp_pose = mp.solutions.pose
        pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

        cap = cv2.VideoCapture(in_path)

        if cap.isOpened() == False:
            raise Exception("Error opening video stream or file")

        frame_width = int(cap.get(3))
        frame_height = int(cap.get(4))
        fps = round(cap.get(cv2.CAP_PROP_FPS))

        out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(
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
        print('Mediapipe annotated file successfully saved in {out_path}')

    def extract_pose_keypoints(self, in_path):
        frames_keypoints = []
        mp_pose = mp.solutions.pose
        pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

        cap = cv2.VideoCapture(in_path)

        if cap.isOpened() == False:
            raise Exception("Error opening video stream or file")

        frame_width = int(cap.get(3))
        frame_height = int(cap.get(4))

        while cap.isOpened():
            ret, image = cap.read()
            if not ret:
                break

            image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = pose.process(image)
            array = self.landmarks_list_to_array(results.pose_landmarks)
            frames_keypoints.append(array)

        pose.close()
        cap.release()
 
        return frames_keypoints
