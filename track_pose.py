import cv2
import matplotlib.pyplot as plt
from mmpose.apis import MMPoseInferencer
import numpy as np

from toolkit import process_frame

def track_kp(path, inferencer):
    all_kp = []
    
    cap = cv2.VideoCapture(path)  # 0 is usually the default camera

    i =0
    while True:
        try:
            ret, frame = cap.read()
            if ret:
                new_frame = process_frame(frame)

                result_generator = inferencer(new_frame, show=False)
                result = next(result_generator)

                keypoints = result["predictions"][0][0]["keypoints"]
                keypoints = np.array(keypoints)

                all_kp.append(keypoints)
            else:
                print("Failed to capture frame")
                break
        except RuntimeError as e:
            print("An error occurred", e)

        i+=1
        if i % 10 == 0:
            print(str(i)+" frame completed")

    cap.release()

    return np.array(all_kp)

if __name__ == "__main__":
    inferencer_3d = MMPoseInferencer(pose3d="human3d")
    path = "000.mp4"
    res = track_kp(path, inferencer_3d)
    res = np.round(res,3)
    np.save('array3d.npy', res)