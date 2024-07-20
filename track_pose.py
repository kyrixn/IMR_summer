import cv2
import matplotlib.pyplot as plt
from mmpose.apis import MMPoseInferencer
import numpy as np
import json

from toolkit import process_frame, draw_pic
from lpf import LowPassFilter

lpf = LowPassFilter(
            sampling_frequency=30, damping_frequency=3, damping_intensity=0.5, outlier_threshold=60
        )

def extract_kp(res):
    highest_avg_score = -1
    best_keypoints_group = None

    for group in res:
        keypoint_scores = group['keypoint_scores']
        avg_score = np.mean(keypoint_scores)
        if avg_score > highest_avg_score:
            highest_avg_score = avg_score
            best_keypoints_group = group

    return best_keypoints_group['keypoints']

def get_kp(frame, inferencer):
    new_frame = process_frame(frame)

    result_generator = inferencer(new_frame, show=False)
    result = next(result_generator)

    keypoints = result["predictions"][0]
    
    return np.array(extract_kp(keypoints))    

def get_arm_len(v_source, inferencer):
    r_arm = []
    l_arm = []
    cap = cv2.VideoCapture(v_source)
    frame_cnt = 0
    while frame_cnt < 10:
        #ret = 1
        ret, frame = cap.read()
        if ret:
            cur_kp = get_kp(frame, inferencer)
            #cur_kp = read_json("000000.json")

            r_arm.append(np.linalg.norm(np.array(cur_kp[6]) - np.array(cur_kp[10])))
            l_arm.append(np.linalg.norm(np.array(cur_kp[5]) - np.array(cur_kp[9])))
        else:
            print("Failed to capture frame")
            break
        frame_cnt+=1
    
    return [np.mean(r_arm), np.mean(l_arm)]

def get_arm_angle(cur_kp, r_arm, l_arm):

    def calc_angle(diff, arm_length):
        arm_ratio = np.clip(diff / arm_length, -1, 1)  # Ensure the ratio is within [-1, 1]
        angle = np.arccos(abs(arm_ratio))
        return [arm_ratio, np.degrees(angle)]

    # Calculate right and left arm ratios and angles
    [cur_r_arm, r_angle] = calc_angle(cur_kp[6][1] - cur_kp[10][1], r_arm)
    [cur_l_arm, l_angle] = calc_angle(cur_kp[5][1] - cur_kp[9][1], l_arm)

    return [r_angle, l_angle]

def track_pose_2D(path, inferencer):
    all_kp = []
    cap = cv2.VideoCapture(path)

    [r_arm, l_arm] = get_arm_len(path, inferencer)
    r_angle=[]; l_angle=[]

    i=0
    while True:
        try:
            ret, frame = cap.read()
            if ret:
                keypoints = get_kp(frame, inferencer)
                all_kp.append(np.array(keypoints))

                [cur_r_angle, cur_l_angle] = get_arm_angle(keypoints, r_arm, l_arm)
                r_angle.append(cur_r_angle);l_angle.append(cur_l_angle)
            else:
                print("Failed to capture frame")
                break
        except RuntimeError as e:
            print("An error occurred", e)

        i+=1
        if i % 10 == 0:
            print(str(i)+" frame completed")

    cap.release()
    filtered_l_angles = []; filtered_r_angles=[]
    for angle in l_angle:
        filtered_l_angles.append(lpf.update(angle))
    lpf.reset()
    for angle in r_angle:
         filtered_r_angles.append(lpf.update(angle))
    
    return [all_kp, filtered_r_angles, filtered_l_angles, np.argmax(r_angle), np.argmax(l_angle)]

# function for visualization
def draw_skeleton(data):
    connection = [[1,2],[0,2],[0,1],[2,4],[1,3],[3,5],[4,6],[5,6],[6,8],[8,10],[5,7],[7,9],
                  [6,12],[5,11],[11,12],[12,14],[14,16],[11,13],[13,15]]
    rotated_kp = np.array(data)
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    
    for edge in connection:
        [x1,y1] = rotated_kp[edge[0]]
        [x2,y2] = rotated_kp[edge[1]]
        ax.plot([x1, x2], [y1, y2], 'red', linewidth = 0.75)

    for i, (x, y) in enumerate(rotated_kp):
        ax.scatter(x, y, marker='o',c = 'g', s =20)

        if connection[i] != -1:
            parent_index = connection[i]
            px, py = rotated_kp[parent_index]

    min_point = rotated_kp.min(axis=0)
    max_point = rotated_kp.max(axis=0)
    max_range = np.array([max_point[i] - min_point[i] for i in range(2)]).max() / 2.0

    mid_x = (max_point[0] + min_point[0]) * 0.5
    mid_y = (max_point[1] + min_point[1]) * 0.5

    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)

    ax.invert_yaxis()
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    
    plt.title('keypoints and skeleton')
    plt.legend(loc='upper left')
    plt.show()

def read_json(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data[0]['keypoints']

if __name__ == "__main__":
    inferencer = MMPoseInferencer('human')
    path = "001.mp4"
    [b,c,d,e,f] = track_pose_2D(path, inferencer)
    #np.save('r2.npy', c); np.save('l2.npy', d)
    #c = np.load("r1.npy")
    #d = np.load("l1.npy")
    print(e)
    print(f)
    fig, ax = plt.subplots()

    ax.plot(c, label='Right Arm')
    ax.plot(d, label='Left Arm')

    ax.set_title('Arm Track', fontsize=20)
    ax.set_xlabel('frame', fontsize=20)
    ax.set_ylabel('Angle (degrees)', fontsize=20)

    ax.legend(fontsize=16)
    plt.show()
