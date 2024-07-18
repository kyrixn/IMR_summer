import os
from os.path import dirname
import json
import time
from glob import glob

import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

def read_label(txt_path):
    file = open(txt_path, 'r', encoding='utf-8').readlines()
    sub_id = []
    label = []
    length = len(file)
    for i in range(0, length-1):
        label.append(file[i+1].strip('\n').split('\t'))
        sub_id.append(label[i][0])

    return np.asarray(label).astype(np.float32), np.asarray(sub_id).astype(np.uint8)


def decode_depth_16(rgb):
    assert (rgb.dtype == np.uint8)
    r, g, b = cv2.split(rgb)
    depth = (((r.astype(np.uint16) + g.astype(np.uint16))/2) + (b.astype(np.uint16) // 16) * 256).astype(np.uint16)
    return depth


def read_intrinsics(filename):
    json_name = glob(dirname(filename) + '/*Param_*.json')[0]
    try:
        data = json.load(open(json_name, 'r'))
        intrinsics = np.array([
            [data['fy'], 0, data['height'] - data['ppy'] - 1],
            [0, data['fx'], data['ppx']],
            [0, 0, 1]
        ])
        return intrinsics

    except:
        print('No json file {} found'.format(json_name))

def process_frame(frame):
        color = frame[0:720, :]
        depth = frame[720:1440, :]
        depth = decode_depth_16(depth)

        color = cv2.rotate(color, cv2.ROTATE_90_COUNTERCLOCKWISE)
        depth = cv2.rotate(depth, cv2.ROTATE_90_COUNTERCLOCKWISE)

        return color


def test1():
    dataroot = './NIHSS_UpperLimbs/'

    label, sub_id = read_label(dataroot + 'labels.txt')

    deom_id = 0
    mp4 = glob(dataroot + '/*/{:05d}/*.mp4'.format(sub_id[deom_id]))[0]
    left_arm_label, right_arm_label = label[deom_id][1], label[deom_id][2]
    cap = cv2.VideoCapture(mp4)

    intrinsic = read_intrinsics(mp4)

    cv2.namedWindow("Color and Depth Image", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Color and Depth Image", 800, 600)
    while True:
        t0 = time.time()
        ret, frame = cap.read()
        if not ret:
            break
        
        show_image = process_frame(frame)
        
        cv2.imshow("Color and Depth Image", show_image)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()

def fit_plane(points):
    A = np.c_[points[:, 0], points[:, 1], np.ones(points.shape[0])]
    C, _, _, _ = np.linalg.lstsq(A, points[:, 2], rcond=None)  # coefficients
    normal = np.array([-C[0], -C[1], 1])
    return normal / np.linalg.norm(normal)

def fit_line(points):
    direction = points[-1] - points[0]
    return direction / np.linalg.norm(direction)

# Calculate the angle between the two arm lines
def angle_between_vectors(v1, v2):
    cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    angle = np.arccos(cos_angle)
    return np.degrees(angle)


def calc_angle(kp):
    body_idx = [14,8,11,7,0,1,4]
    left_arm_idx = [11,12,13]
    right_arm_idx = [14,15,16]

    body_points = kp[body_idx]
    left_arm_points = kp[left_arm_idx]
    right_arm_points = kp[right_arm_idx]
    # Fit the plane to the body points
    body_plane_normal = fit_plane(body_points)

    left_arm_direction = fit_line(left_arm_points)
    right_arm_direction = fit_line(right_arm_points)

    left_arm_body_angle = angle_between_vectors(left_arm_direction, body_plane_normal)
    right_arm_body_angle = angle_between_vectors(right_arm_direction, body_plane_normal)

    #print(f"Angle between left arm line and body plane: {left_arm_body_angle:.2f} degrees")
    #print(f"Angle between right arm line and body plane: {right_arm_body_angle:.2f} degrees")
    return [left_arm_body_angle, right_arm_body_angle]

def angle_track(all_kp):
    right_angle = []
    left_angle = []
    for cur_kp in all_kp:
        [tmpl, tmpr] = calc_angle(cur_kp)
        right_angle.append(tmpr)
        left_angle.append(tmpl)

    left_out = np.array(left_angle)-90
    right_out = np.array(right_angle)-90

    return [savgol_filter(left_out, 10, 2), savgol_filter(right_out, 10, 2)]

if __name__ == "__main__":
    arr = np.load("array3d.npy")
    [l,r] = angle_track(arr)
    print(l)
    fig = plt.figure()
    plt.plot(l)
    plt.plot(r)
    plt.show()








