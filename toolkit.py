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

def draw_pic(frame, points):
    connection = [[1,2],[0,2],[0,1],[2,4],[1,3],[3,5],[4,6],[5,6],[6,8],[8,10],[5,7],[7,9],
                  [6,12],[5,11],[11,12],[12,14],[14,16],[11,13],[13,15]]

    for point in points:
        cv2.circle(frame, (int(point[0]), int(point[1])), 5, (0, 255, 0), -1)

    for connection in connection:
        point1 = points[connection[0]]
        point2 = points[connection[1]]
        cv2.line(frame, (int(point1[0]), int(point1[1])), (int(point2[0]), int(point2[1])), (255, 0, 0), 2)

    return frame

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

# if __name__ == "__main__":
#     data = None
#     with open("000002.json", 'r') as file:
#         data = json.load(file)
#     print(extract_kp(data))