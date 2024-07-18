import json
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import pprint

connection = [-1,0,1,2,0,4,5,0,7,8,9,8,11,12,8,14,15]

def rotate_x(points, theta):
    theta = np.radians(theta)
    c, s = np.cos(theta), np.sin(theta)
    R = np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    #R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    #R = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    return np.dot(points, R.T)

def read_json(file_path):

    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

def draw_skeleton(data):
    allarr = []
    keypoints = data[0]['keypoints']
    keypoints = np.array(keypoints)
    allarr.append(keypoints)
    allarr.append(keypoints)
    all_arrays = np.array(allarr)
    print(all_arrays[0])
    
    rotated_kp = np.array([rotate_x(np.array(point), 0) for point in keypoints])

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    for i, (x, y, z) in enumerate(rotated_kp):
        ax.scatter(x, y, z, marker='o',c = 'g', s =3.5)

        if connection[i] != -1:
            parent_index = connection[i]
            px, py, pz = rotated_kp[parent_index]
            ax.plot([x, px], [y, py], [z, pz], 'red', linewidth = 0.75)  

    ax.view_init(elev=20, azim=50)

    min_point = rotated_kp.min(axis=0)
    max_point = rotated_kp.max(axis=0)
    max_range = np.array([max_point[i] - min_point[i] for i in range(3)]).max() / 2.0

    mid_x = (max_point[0] + min_point[0]) * 0.5
    mid_y = (max_point[1] + min_point[1]) * 0.5
    mid_z = (max_point[2] + min_point[2]) * 0.5

    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.title('keypoints and skeleton')
    plt.legend(loc='upper left')
    plt.show()

if __name__ == "__main__":
    dt = read_json("000000.json")
    draw_skeleton(dt)