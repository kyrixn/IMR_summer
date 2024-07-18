import json
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

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
    keypoints = data[0]['keypoints']

    rotated_kp = [rotate_x(np.array(point), 0) for point in keypoints]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    for i, (x, y, z) in enumerate(rotated_kp):
        ax.scatter(x, y, z, marker='o')
        ax.text(x, y, z, f' {i}', color='blue', fontsize=7)

        if connection[i] != -1:
            parent_index = connection[i]
            px, py, pz = rotated_kp[parent_index]
        ax.plot([x, px], [y, py], [z, pz], 'gray')  

    ax.view_init(elev=20, azim=50)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.title('keypoints and skeleton')
    plt.legend(loc='upper left')
    plt.show()

if __name__ == "__main__":
    dt = read_json("000000.jpg")
    draw_skeleton(dt)