import gym
import numpy as np
import cv2
import math
from scipy.spatial import Voronoi, voronoi_plot_2d
import time

INDICATOR_HEIGHT_FACTOR = 1/8.0
NUM_NODES_IN_PATH = 10
SAFE_RADIUS = 50

env = gym.make('CarRacing-v0')
env.reset()
car_img = cv2.imread('car_shape.png', cv2.IMREAD_GRAYSCALE)
_, temp_contours, _ = cv2.findContours(car_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
car_contour = temp_contours[0]


def find_path_nodes(contour_track, width, height):

    # Just like Voronoi algorithm, the idea here is to find the safest path
    # (mid-way of left and right edge)

    # First remove points which are part of the frame boundary so that we just have left and right edges of the track
    track_edges = contour_track.tolist()
    track_edges = [[x,y] for [x,y] in track_edges if x!=1 and x!=width-2 and y!=1 and y!=height-2]
    #print "track edges", track_edges

    edge_number = 0
    last_point = track_edges[0]
    left_edge_points = []
    right_edge_points = []
    ind = -1
    for [x,y] in track_edges[1:]:

        if not(abs(x - last_point[0]) == 1 or abs(y - last_point[1]) == 1):     # check continuity
            edge_number += 1            # toggle edge type on no continuity

        if edge_number%2 == 0:
            if edge_number > 0:
                ind += 1
                left_edge_points.insert(ind, [x,y])
            else:
                left_edge_points.append([x,y])
        else:
            right_edge_points.insert(0, [x,y])
        last_point[:] = [x,y]

    num_nodes_in_path = max(1,min(len(left_edge_points), len(right_edge_points), NUM_NODES_IN_PATH))
    left_step = len(left_edge_points)/num_nodes_in_path
    right_step = len(right_edge_points)/num_nodes_in_path

    left_nodes = np.array(left_edge_points[left_step-1::left_step])
    right_nodes = np.array(right_edge_points[right_step-1::right_step])

    mid_nodes = (left_nodes + right_nodes)/2
    #print "path nodes", path_nodes
    #print "car center", car_x, car_y
    return mid_nodes, left_edge_points, right_edge_points


def race():

    for frame_i in range(400):

        env.render()

        # Ignore first 2 seconds : the "zoom-in" part
        if frame_i < 100:
            observation, reward, done, info = env.step(np.array([0, 0, 0]))
            continue

        observation, reward, done, info = env.step(np.array([0.0, 0.01, 0]))
        win_height, win_width = observation.shape[0:2]

        play_height = int((1-INDICATOR_HEIGHT_FACTOR) * win_height)
        play_width = win_width
        indicator_height = int(INDICATOR_HEIGHT_FACTOR * win_height)

        # An image to draw different things for debugging
        grafitti = np.zeros((play_height, win_width, 3), dtype=np.uint8)

        #observation = cv2.resize(observation, None, fx=4, fy=4, interpolation=cv2.INTER_LINEAR)
        playfield_roi = observation[0:play_height,:,:]
        indicator_roi = observation[play_height:,:,:]

        # Conversion to HSV space is not really requried in this application
        # where the colors are not real-world, BGR should work just fine
        # playfield_roi = cv2.cvtColor(playfield_roi, cv2.COLOR_BGR2HSV)

        lower_red = np.array([200, 0, 0])
        upper_red = np.array([255, 50, 50])

        lower_grey = np.array([90, 90, 90])
        upper_grey = np.array([110, 110, 110])

        mask_car = cv2.inRange(playfield_roi, lower_red, upper_red)
        mask_track = cv2.inRange(playfield_roi, lower_grey, upper_grey)

        mask_copy = np.copy(mask_car)
        _, contours, _ = cv2.findContours(mask_copy, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        match_shape_res = 100*np.ones(len(contours)) # Some high value

        for i, contour in enumerate(contours):
            match_shape_res[i] = cv2.matchShapes(contour, car_contour, 3, 0.0)

        car_cnt_ind = np.argmin(match_shape_res)
        car_cnt = contours[car_cnt_ind][:,0,:]

        (car_x, car_y), (_, _), car_angle = cv2.minAreaRect(car_cnt)
        rect = cv2.minAreaRect(car_cnt)
        mask_track_copy = np.copy(mask_track)
        _, contours_track, _ = cv2.findContours(mask_track_copy, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        # TODO: Randomly selecting first contour. Need a better way to correctly identify the contour car is in
        cnt_track = contours_track[0][:,0,:]

        t1 = time.time()
        path_nodes, track_left_lim, track_right_lim = find_path_nodes(cnt_track, play_width, play_height)
        # print "time to find path = ", time.time() - t1

        for i in xrange(len(path_nodes)):
            grafitti[path_nodes[i,1], path_nodes[i,0], :] = [255,255,255]

        for [x,y] in track_left_lim:
            grafitti[y, x, :] = [255,0,0]

        for [x,y] in track_right_lim:
            grafitti[y, x, :] = [0,0,255]

        grafitti[car_y, car_x] = [0, 255, 0]

        cv2.imshow('frame', grafitti)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            #print "track edges", track_edges
            break


    # TODO: Ignore nodes which are within specified radius from the car to avoid jitter in control

    cv2.destroyAllWindows()

if __name__ == '__main__':
    race()
