import gym
import numpy as np
import cv2
import math
import time
import sys

INDICATOR_HEIGHT_FACTOR = 1/8.0
NUM_NODES_IN_PATH = 10
SAFE_RADIUS = 50
DES_TRAJ_NODE_IND_1 = 2
DES_TRAJ_NODE_IND_2 = 3
CONTROLLER_GAIN = 0.0005
TRACK_WIDTH_TO_WINDOW_WIDTH = 1/2.0

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
        last_point[:] = [x, y]

    num_nodes_in_path = max(1, min(len(left_edge_points), len(right_edge_points), NUM_NODES_IN_PATH))
    left_step = len(left_edge_points)/num_nodes_in_path
    right_step = len(right_edge_points)/num_nodes_in_path
    print len(left_edge_points), len(right_edge_points)

    left_nodes = left_edge_points[left_step-1::left_step]
    right_nodes = right_edge_points[right_step-1::right_step]
    mid_nodes = []
    for i in xrange(num_nodes_in_path):
        mid_nodes.append([(left_nodes[i][0] + right_nodes[i][0])/2, (left_nodes[i][1] + right_nodes[i][1])/2])

    # Sort path nodes such that they form continuous trajectory starting from the vehicle
    # Since the x position of the vehicle is anchored at half of window width,
    # the first point of the sorted nodes is the lowermost point in a thin window in the middle.
    # After that, the consecutive points are found by the least distance criteria.
    mid_nodes_sorted = np.empty([num_nodes_in_path,2],dtype=int)
    y_max = 0
    i_max = 0
    for i, [x, y] in enumerate(mid_nodes):
        if (1 + TRACK_WIDTH_TO_WINDOW_WIDTH)*width/2 > x > (1 - TRACK_WIDTH_TO_WINDOW_WIDTH)*width/2 and y > y_max:
            y_max = y
            i_max = i
    mid_nodes_sorted[0,:] = mid_nodes[i_max]
    mid_nodes.pop(i_max)

    sort_ind = 0
    while mid_nodes:    # While the list is not empty
        min_dist = width**2 + height**2     # Some high value
        curr_node = mid_nodes_sorted[sort_ind, :]
        for i, [x, y] in enumerate(mid_nodes):
            dist = distance_point_to_point(curr_node, [x, y])
            if dist < min_dist:
                min_dist = dist
                i_min = i
        sort_ind += 1
        mid_nodes_sorted[sort_ind,:] = mid_nodes[i_min]
        mid_nodes.pop(i_min)

    return mid_nodes_sorted, left_edge_points, right_edge_points


def distance_point_from_line(line_p1, line_p2, p):
    x1 = line_p1[0]
    y1 = line_p1[1]
    x2 = line_p2[0]
    y2 = line_p2[1]
    xp = p[0]
    yp = p[1]
    dist = float((y1 - y2)*xp + (x2 - x1)*yp - x2*y1 + y2*x1)/math.sqrt((y2 - y1)**2 + (x2 - x1)**2)
    return dist


def distance_point_to_point(p1, p2):
    x1 = p1[0]
    y1 = p1[1]
    x2 = p2[0]
    y2 = p2[1]
    return math.sqrt((y2 - y1)**2 + (x2 - x1)**2)


def race():

    for frame_i in xrange(sys.maxint):

        env.render()

        # Ignore first 2 seconds : the "zoom-in" part
        if frame_i < 100:
            observation, reward, done, info = env.step(np.array([0, 0, 0]))
            continue

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
        car_cent = np.array([car_x, car_y])
        mask_track_copy = np.copy(mask_track)
        _, contours_track, _ = cv2.findContours(mask_track_copy, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        # Among all the grey colored contours, track is the one with maximum area
        max_area = 0
        for i, contour in enumerate(contours_track):
            area = cv2.contourArea(contour)
            if area > max_area:
                max_area = area
                i_max = i
        cnt_track = contours_track[i_max][:, 0, :]

        t1 = time.time()
        path_nodes, track_left_lim, track_right_lim = find_path_nodes(cnt_track, play_width, play_height)

        # print "time to find path = ", time.time() - t1

        for i in xrange(len(path_nodes)):
            grafitti[path_nodes[i, 1], path_nodes[i, 0], :] = [255, 255, 255]

        for [x,y] in track_left_lim:
            grafitti[y, x, :] = [255, 0, 0]

        for [x,y] in track_right_lim:
            grafitti[y, x, :] = [0, 0, 255]

        grafitti[car_y, car_x] = [0, 255, 0]

        num_path_nodes = path_nodes.shape[0]
        desired_traj = np.array([path_nodes[DES_TRAJ_NODE_IND_2, ], path_nodes[DES_TRAJ_NODE_IND_1, ]])
        cv2.line(grafitti, (desired_traj[0,0], desired_traj[0,1]), (desired_traj[1,0], desired_traj[1,1]), (255, 255, 255), 1)
        crosstrack_error = distance_point_from_line(desired_traj[0,:], desired_traj[1,:], car_cent)
        heading_error = math.pi/2 + math.atan2(desired_traj[0,1] - desired_traj[1,1], desired_traj[0,0] - desired_traj[1,0])
        print "crosstrack error", crosstrack_error, "heading error", heading_error

        # Speeding law
        speed = 0.05

        # Steering law
        steer = heading_error + math.atan(CONTROLLER_GAIN * crosstrack_error / speed)
        #steer = heading_error

        cv2.imshow('frame', grafitti)

        observation, reward, done, info = env.step(np.array([steer, speed, 0]))

        if (cv2.waitKey(1) & 0xFF == ord('q')):
            #print "track edges", track_edges
            break
        if done:
            print "Reward = ", reward
            break

    cv2.destroyAllWindows()

if __name__ == '__main__':
    race()
