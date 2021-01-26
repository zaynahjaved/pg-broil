from .obstacle import Obstacle, ComplexObstacle

"""
Constants associated with the PointBot env.
"""
START_POS = [-110, 0]
END_POS = [0, 0]
GOAL_THRESH = 4.
START_STATE = [START_POS[0], 0, START_POS[1], 0]
GOAL_STATE = [END_POS[0], 0, END_POS[1], 0]

TRASH = True
TRASH_BONUS = 0
TRASH_RADIUS = 5
NUM_TRASH_LOCS = 5
TRASH_LOCS = [(-170, -115), (-85, -105)]
TRASH_BUFFER = 10

MAX_FORCE = 1
HORIZON = 100

NOISE_SCALE = 0.05
AIR_RESIST = 0.2

COLLISION_COST = 0
MODE = 8

OBSTACLE = {
	1: ComplexObstacle([[[-1000, -999], [-1000, -999]]]),
	2: ComplexObstacle([[[-90, -20], [-20, 50]]]),
	3: ComplexObstacle([[[-30, -20], [-20, -10]], [[-30, -20], [0, 20]]]),
	4: ComplexObstacle([[[-30, -20], [-20, 20]], [[-20, 5], [10, 20]], [[0, 5], [5, 10]], [[-20, 5], [-20, -10]]]),
	5: ComplexObstacle([[[-195, -150], [-50, 80]], [[-70, -25], [-20, 180]], [[-240, -195], [-50, 0]], [[-240, -70], [130, 180]]]),
	6: ComplexObstacle([[[-150, -25], [-20, 70]], [[-150, -25], [100, 190]]]),
	7: ComplexObstacle([[[-155, -110], [-190, -80]], [[-200, -155], [-190, -140]], [[-70, -25], [-130, 50]], [[-155, -110], [10, 90]], [[-200, 20], [90, 140]], [[-250, -200], [-190, 140]]]),
	8: ComplexObstacle([[[-200, -75], [-190, -140]], [[-75, -25], [-190, 140]], [[-200, -75], [90, 140]], [[-250, -200], [-190, 140]]]) 

}