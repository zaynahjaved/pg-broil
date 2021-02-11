from .obstacle import Obstacle, ComplexObstacle

"""
Constants associated with the PointBot env.
"""
#POINTBOT NAVIGATION
# MODE = 7
# NOISE_SCALE = 0.05
# TRASH = False
# START_POS = [-170, -130]
# END_POS = [0, 0]
# MAX_FORCE = 1

#TRASHBOT
# MODE = 10
# NOISE_SCALE = 0
# TRASH = True
# MAX_FORCE = 0.5


START_POS = [-170, -130]
END_POS = [0, 0]
GOAL_THRESH = 1.
START_STATE = [START_POS[0], 0, START_POS[1], 0]
GOAL_STATE = [END_POS[0], 0, END_POS[1], 0]

TRASH = False
TRASH_BONUS = 0
TRASH_RADIUS = 1.5
TRASH_BUFFER = 10
START_BUFFER = 5

MAX_FORCE = 1
HORIZON = 100

NOISE_SCALE = 0.05
AIR_RESIST = 0.2

COLLISION_COST = 0
MODE = 7

OBSTACLE = {
	1: ComplexObstacle([[[-1000, -999], [-1000, -999]]]),
	2: ComplexObstacle([[[-90, -20], [-20, 50]]]),
	3: ComplexObstacle([[[-30, -20], [-20, -10]], [[-30, -20], [0, 20]]]),
	4: ComplexObstacle([[[-30, -20], [-20, 20]], [[-20, 5], [10, 20]], [[0, 5], [5, 10]], [[-20, 5], [-20, -10]]]),
	5: ComplexObstacle([[[-195, -150], [-50, 80]], [[-70, -25], [-20, 180]], [[-240, -195], [-50, 0]], [[-240, -70], [130, 180]]]),
	6: ComplexObstacle([[[-150, -25], [-20, 70]], [[-150, -25], [100, 190]]]),
	7: ComplexObstacle([[[-155, -110], [-190, -80]], [[-200, -155], [-190, -140]], [[-70, -25], [-130, 50]], [[-155, -110], [10, 90]], [[-200, 20], [90, 140]], [[-250, -200], [-190, 140]]]),
	8: ComplexObstacle([[[-200, -75], [-190, -140]], [[-75, -25], [-190, 140]], [[-200, -75], [90, 140]], [[-250, -200], [-190, 140]]]),
	9: ComplexObstacle([[[-100, -40], [-95, -70]], [[-40, -15], [-95, 70]], [[-100, -40], [45, 70]], [[-125, -100], [-95, 70]]]),
	10: ComplexObstacle([[[0, 40], [-10, 10]], [[0, 40], [50, 70]], [[-20, 0], [-10, 70]], [[40, 60], [-10, 70]]])

}



