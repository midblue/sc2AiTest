
import logging
import random
from collections import deque
from sklearn.cluster import KMeans
from learningmodules.objects import *
from pysc2.lib import features

_UNIT_TYPE = features.SCREEN_FEATURES.unit_type.index

def getCountOfObjectOnScreen(obs, id):
  x, y = getCoordsOfObjectOnScreen(obs, id)
  if not id in SIZES:
    logging.info('No saved size for object ' + id + '. Total Pixels: ' + str(len(y)))
    return -1
  # logging.info(id +' '+ str(SIZES[id]))
  return int((len(y) + (SIZES[id]/2)) / (SIZES[id]))

def getPointOnObjectOnScreen(obs, object):
	x, y = getCoordsOfObjectOnScreen(obs, object)
	if y.any():
		randomPoint = random.randint(0, len(y) - 1)
		return [x[randomPoint], y[randomPoint]]

def getCoordsOfObjectOnScreen(obs, object):
	unit_type = obs.observation['screen'][_UNIT_TYPE]
	unit_y, unit_x = (unit_type == object).nonzero()
	return (unit_x, unit_y)

def getCenterOfOneObject(obs, object):
    x, y = getCoordsOfObjectOnScreen(obs, object)
    count = getCountOfObjectOnScreen(obs, object)
    if count > 0:
        pairs = []
        for i in range(0, len(y)):
            pairs.append((x[i], y[i]))
        kmeans = KMeans(n_clusters=count)
        kmeans.fit(pairs)
        r = random.randint(0, count - 1)
        return (int(kmeans.cluster_centers_[r][0]), int(kmeans.cluster_centers_[r][1]))

def isSpecificTargetOnMap(obs, x, y):
    attackable = [1, 2, 4]
    target = obs.observation['minimap'][features.SCREEN_FEATURES.player_relative.index][x][y]
    if target in attackable:
      return True
    return False

def findGenericTargetOnMap(obs, x, y):
    if not isSpecificTargetOnMap(obs, x, y):
        return x, y
    offset = 0
    while True:
        for i in range(-offset, offset):
            for j in range(-offset, offset):
                ox = x + i
                oy = y + j
                if x > 63 or y > 63 or x < 0 or y < 0:
                    continue
                else:
                    if not isSpecificTargetOnMap(obs, ox, oy):
                        return ox, oy
        offset += 1

def isSelected(obs, object):
    if len(obs.observation['single_select']) > 0 and obs.observation['single_select'][0][0] == object:
        return True
    if len(obs.observation['multi_select']) > 0 and obs.observation['multi_select'][0][0] == object:
        return True
    return False

class Memory:
    def __init__(self, max_size=300000):
        self.max_size=max_size
        self.reset()

    def reset(self):
        self.buffer = deque(maxlen=self.max_size)

    def push(self, state, action, reward):
        self.buffer.append((str(state), action, reward))

    def pop(self):
        if self.buffer:
            return self.buffer.pop()

    def len(self):
        return len(self.buffer)
