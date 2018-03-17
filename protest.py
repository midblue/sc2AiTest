from pysc2.agents import base_agent
from pysc2.lib import actions
from pysc2.lib import features

import random
import math
import logging
import numpy as np

from learningmodules.qlearning import QLearningTable
from learningmodules.possibleactions_protoss import *
from learningmodules.objects import *
from learningmodules.helpers import *

_NO_OP = actions.FUNCTIONS.no_op.id
_SELECT_POINT = actions.FUNCTIONS.select_point.id
_HARVEST = actions.FUNCTIONS.Harvest_Gather_screen.id
_BUILD_PYLON = actions.FUNCTIONS.Build_Pylon_screen.id
_BUILD_GATEWAY = actions.FUNCTIONS.Build_Gateway_screen.id
_BUILD_CYBERNETICSCORE = actions.FUNCTIONS.Build_CyberneticsCore_screen.id
_BUILD_ASSIMILATOR = actions.FUNCTIONS.Build_Assimilator_screen.id
_BUILD_ZEALOT = actions.FUNCTIONS.Train_Zealot_quick.id
_BUILD_SENTRY = actions.FUNCTIONS.Train_Sentry_quick.id
_BUILD_STALKER = actions.FUNCTIONS.Train_Stalker_quick.id
_BUILD_ADEPT = actions.FUNCTIONS.Train_Adept_quick.id
_SELECT_ARMY = actions.FUNCTIONS.select_army.id
_ATTACK_MINIMAP = actions.FUNCTIONS.Attack_minimap.id

_PLAYER_RELATIVE = features.SCREEN_FEATURES.player_relative.index
_UNIT_TYPE = features.SCREEN_FEATURES.unit_type.index
_PLAYER_ID = features.SCREEN_FEATURES.player_id.index

_PLAYER_SELF = 1
_PLAYER_HOSTILE = 4

_NOT_QUEUED = [0]
_QUEUED = [1]

# --

KILL_UNIT_REWARD = 0.1
KILL_BUILDING_REWARD = 0.5

class Agent(base_agent.BaseAgent):

  def transformDistance(self, x, x_distance, y, y_distance):
    if not self.base_top_left:
      return [x - x_distance, y - y_distance]
    return [x + x_distance, y + y_distance]

  def transformLocation(self, x, y):
    if not self.base_top_left:
      return [64 - x, 64 - y]
    return [x, y]

  def getPointOnObjectOnScreen(obs, self, object):
    unit_type = obs.observation['screen'][_UNIT_TYPE]
    unit_y, unit_x = (unit_type == object).nonzero()
    if unit_y.any():
      randomPoint = random.randint(0, len(unit_y) - 1)
      return [unit_x[randomPoint], unit_y[randomPoint]]
    return False

  def __init__(self):
    super(Agent, self).__init__()
    self.qlearn = QLearningTable( actions = list(range(len(ai_actions))) )
    self.previous_killed_unit_score = 0
    self.previous_killed_building_score = 0
    self.previous_action = None
    self.previous_state = None

  def step(self, obs):
    super(Agent, self).step(obs)

    # logging.info(obs.observation)

    player_y, player_x = (obs.observation['minimap'][_PLAYER_RELATIVE] == _PLAYER_SELF).nonzero()
    self.base_top_left = 1 if player_y.any() and player_y.mean() <= 31 else 0

    pylon_count = getCountOfObjectOnScreen(obs, PROTOSS_PYLON, 'PROTOSS_PYLON')
    gateway_count = getCountOfObjectOnScreen(obs, PROTOSS_PYLON, 'PROTOSS_PYLON')
    assimilator_count = getCountOfObjectOnScreen(obs, PROTOSS_PYLON, 'PROTOSS_PYLON')

    minerals = obs.observation['player'][1]
    gas = obs.observation['player'][2]
    supply_limit = obs.observation['player'][4]
    army_supply = obs.observation['player'][5]

    temp_state = [
      minerals,
      gas,
      pylon_count,
      gateway_count,
      assimilator_count,
      supply_limit,
      army_supply,
    ]
    state_length = len(temp_state)

    current_state = np.zeros(16 + state_length)
    for i in range(0, state_length):
      current_state[i] = temp_state[i]

    hot_squares = np.zeros(16)        
    enemy_y, enemy_x = (obs.observation['minimap'][_PLAYER_RELATIVE] == _PLAYER_HOSTILE).nonzero()
    for i in range(0, len(enemy_y)):
      y = int(math.ceil((enemy_y[i] + 1) / 16))
      x = int(math.ceil((enemy_x[i] + 1) / 16))
      hot_squares[((y - 1) * 4) + (x - 1)] = 1
    
    if not self.base_top_left:
      hot_squares = hot_squares[::-1]
    
    for i in range(0, 16):
      current_state[i + 4] = hot_squares[i]

    # logging.info(current_state)

    reward = 0
    killed_unit_score = obs.observation['score_cumulative'][5]
    killed_building_score = obs.observation['score_cumulative'][6]

    if self.previous_action is not None:
      if killed_unit_score != self.previous_killed_unit_score:
        if killed_unit_score > self.previous_killed_unit_score:
          reward += KILL_UNIT_REWARD
        self.previous_killed_unit_score = killed_unit_score

      if killed_building_score != self.previous_killed_building_score:
        if killed_unit_score > self.previous_killed_unit_score:
          reward += KILL_BUILDING_REWARD
        self.previous_killed_building_score = killed_building_score

      self.qlearn.learn(str(self.previous_state), self.previous_action, reward, str(current_state))

    rl_action = self.qlearn.choose_action(str(current_state))
    self.previous_state = current_state
    self.previous_action = rl_action

    ai_action = ai_actions[rl_action]

    x = 0
    y = 0
    if '_' in ai_action:
      ai_action, x, y = ai_action.split('_')
      x = int(x)
      y = int(y)

    if ai_action == ACTION_DO_NOTHING:
      return actions.FunctionCall(_NO_OP, [])

    elif ai_action == ACTION_SELECT_PROBE:
      unit_type = obs.observation['screen'][_UNIT_TYPE]
      unit_y, unit_x = (unit_type == PROTOSS_PROBE).nonzero()
      if unit_y.any():
        i = random.randint(0, len(unit_y) - 1)
        target = [unit_x[i], unit_y[i]]
        return actions.FunctionCall(_SELECT_POINT, [_NOT_QUEUED, target])

    elif ai_action == ACTION_BUILD_PYLON:
      if _BUILD_PYLON in obs.observation['available_actions']:
        unit_type = obs.observation['screen'][_UNIT_TYPE]
        unit_y, unit_x = (unit_type == PROTOSS_NEXUS).nonzero()
        if unit_y.any():
          target = self.transformDistance(int(unit_x.mean()), 0, int(unit_y.mean()), 15)
          return actions.FunctionCall(_BUILD_PYLON, [_NOT_QUEUED, target])

    elif ai_action == ACTION_BUILD_GATEWAY:
      if _BUILD_GATEWAY in obs.observation['available_actions']:
        unit_type = obs.observation['screen'][_UNIT_TYPE]
        unit_y, unit_x = (unit_type == PROTOSS_NEXUS).nonzero()
        if unit_y.any():
          target = self.transformDistance(int(unit_x.mean()), 15, int(unit_y.mean()), 0)
          return actions.FunctionCall(_BUILD_GATEWAY, [_NOT_QUEUED, target])

    elif ai_action == ACTION_BUILD_CYBERNETICSCORE:
      if _BUILD_CYBERNETICSCORE in obs.observation['available_actions']:
        unit_type = obs.observation['screen'][_UNIT_TYPE]
        unit_y, unit_x = (unit_type == PROTOSS_NEXUS).nonzero()
        if unit_y.any():
          target = self.transformDistance(int(unit_x.mean()), 13, int(unit_y.mean()), 13)
          return actions.FunctionCall(_BUILD_CYBERNETICSCORE, [_NOT_QUEUED, target])

    elif ai_action == ACTION_BUILD_ASSIMILATOR:
      if _BUILD_ASSIMILATOR in obs.observation['available_actions']:
        target = getPointOnObjectOnScreen(obs, NEUTRAL_VESPENEGEYSER)
        if target != False:
          return actions.FunctionCall(_BUILD_ASSIMILATOR, [_NOT_QUEUED, target])

    elif ai_action == ACTION_HARVEST_GAS:
      if _HARVEST in obs.observation['available_actions']:
        target = getPointOnObjectOnScreen(obs, PROTOSS_ASSIMILATOR)
        if target != False:
          return actions.FunctionCall(_HARVEST, [_NOT_QUEUED, target])

    elif ai_action == ACTION_SELECT_GATEWAY:
      unit_type = obs.observation['screen'][_UNIT_TYPE]
      unit_y, unit_x = (unit_type == PROTOSS_GATEWAY).nonzero()
      if unit_y.any():
        target = [int(unit_x.mean()), int(unit_y.mean())]
        return actions.FunctionCall(_SELECT_POINT, [_NOT_QUEUED, target])

    elif ai_action == ACTION_BUILD_ZEALOT:
      if _BUILD_ZEALOT in obs.observation['available_actions']:
        # logging.info('Zealot')
        return actions.FunctionCall(_BUILD_ZEALOT, [_QUEUED])

    elif ai_action == ACTION_BUILD_STALKER:
      if _BUILD_STALKER in obs.observation['available_actions']:
        # logging.info('Stalker')
        return actions.FunctionCall(_BUILD_STALKER, [_QUEUED])

    elif ai_action == ACTION_BUILD_SENTRY:
      if _BUILD_SENTRY in obs.observation['available_actions']:
        # logging.info('Sentry')
        return actions.FunctionCall(_BUILD_SENTRY, [_QUEUED])

    elif ai_action == ACTION_BUILD_ADEPT:
      if _BUILD_ADEPT in obs.observation['available_actions']:
        # logging.info('Adept')
        return actions.FunctionCall(_BUILD_ADEPT, [_QUEUED])

    elif ai_action == ACTION_SELECT_ARMY:
      if _SELECT_ARMY in obs.observation['available_actions']:
        return actions.FunctionCall(_SELECT_ARMY, [_NOT_QUEUED])
                  
    elif ai_action == ACTION_ATTACK:
      if obs.observation['single_select'][0][0] != PROTOSS_PROBE and _ATTACK_MINIMAP in obs.observation["available_actions"]:
        return actions.FunctionCall(_ATTACK_MINIMAP, [_NOT_QUEUED, self.transformLocation(x, y)])

    return actions.FunctionCall(_NO_OP, [])
