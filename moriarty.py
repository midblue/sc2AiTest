from pysc2.agents import base_agent
from pysc2.lib import actions
from pysc2.lib import features

import random
import math
import logging

from learningmodules.qlearning import QLearningTable
from learningmodules.possibleactions import *


_NO_OP = actions.FUNCTIONS.no_op.id
_SELECT_POINT = actions.FUNCTIONS.select_point.id
_BUILD_SUPPLY_DEPOT = actions.FUNCTIONS.Build_SupplyDepot_screen.id
_BUILD_BARRACKS = actions.FUNCTIONS.Build_Barracks_screen.id
_TRAIN_MARINE = actions.FUNCTIONS.Train_Marine_quick.id
_SELECT_ARMY = actions.FUNCTIONS.select_army.id
_ATTACK_MINIMAP = actions.FUNCTIONS.Attack_minimap.id

_PLAYER_RELATIVE = features.SCREEN_FEATURES.player_relative.index
_UNIT_TYPE = features.SCREEN_FEATURES.unit_type.index
_PLAYER_ID = features.SCREEN_FEATURES.player_id.index

_PLAYER_SELF = 1

_TERRAN_COMMANDCENTER = 18
_TERRAN_SCV = 45 
_TERRAN_SUPPLY_DEPOT = 19
_TERRAN_BARRACKS = 21

_NOT_QUEUED = [0]
_QUEUED = [1]

# --

KILL_UNIT_REWARD = 0.1
KILL_BUILDING_REWARD = 0.5

class Agent(base_agent.BaseAgent):

  def transformLocation(self, x, x_distance, y, y_distance):
    if not self.base_top_left:
      return [x - x_distance, y - y_distance]
    return [x + x_distance, y + y_distance]

  def __init__(self):
    super(Agent, self).__init__()
    self.qlearn = QLearningTable( actions = list(range(len(ai_actions))) )
    self.previous_killed_unit_score = 0
    self.previous_killed_building_score = 0
    self.previous_action = None
    self.previous_state = None

  def step(self, obs):
    super(Agent, self).step(obs)
    player_y, player_x = (obs.observation['minimap'][_PLAYER_RELATIVE] == _PLAYER_SELF).nonzero()
    self.base_top_left = 1 if player_y.any() and player_y.mean() <= 31 else 0

    unit_type = obs.observation['screen'][_UNIT_TYPE]

    depot_y, depot_x = (unit_type == _TERRAN_SUPPLY_DEPOT).nonzero()
    supply_depot_count = 1 if depot_y.any() else 0

    barracks_y, barracks_x = (unit_type == _TERRAN_BARRACKS).nonzero()
    barracks_count = 1 if barracks_y.any() else 0

    supply_limit = obs.observation['player'][4]
    army_supply = obs.observation['player'][5]

    current_state = [
      supply_depot_count,
      barracks_count,
      supply_limit,
      army_supply,
    ]

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

    if ai_action == ACTION_DO_NOTHING:
      return actions.FunctionCall(_NO_OP, [])

    elif ai_action == ACTION_SELECT_SCV:
      unit_type = obs.observation['screen'][_UNIT_TYPE]
      unit_y, unit_x = (unit_type == _TERRAN_SCV).nonzero()
      if unit_y.any():
        i = random.randint(0, len(unit_y) - 1)
        target = [unit_x[i], unit_y[i]]
        return actions.FunctionCall(_SELECT_POINT, [_NOT_QUEUED, target])

    elif ai_action == ACTION_BUILD_SUPPLY_DEPOT:
      if _BUILD_SUPPLY_DEPOT in obs.observation['available_actions']:
        unit_type = obs.observation['screen'][_UNIT_TYPE]
        unit_y, unit_x = (unit_type == _TERRAN_COMMANDCENTER).nonzero()
        if unit_y.any():
          target = self.transformLocation(int(unit_x.mean()), 0, int(unit_y.mean()), 15)
          return actions.FunctionCall(_BUILD_SUPPLY_DEPOT, [_NOT_QUEUED, target])

    elif ai_action == ACTION_BUILD_BARRACKS:
      if _BUILD_BARRACKS in obs.observation['available_actions']:
        unit_type = obs.observation['screen'][_UNIT_TYPE]
        unit_y, unit_x = (unit_type == _TERRAN_COMMANDCENTER).nonzero()
        if unit_y.any():
          target = self.transformLocation(int(unit_x.mean()), 15, int(unit_y.mean()), 0)
          return actions.FunctionCall(_BUILD_BARRACKS, [_NOT_QUEUED, target])

    elif ai_action == ACTION_SELECT_BARRACKS:
      unit_type = obs.observation['screen'][_UNIT_TYPE]
      unit_y, unit_x = (unit_type == _TERRAN_BARRACKS).nonzero()
      if unit_y.any():
        target = [int(unit_x.mean()), int(unit_y.mean())]
        return actions.FunctionCall(_SELECT_POINT, [_NOT_QUEUED, target])
            
    elif ai_action == ACTION_BUILD_MARINE:
      if _TRAIN_MARINE in obs.observation['available_actions']:
        return actions.FunctionCall(_TRAIN_MARINE, [_QUEUED])
                  
    elif ai_action == ACTION_SELECT_ARMY:
      if _SELECT_ARMY in obs.observation['available_actions']:
        return actions.FunctionCall(_SELECT_ARMY, [_NOT_QUEUED])
                  
    elif ai_action == ACTION_ATTACK:
      if _ATTACK_MINIMAP in obs.observation["available_actions"]:
        if self.base_top_left:
          return actions.FunctionCall(_ATTACK_MINIMAP, [_NOT_QUEUED, [39, 45]])                           
        return actions.FunctionCall(_ATTACK_MINIMAP, [_NOT_QUEUED, [20, 24]])

    return actions.FunctionCall(_NO_OP, [])

  # def step(self, obs):
  #   super(Agent, self).step(obs)

  #   if self.base_top_left is None:
  #     player_y, player_x = (obs.observation["minimap"][_PLAYER_RELATIVE] == _PLAYER_SELF).nonzero()
  #     self.ccX = int(player_x.mean())
  #     self.ccY = int(player_y.mean())
  #     self.base_top_left = player_y.mean() <= 31
  #     logging.info(self.base_top_left)

  #   if not self.supply_depot_built:
  #     if not self.scv_selected:
  #       unit_type = obs.observation["screen"][_UNIT_TYPE]
  #       unit_y, unit_x = (unit_type == _TERRAN_SCV).nonzero()
      #       target = [unit_x[0], unit_y[0
      #       self.scv_selected = Tr
  #       return actions.FunctionCall(_SELECT_POINT, [_NOT_QUEUED, target])
    
  #     elif _BUILD_SUPPLYDEPOT in obs.observation["available_actions"]:
  #       unit_type = obs.observation["screen"][_UNIT_TYPE]
  #       unit_y, unit_x = (unit_type == _TERRAN_COMMANDCENTER).nonzero()
  #       target = self.transformLocation(int(unit_x.mean()), 20, int(unit_y.mean()), 0)
  #       self.supply_depot_built = True
  #       logging.info('building sd')
  #       logging.info(target)
  #       return actions.FunctionCall(_BUILD_SUPPLYDEPOT, [_NOT_QUEUED, target])

  #   elif not self.barracks_built:
  #     if _BUILD_BARRACKS in obs.observation["available_actions"]:
  #       unit_type = obs.observation["screen"][_UNIT_TYPE]
  #       unit_y, unit_x = (unit_type == _TERRAN_COMMANDCENTER).nonzero()
  #       target = self.transformLocation(int(unit_x.mean()), 20, int(unit_y.mean()), 20)
  #       self.barracks_built = True
  #       return actions.FunctionCall(_BUILD_BARRACKS, [_NOT_QUEUED, target])
      
  #   elif not self.barracks_rallied:
  #     if not self.barracks_selected:
  #       unit_type = obs.observation["screen"][_UNIT_TYPE]
  #       unit_y, unit_x = (unit_type == _TERRAN_BARRACKS).nonzero()
  #       if unit_y.any():
  #           target = [int(unit_x.mean()), int(unit_y.mean())]
  #           self.barracks_selected = True
  #           return actions.FunctionCall(_SELECT_POINT, [_NOT_QUEUED, target])
  #     else:
  #       self.barracks_rallied = True
  #       target = self.transformLocation(self.ccX, 7, self.ccY, 2)
  #       return actions.FunctionCall(_RALLY_UNITS_MINIMAP, [_NOT_QUEUED, target])

  #   elif obs.observation["player"][_SUPPLY_USED] < obs.observation["player"][_SUPPLY_MAX] and not self.barracks_selected:
  #     if not self.barracks_selected:
  #       unit_type = obs.observation["screen"][_UNIT_TYPE]
  #       unit_y, unit_x = (unit_type == _TERRAN_BARRACKS).nonzero()
  #       if unit_y.any():
  #           target = [int(unit_x.mean()), int(unit_y.mean())]
  #           self.barracks_selected = True
  #           return actions.FunctionCall(_SELECT_POINT, [_NOT_QUEUED, target])

  #   elif obs.observation["player"][_SUPPLY_USED] < obs.observation["player"][_SUPPLY_MAX] and _TRAIN_MARINE in obs.observation["available_actions"]:
  #     logging.info('train marine')
  #     return actions.FunctionCall(_TRAIN_MARINE, [_QUEUED])

  #   else:
  #     if not self.army_selected:
  #       if _SELECT_ARMY in obs.observation["available_actions"]:
  #         self.army_selected = True
  #         self.barracks_selected = False
  #         return actions.FunctionCall(_SELECT_ARMY, [_NOT_QUEUED])
  #     elif _ATTACK_MINIMAP in obs.observation["available_actions"]:
  #       self.army_selected = False
  #       target = self.transformLocation(self.ccX, 25, self.ccY, 25)
  #       logging.info(target)
  #       return actions.FunctionCall(_ATTACK_MINIMAP, [_NOT_QUEUED, target])
  #     else:
  #       self.army_selected = False

  #   return actions.FunctionCall(_NOOP, [])
    
  # def transformLocation(self, x, x_distance, y, y_distance):
  #   if not self.base_top_left:
  #       return [x - x_distance, y - y_distance]
    
  #   return [x + x_distance, y + y_distance]