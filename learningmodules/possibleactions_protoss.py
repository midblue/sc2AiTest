ACTION_DO_NOTHING = 'donothing'
ACTION_BUILD_PYLON = 'buildpylon'
ACTION_BUILD_GATEWAY = 'buildgateway'
ACTION_BUILD_CYBERNETICSCORE = 'buildcyberneticscore'
ACTION_BUILD_ASSIMILATOR = 'buildassimilator'
ACTION_HARVEST_GAS = 'harvestgas'
ACTION_BUILD_ZEALOT = 'buildzealot'
ACTION_BUILD_STALKER = 'buildstalker'
ACTION_BUILD_ADEPT = 'buildadept'
ACTION_BUILD_SENTRY = 'buildsentry'
ACTION_ATTACK = 'attack'

ai_actions = [
    ACTION_DO_NOTHING,              #0
    ACTION_BUILD_PYLON,             #1
    ACTION_BUILD_GATEWAY,           #2
    ACTION_BUILD_CYBERNETICSCORE,   #3
    ACTION_BUILD_ASSIMILATOR,       #4
    ACTION_HARVEST_GAS,             #5
    ACTION_BUILD_ZEALOT,            #6
    ACTION_BUILD_STALKER,           #7
    ACTION_BUILD_ADEPT,             #8
    ACTION_BUILD_SENTRY,            #9
    #attack own base                #10
    #attack own natural             #11
    #attack enemy natural           #12
    #attack enemy base              #13
]

MINIMAP_TARGET_COLUMNS = 2
MINIMAP_TARGET_GRID_TOTAL_CELLS = MINIMAP_TARGET_COLUMNS * MINIMAP_TARGET_COLUMNS
MINIMAP_TARGET_CELL_WIDTH = int(64 / MINIMAP_TARGET_COLUMNS)
MINIMAP_TARGET_CELL_CENTER_OFFSET = int(MINIMAP_TARGET_CELL_WIDTH / 2)

# import logging
for mm_x in range(0, MINIMAP_TARGET_COLUMNS):
    for mm_y in range(0, MINIMAP_TARGET_COLUMNS):
        ai_actions.append(ACTION_ATTACK + '_' + str((mm_x * MINIMAP_TARGET_CELL_WIDTH) + MINIMAP_TARGET_CELL_CENTER_OFFSET) + '_' + str((mm_y * MINIMAP_TARGET_CELL_WIDTH) + MINIMAP_TARGET_CELL_CENTER_OFFSET))
