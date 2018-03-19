import numpy as np
import matplotlib.pyplot as plt
from threading import Timer
from time import sleep

calculation_unit = 20
WIN = "1"
LOSE = "-1"
TIE = "0"

def printGraph():
    print('drawing')
    results = []
    win_dicts = {}
    lose_dicts = {}
    tie_dicts = {}
    with open('result.txt', 'r') as f:
        for row in f:
            results.append(row.strip())

    win_count = 0
    lose_count = 0
    tie_count = 0
    for i, result in enumerate(results):
        if result == WIN:
            win_count += 1
        elif result == LOSE:
            lose_count += 1
        else:
            tie_count += 1

        # 一定回数毎に集計
        if (i + 1) % calculation_unit is 0:
            win_dicts[str(int((i + 1) / calculation_unit))] = win_count / calculation_unit
            lose_dicts[str(int((i + 1) / calculation_unit))] = lose_count / calculation_unit
            tie_dicts[str(int((i + 1) / calculation_unit))] = tie_count / calculation_unit
            tie_count = 0
            win_count = 0
            lose_count = 0
   
    plt.plot(np.array(list(win_dicts.keys())), np.array(list(win_dicts.values())), color="#00bb00")
    plt.plot(np.array(list(tie_dicts.keys())), np.array(list(tie_dicts.values())), color="#bbbbbb")
    plt.plot(np.array(list(lose_dicts.keys())), np.array(list(lose_dicts.values())), color="#bb0000")
    plt.draw()

plt.ion()
plt.xlabel("number of trials")
plt.ylabel("percentage of victories(%)")
# plt.show()
while True:
    printGraph()
    input('key to reprint')
