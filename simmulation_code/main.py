from FileWriter import FileWriter

from algorithms.muUCB1 import muUCB1
from algorithms.UCB1 import UCB1
from algorithms.MLI import MLI
from algorithms.GradientBandit import *
from algorithms.VariableGradientBandit import *
from algorithms.DiscountedUCB1 import DiscountedUCB1
from algorithms.VariableDiscountedUCB1 import VariableDiscountedUCB1
from algorithms.SlidingWindowUCB1 import SlidingWindowUCB1
from algorithms.VariableSlidingWindowUCB1 import VariableSlidingWindowUCB1
from algorithms.SlidingWindowMLI import SlidingWindowMLI
from algorithms.VariableSlidingWindowMLI import VariableSlidingWindowMLI
from algorithms.DiscountedMLI import DiscountedMLI
from algorithms.VariableDiscountedMLI import VariableDiscountedMLI
from algorithms.egreedy import egreedy
from algorithms.Variableegreedy import Variableegreedy
from algorithms.SlidingWindowmuUCB1 import SlidingWindowmuUCB1
from algorithms.DiscountedmuUCB1 import DiscountedmuUCB1
from algorithms.VariableDiscountedmuUCB1 import VariableDiscountedmuUCB1
from algorithms.VariableSlidingWindowmuUCB1 import VariableSlidingWindowmuUCB1

from Scenario1 import Scenario1
from Scenario2 import Scenario2
from Scenario3 import Scenario3
from Scenario4 import Scenario4

strat = {
    "muUCB1": muUCB1,
    "MLI": MLI,
    "e-greedy": egreedy,
    "UCB1": UCB1,
    "GradientBandit": GradientBandit,
    "Sliding-Window-muUCB1": SlidingWindowmuUCB1,
    "VariableSliding-Window-muUCB1": VariableSlidingWindowmuUCB1,
    "Discounted-muUCB1": DiscountedmuUCB1,
    "VariableDiscounted-muUCB1": VariableDiscountedmuUCB1,
    "Sliding-WindowMLI": SlidingWindowMLI,
    "VariableSliding-WindowMLI": VariableSlidingWindowMLI,
    "DiscountedMLI": DiscountedMLI,
    "VariableDiscountedMLI": VariableDiscountedMLI,
    "Variable_e-greedy": Variableegreedy,
    "Sliding-WindowUCB1": SlidingWindowUCB1,
    "VariableSliding-WindowUCB1": VariableSlidingWindowUCB1,
    "DiscountedUCB1": DiscountedUCB1,
    "VariableDiscountedUCB1": VariableDiscountedUCB1,
    "VariableGradientBandit": VariableGradientBandit,
}

scenario = Scenario1()

fw = FileWriter(scenario)
fw.write_settings(strat.keys())

mean = 300
maxG = np.zeros((mean, scenario.NSteps))

for run in range(mean):
    i = 0
    while i < scenario.NSteps:
        j = min(scenario.nu, scenario.NSteps-i)
        maxG[run, i:(i+j)] = j*scenario.generate_reward(0, i, maxG = True) + maxG[run, i-1]
        i += j

mean_maxG = np.zeros(scenario.NSteps)
for i in range(scenario.NSteps):
    mean_maxG[i] = sum(maxG[:, i])/mean

for stratKey in strat.keys():
    avg_regret, BA, time = strat[stratKey](scenario, mean_maxG)
    fw.write_results(stratKey, avg_regret, BA, time)
fw.finish()