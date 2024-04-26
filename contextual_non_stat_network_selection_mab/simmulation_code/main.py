__author__ = 'lekesen'


import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

# Import Libraries
import numpy as np

# Import algorithms
from algorithms.egreedy import egreedy
from algorithms.UCB1 import UCB1
from algorithms.LinearContextualUCB1 import LinearContextualUCB1
from algorithms.QuadraticContextualUCB1 import QuadraticContextualUCB1
from algorithms.CubicContextualUCB1 import CubicContextualUCB1
from algorithms.DiscountedLinearContextualUCB1 import DiscountedLinearContextualUCB1
from algorithms.DiscountedCubicContextualUCB1 import DiscountedCubicContextualUCB1
from algorithms.RLSLinearContextualUCB1 import RLSLinearContextualUCB1
from algorithms.RegularizedRLSLinearContextualUCB1 import RegularizedRLSLinearContextualUCB1
from algorithms.RegularizedRLSCubicContextualUCB1 import RegularizedRLSCubicContextualUCB1
from algorithms.QRDRLSLinearContextualUCB1 import QRDRLSLinearContextualUCB1
from algorithms.QRDRLSQuadraticContextualUCB1 import QRDRLSQuadraticContextualUCB1
from algorithms.QRDRLSCubicContextualUCB1 import QRDRLSCubicContextualUCB1
from algorithms.SlidingWindowUCB1 import SlidingWindowUCB1
from algorithms.ThompsonSampling import ThompsonSampling
from algorithms.LinearContextualThompsonSampling import LinearContextualThompsonSampling
from algorithms.QuadraticContextualThompsonSampling import QuadraticContextualThompsonSampling
from algorithms.CubicContextualThompsonSampling import CubicContextualThompsonSampling
from algorithms.ThompsonSampling2 import ThompsonSampling2
from algorithms.DeepThompsonSampling import DeepThompsonSampling
from algorithms.DeepThompsonSampling2 import DeepThompsonSampling2
from algorithms.DeepThompsonSampling3 import DeepThompsonSampling3
from algorithms.VariableDeepThompsonSampling import VariableDeepThompsonSampling
from algorithms.VariableDeepThompsonSampling2 import VariableDeepThompsonSampling2
from algorithms.VariableDeepThompsonSampling3 import VariableDeepThompsonSampling3

# Import scenarios
from Scenarios.Scenario1 import Scenario1
from Scenarios.Scenario2 import Scenario2
from Scenarios.Scenario3 import Scenario3
from Scenarios.Scenario4 import Scenario4
from Scenarios.ScenarioTest import ScenarioTest
from Scenarios.ScenarioLinearContext import ScenarioLinearContext
from Scenarios.ScenarioPeriodLinearContext import ScenarioPeriodLinearContext
from Scenarios.ScenarioPolynomialContext import ScenarioPolynomialContext
from Scenarios.Scenario2QoSContext import Scenario2QoSContext
from Scenarios.Scenario2Hard import Scenario2Hard
from Scenarios.Scenario3Contextual import Scenario3Contextual

from Scenarios.ScenarioNonStatContext import ScenarioNonStatContext


# Import debugging and other useful libraries/classes
from FileWriter import FileWriter


# Strategies that we want to run
strat = {
    #'e-greedy': egreedy,
    'UCB1': UCB1,
    'LinearContextualUCB1': LinearContextualUCB1,
    'QuadraticContextualUCB1': QuadraticContextualUCB1,
    'CubicContextualUCB1': CubicContextualUCB1,
    #'DiscountedLinearContextualUCB1': DiscountedLinearContextualUCB1,
    #'DiscountedCubicContextualUCB1': DiscountedCubicContextualUCB1,
    #'RLSLinearContextualUCB1': RLSLinearContextualUCB1,
    #'RegularizedRLSLinearContextualUCB1': RegularizedRLSLinearContextualUCB1,
    #'RegularizedRLSCubicContextualUCB1': RegularizedRLSCubicContextualUCB1,
    'QRDRLSLinearContextualUCB1': QRDRLSLinearContextualUCB1,
    'QRDRLSQuadraticContextualUCB1': QRDRLSQuadraticContextualUCB1,
    #'QRDRLSCubicContextualUCB1': QRDRLSCubicContextualUCB1,
    #'SlidingWindowUCB1': SlidingWindowUCB1,
    #'ThompsonSampling': ThompsonSampling,
    #'LinearContextualThompsonSampling': LinearContextualThompsonSampling,
    #'QuadraticContextualThompsonSampling': QuadraticContextualThompsonSampling,
    #'CubicContextualThompsonSampling': CubicContextualThompsonSampling,
    #'ThompsonSampling2': ThompsonSampling2,
    #'DeepThompsonSampling': DeepThompsonSampling,
    #'DeepThompsonSampling2': DeepThompsonSampling2,
    #'DeepThompsonSampling3': DeepThompsonSampling3,
    #'VariableDeepThompsonSampling': VariableDeepThompsonSampling,
    #'VariableDeepThompsonSampling2': VariableDeepThompsonSampling2,
    #'VariableDeepThompsonSampling3': VariableDeepThompsonSampling3,
}

# Use desired scenario from Scenarios folder
scenario = Scenario3Contextual()
# Create documents where to store obtained results
fw = FileWriter(scenario)
fw.write_settings(strat.keys())

# Calculate the optimal G (gain) of the scenario
mean = 50 # mean number, as this is higher the more accurate it is
maxG = np.zeros((mean, scenario.NSteps))

for run in range(mean):
    i = 0
    while i < scenario.NSteps:
        j = min(scenario.nu, scenario.NSteps-i)

        g = 0
        for step in range(j):
            w = scenario.generate_reward(0, i+step, maxG = True)
            g += w
        maxG[run, i:(i+j)] = g + maxG[run, i-1]
        i += j

mean_maxG = np.zeros(scenario.NSteps)
for i in range(scenario.NSteps):
    mean_maxG[i] = sum(maxG[:, i])/mean


# Calculate the average regret, best-action % and time for each scenario and algorithm.
for stratKey in strat.keys():
    avg_regret, BA, time = strat[stratKey](scenario, mean_maxG)
    fw.write_results(stratKey, avg_regret, BA, time)
    print("")
fw.finish()