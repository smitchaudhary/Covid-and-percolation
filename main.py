import numpy as np
import matplotlib.pyplot as plt
from library import *
from config import *
from scipy import stats
L = L

vac_rate = [0, 1/10000, 1/3000, 1/500]

num_experiments = 1

mobility_sweep = [0.001]

infected_population = []
infected_population_err = []
lethality = []
lethality_err = []
for index, rate in enumerate(vac_rate):
    np.random.seed(0)
    for mobility_factor in mobility_sweep:
        print(f'Mobility factor = {mobility_factor}')
        inf_pop_avg = []
        deaths_avg = []
        leth_avg = []
        for i in range(num_experiments):
            sim = Simulation(L= L, mobility_factor = mobility_factor, vac_rate = rate )
            sim.run_simulation(visualization = False)
            name = f'{index}'
            sim.epidemic_curves(plot_bool=False, name = name, rate = rate)

plt.axvline(x=vaccine_dev_time, color = 'y', linestyle = '-', label = 'Vaccination starts')
plt.axhline(y=100*sim.healthcare_threshold/sim.population, color='r', linestyle='-', label = 'Healthcare threshold')
plt.axhline(y=100*lockdown_threshold, color='g', linestyle='-', label = 'Lockdown imposing threshold')
plt.axhline(y=100*lifting_lockdown_threshold, color='b', linestyle='-', label = 'Lockdown lifting threshold')
plt.title('Percentage of population infected for different vaccination rates')
plt.xlabel('Days')
plt.ylabel('Percentage of population infected')
plt.legend()
plt.show()
