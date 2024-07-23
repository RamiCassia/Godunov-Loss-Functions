import sys
import os
base_path = os.getcwd() + '/'
sys.path.append(base_path)
import matplotlib.pyplot as plt
import numpy as np
from jaxfluids import InputReader, Initializer, SimulationManager
from jaxfluids.post_process import load_data, create_contourplot

''' 

The following maps the value of variable CONFIG to the intended configuration:

3   :  4S (1)

'''

CONFIG = 3

input_reader = InputReader(base_path + 'data/JSON/configuration_' + str(CONFIG) + ".json", base_path + "data/JSON/numerical_setup.json")
initializer  = Initializer(input_reader)
sim_manager  = SimulationManager(input_reader)
buffer_dictionary = initializer.initialization()
sim_manager.simulate(buffer_dictionary)

quantities = ["density", "velocity", "pressure"]
_, _, _, data_dict = load_data(base_path + 'data/WENO_output/' + str(CONFIG) + '/domain/', quantities)

rho = data_dict['density']
rho = np.swapaxes(rho, 1,2)
rho = np.squeeze(rho)
rho = np.expand_dims(rho,1)

vel = data_dict['velocity']
u = vel[:,0]
v = vel[:,1]
u = np.swapaxes(u, 1,2)
u = np.squeeze(u)
u = np.expand_dims(u,1)
v = np.swapaxes(v, 1,2)
v = np.squeeze(v)
v = np.expand_dims(v,1)

p = data_dict['pressure']
p = np.swapaxes(p, 1,2)
p = np.squeeze(p)
p = np.expand_dims(p,1)
Ep = (p/(rho*(1.4-1)))

field = np.concatenate((rho,u,v,Ep), 1)

np.save(base_path + 'data/WENO_2x4x256x256_' + str(CONFIG) + '.npy', field)