import numpy as np
from autopenta import get_modes
import time

numberrep_config_0 = np.random.randint(0, 4, (3, 3))
numberrep_config_1 = np.random.randint(0, 4, (4, 4))
numberrep_config_2 = np.random.randint(0, 4, (5, 5))
numberrep_config_3 = np.random.randint(0, 4, (6, 6))
numberrep_config_4 = np.random.randint(0, 4, (7, 7))
numberrep_config_5 = np.random.randint(0, 4, (8, 8))
# numberrep_config_3 = np.random.randint(0, 4, (6, 6))
numberrep_configs = [numberrep_config_0, numberrep_config_1, numberrep_config_2, numberrep_config_3,
                     numberrep_config_4, numberrep_config_5]
nmodes = []
times = []

for i in range(len(numberrep_configs)):
    start_time = time.time()
    for n in range(2, 5):
        nmodes.append(get_modes(numberrep_configs[i], n, n, use_qr=1))
    times.append(time.time() - start_time)

print(times)

