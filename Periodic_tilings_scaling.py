"Ryan van Mastrigt, 31.1.2022"
"This script calculates the number of modes of k x k unit cells"
import numpy as np
import time
from matplotlib import pyplot as plt
import itertools as iter
from matplotlib import cm as cm

"import necessary functions:"
from autopenta import get_modes

"Use the same seed for reproducable results"
np.random.seed(0)

def det_modes(config, Nx, Ny, qr=0, twist=0, pbc=0):
    # input_configuration = np.reshape(config, (1, Nx*Ny))
    number_of_modes = get_modes(config, 1, 1, displacement=twist, use_qr=qr, pbc=pbc)
    return number_of_modes

def main():
    "set starting parameters"
    t0=time.time()
    k = 7
    # combinations = iter.product(range(4), repeat=k*k)
    combinations = np.random.randint(0, 4, size=(50, k*k))
    # combinations = np.fromiter(iter.chain(*combinations), int).reshape(-1, k*k)
    modes = np.zeros(4)
    modes_x = np.zeros(4)
    modes_y = np.zeros(4)
    #linear = np.zeros(np.size(combinations,0))
    file = open('data_new_rrQR_i_n_Mx_My_n4_{:d}x{:d}.txt'.format(k, k), 'wb+')
    # results_old = np.loadtxt('results//modescaling//data_i_n_M_{:d}x{:d}.txt'.format(k, k), delimiter=',')
    for i in range(0, np.size(combinations, 0)):
        block = np.resize(combinations[i, :].copy(), (k, k))
        for tiling in range(1, 5, 1):
            # lattice = np.tile(block, (tiling, tiling))
            lattice_x = np.tile(block, (tiling, 1))
            lattice_y = np.tile(block, (1, tiling))

            modes_x[tiling-1] = det_modes(lattice_x, tiling*k, k, qr=1, pbc=0)
            modes_y[tiling-1] = det_modes(lattice_y, k, tiling*k, qr=1, pbc=0)
            # modes[tiling - 1] = det_modes(lattice, tiling * k, tiling * k, qr=1, pbc=0)

        #if (modes[1]>modes[0]):
        #    linear[i] = 1
        #elif (modes[1]==modes[0]):
        #    linear[i] = 0
        #else:
        #    linear[i] = 2
        #    print('decreasing scaling for i={:d}'.format(i))
        #np.savetxt(file, (i, linear[i]), delimiter=',')
        # thing = [np.r_[i, combinations[i], modes]]
        thing = [np.r_[i, combinations[i], modes_x, modes_y]]
        np.savetxt(file, thing, delimiter=',')
    file.close()
    t1 = time.time()
    dt = t1 - t0
    print(dt)
    # scaling_count = np.zeros(3)
    # for i in range(3):
    #     scaling_count[i] = np.count_nonzero(linear == i)
    # pscaling = scaling_count/np.size(combinations, 0)
    # objects = ('constant', 'increasing', 'decreasing')
    # file = open('results/modescaling/p_scaling_lin_vs_const_{:d}x{:d}.dat'.format(k, k), 'wb')
    # np.savetxt(file, (np.array((0, 1, 2)), pscaling), delimiter=',')
    # file.close()
    # plt.figure()
    # y_pos = np.arange(len(objects))
    # plt.bar(y_pos, pscaling, align='center', alpha=0.5)
    # plt.xticks(y_pos, objects)
    # plt.ylabel('Normal distribution')
    # plt.savefig('results/modescaling/p_scaling_lin_vs_const_hist_{:d}x{:d}.pdf'.format(k, k))
    # plt.show()
    # plt.close()



if __name__ == '__main__':
        main()