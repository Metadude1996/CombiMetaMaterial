import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib
import itertools
import os
from scipy.optimize import curve_fit

def cm_to_inch(x):
    return x / 2.54

"generate 2d slices of configurations composed using the smiley face paper unit cells"
"in 2D Red & Green are the same, Blue is different"

"check for the rules"
def check_compatible(config):
    "for a config of size kx x ky"
    kx, ky = np.shape(config)
    'scan from left to right'
    "check if left and right neighbouring columns are the same or different"
    valid = 1
    for p in range(kx -1):
        if np.all(config[p, :] - config[p+1, :] == 0):
            continue
        elif np.sum(np.abs(config[p, :] - config[p+1, :])) == ky:
            continue
        else:
            valid = 0
            break
    # print(valid)
    return valid

"check all possible k = (2, 3, 4) configs for the rules"

def generate_all_configs_pdf(krange=[2, 3, 4], nopdf=False):
    for k in krange:
        # combinations = iter.product(range(4), repeat=k*k)
        combinations = np.random.randint(0, 4, size=(50, k * k))
        # combinations = np.fromiter(iter.chain(*combinations), int).reshape(-1, k*k)
        combis = itertools.product(range(2), repeat=k*k)
        configs = np.fromiter(itertools.chain(*combis), dtype=int).reshape(-1, k, k)
        compatible = np.zeros(np.shape(configs)[0], dtype=int)
        if not nopdf:
            with PdfPages(u'.\\smiley cube\\all_smiley_cube_configs_class_{:d}x{:d}.pdf'.format(k, k)) as pdf:
                for i in range(np.shape(configs)[0]):
                    compatible[i] = check_compatible(configs[i])
                    f, ax = plt.subplots(1, 1)
                    ax .imshow(configs[i], cmap='Greys', vmin=0, vmax=1)
                    plt.title('Class {:d}'.format(compatible[i]))
                    pdf.savefig(f)
                    plt.close()
            np.savez('.\\smiley cube\\smiley_cube_x_y_{:d}x{:d}.npz'.format(k, k), configs = configs, compatible = compatible)
            # print(compatible)
        else:
            for i in range(np.shape(configs)[0]):
                compatible[i] = check_compatible(configs[i])
            np.savez('.\\smiley cube\\smiley_cube_x_y_{:d}x{:d}.npz'.format(k, k), configs = configs, compatible = compatible)
    return configs, compatible

def generate_random_samples_class(kx, ky):
    configs = np.random.randint(low=0, high=2, size=(35000000, kx, ky), dtype=int)
    compatible = np.zeros(np.shape(configs)[0], dtype=int)
    for i in range(np.shape(configs)[0]):
        compatible[i] = check_compatible(configs[i])
    np.savez('.\\smiley cube\\smiley_cube_uniform_sample_x_y_{:d}x{:d}.npz'.format(kx, ky), x=configs, y=compatible)
    return configs, compatible

def generate_all_samples_class(kx, ky):
    combis = itertools.product(range(2), repeat=kx * ky)
    configs = np.fromiter(itertools.chain(*combis), dtype=int).reshape(-1, kx, ky)
    compatible = np.zeros(np.shape(configs)[0], dtype=int)
    for i in range(np.shape(configs)[0]):
        compatible[i] = check_compatible(configs[i])
    np.savez('.\\smiley cube\\smiley_cube_x_y_{:d}x{:d}.npz'.format(kx, ky), configs=configs, compatible=compatible)
    return configs, compatible

def class_distribution_random_samples(klist):
    C0_list = []
    C1_list = []
    for i, k in enumerate(klist):
        if k < 6:
            dataz = np.load('.\\smiley cube\\smiley_cube_x_y_{:d}x{:d}.npz'.format(k, k))
            x = dataz['configs']
            y = dataz['compatible']
        else:
            dataz = np.load('.\\smiley cube\\smiley_cube_uniform_sample_x_y_{:d}x{:d}.npz'.format(k, k))
            x = dataz['x']
            y = dataz['y']
        C0_list.append(np.sum(y == 0) / np.shape(y)[0])
        C1_list.append(np.sum(y == 1) / np.shape(y)[0])
    f, ax = plt.subplots()
    ax.plot(klist, C0_list, '.-', c='tab:blue', label='0')
    ax.plot(klist, C1_list, '.-', c='tab:pink', label='1')
    ax.set_xlabel('$k$')
    ax.set_ylabel('pdf')
    plt.legend(loc='center right')
    f.savefig('.\\smiley cube\\smiley_cube_class_distribution.pdf', facecolor=f.get_facecolor())
    f.savefig('.\\smiley cube\\smiley_cube_class_distribution.svg', facecolor=f.get_facecolor())
    f.savefig('.\\smiley cube\\smiley_cube_class_distribution.png', facecolor=f.get_facecolor(), dpi=400)
    plt.show()
    plt.close()
    return 0

def class_distribution_rectangular_configs(kxlist, kylist):

    plt.style.use(r'C:\\Users\\ryanv\\PycharmProjects\\Matplotlib styles\\paper-onehalf.mplstyle')
    cmap = matplotlib.cm.get_cmap("tab10")
    f = plt.figure(figsize=(cm_to_inch(8.6), cm_to_inch(8.6)), dpi=400)
    xlength = 3.32
    ylength = 0.3
    xoffset = 0.25 * (xlength / 8.6)
    yoffset = 0.2 * (xlength / 8.6)
    figfracx = (8.6 - 1. * xoffset * 8.6 - 0.3) / 8.6
    figfracy = figfracx
    # figfracy = 0.7
    ax = f.add_axes([xoffset, yoffset, figfracx, figfracy])
    for i, kx in enumerate(kxlist):
        C0_list = []
        C1_list = []
        for j, ky in enumerate(kylist):
            if os.path.exists('.\\smiley cube\\smiley_cube_x_y_{:d}x{:d}.npz'.format(kx, ky)):
                dat = np.load('.\\smiley cube\\smiley_cube_x_y_{:d}x{:d}.npz'.format(kx, ky))
                configs = dat['configs']
                labels = dat['compatible']
            elif os.path.exists('.\\smiley cube\\smiley_cube_uniform_sample_x_y_{:d}x{:d}.npz'.format(kx, ky)):
                dat = np.load('.\\smiley cube\\smiley_cube_uniform_sample_x_y_{:d}x{:d}.npz'.format(kx, ky))
                configs = dat['x']
                labels = dat['y']
            elif os.path.exists('.\\smiley cube\\smiley_cube_x_y_{:d}x{:d}.npz'.format(ky, kx)):
                dat = np.load('.\\smiley cube\\smiley_cube_x_y_{:d}x{:d}.npz'.format(ky, kx))
                configs = dat['configs']
                labels = dat['compatible']
            elif os.path.exists('.\\smiley cube\\smiley_cube_uniform_sample_x_y_{:d}x{:d}.npz'.format(ky, kx)):
                dat = np.load('.\\smiley cube\\smiley_cube_uniform_sample_x_y_{:d}x{:d}.npz'.format(ky, kx))
                configs = dat['x']
                labels = dat['y']
            elif np.power(2, kx*ky) < 35000000:
                configs, labels = generate_all_samples_class(kx, ky)
            else:
                configs, labels = generate_random_samples_class(kx, ky)
            C0_list.append(np.sum(labels == 0) / np.shape(labels)[0])
            C1_list.append(np.sum(labels == 1) / np.shape(labels)[0])

        # ax.plot(kylist, C0_list, '.-', c='tab:blue', label='0')
        ax.plot(kylist, C1_list, '.-', c=cmap.colors[i], label='$k_x={:d}$'.format(kx))
    ax.set_xlabel('$k_y$', labelpad=0)
    ax.set_ylabel('pdf', labelpad=0)
    ax.set_yscale('log')
    ax.set_xticks(kylist)
    # ax.set_title('$k_x = {:d}$'.format(kx))
    plt.legend(loc='bottom left')
    f.savefig('.\\smiley cube\\smiley_cube_kxlist_kylist_class_distribution.pdf', facecolor=f.get_facecolor())
    f.savefig('.\\smiley cube\\smiley_cube_kxlist_kylist_class_distribution.svg', facecolor=f.get_facecolor())
    f.savefig('.\\smiley cube\\smiley_cube_kxlist_kylist_class_distribution.png', facecolor=f.get_facecolor(), dpi=400)
    plt.show()
    plt.close()

def random_walk_testset(kx, ky, Nmoves):
    data = np.load('..\\MetaCombiNN\\data_smiley_cube_train_trainraw_test_{:d}x{:d}.npz'.format(kx, ky))
    y_test = data['y_test']
    x_test = data['x_test'][:, :, :, 0]
    ItoI = np.zeros(Nmoves, dtype=int)
    ItoC = np.zeros(Nmoves, dtype=int)
    CtoC = np.zeros(Nmoves, dtype=int)
    CtoI = np.zeros(Nmoves, dtype=int)
    if not os.path.exists(r'D:\\data\random_walks_Smiley_Cube_{:d}x{:d}\\'.format(kx, ky)):
        os.makedirs(r'D:\\data\random_walks_Smiley_Cube_{:d}x{:d}\\'.format(kx, ky))
    config_path = u'D:\\data\\random_walks_Smiley_Cube_{:d}x{:d}\\' \
                  u'configlist_test_{:d}.npy'
    classlist_path = u'D:\\data\\random_walks_Smiley_Cube_{:d}x{:d}\\classlist_test_{:d}.npy'
    for i, config in enumerate(x_test):
        if os.path.exists(config_path.format(kx, ky, i)) and os.path.exists(classlist_path.format(kx, ky, i)):
            # configlist = np.load(config_path.format(k, k, i))
            classlist = np.load(classlist_path.format(kx, ky, i))
            ind_toI = np.argwhere(classlist[1:]==0)
            ind_toC = np.argwhere(classlist[1:]==1)
            if classlist[0] == 0:
                ItoI[ind_toI[:, 0]] += 1
                ItoC[ind_toC[:, 0]] += 1
            else:
                CtoI[ind_toI[:, 0]] += 1
                CtoC[ind_toC[:, 0]] += 1
            print('loaded {:d}, skipped calculations'.format(i))
            continue

        configlist = np.zeros((Nmoves + 1, np.shape(config)[0], np.shape(config)[1]), dtype=int)
        classlist = np.zeros((Nmoves + 1), dtype=int)
        "calculate number of modes for n=3 and check line mode"
        classlist[0] = check_compatible(config)
        configlist[0] = config
        print("index: {:d} \t class: {:d} ".format(i, y_test[i]))
        if classlist[0] == 0:
            "class I"
            for moves in range(Nmoves):
                ix = np.random.randint(0, kx)
                iy = np.random.randint(0, ky)
                config[ix, iy] += 1
                config[ix, iy] %= 2
                classlist[moves+1] = check_compatible(config)
                print("step: {:d} \t class: {:d}".format(moves + 1, classlist[moves+1]))
                configlist[moves + 1] = config
                if classlist[moves+1] == 0:
                    ItoI[moves] += 1
                else:
                    ItoC[moves] += 1
        else:
            "class C"
            for moves in range(Nmoves):
                ix = np.random.randint(0, kx)
                iy = np.random.randint(0, ky)
                config[ix, iy] += 1
                config[ix, iy] %= 2
                classlist[moves + 1] = check_compatible(config)
                print("step: {:d} \t class: {:d}".format(moves + 1, classlist[moves + 1]))
                configlist[moves + 1] = config
                if classlist[moves + 1] == 0:
                    CtoI[moves] += 1
                else:
                    CtoC[moves] += 1
        if not os.path.exists(u'D:\\data\\random_walks_Smiley_Cube_{:d}x{:d}\\'.format(kx, ky)):
            os.mkdir(u'D:\\data\\random_walks_Smiley_Cube_{:d}x{:d}'.format(kx, ky))
        np.save(config_path.format(kx, ky, i), configlist)
        np.save(classlist_path.format(kx, ky, i), classlist)
        print('configlist and classlist')
    return CtoC, CtoI, ItoI, ItoC

def plot_probability_fit_smiley_cube(kxlist, kylist):
    plt.style.use(r'C:/Users/ryanv/PycharmProjects/Matplotlib styles/paper-onethird.mplstyle')
    # f, ax = plt.subplots(figsize=(0.8*(4/3)*27/8, 0.8*(27/8)))
    f, ax = plt.subplots(figsize=(cm_to_inch(6.5), cm_to_inch(4.5)))
    matplotlib.rcParams.update({'font.size': 8})
    cma = plt.get_cmap('tab20c')
    alphas = []
    betas = []
    for kx in kxlist:
        for i, ky in enumerate(kylist):

            Nsteps = kx * kx
            dat = np.load('.\\smiley cube\\probability_classchange_randomwalk_smiley_cube'
                          '_{:d}x{:d}_test.npz'.format(kx, ky))
            ItoI = dat['ItoI']
            ItoC = dat['ItoC']
            CtoI = dat['CtoI']
            CtoC = dat['CtoC']

            p_ItoC = dat['p_ItoC']
            p_CtoI = dat['p_CtoI']
            p_ItoI = dat['p_ItoI']
            p_CtoC = dat['p_CtoC']

            x = np.arange(1, Nsteps, 0.001)
            xyz = np.load('.\\smiley cube\\smiley_cube_x_y_{:d}x{:d}.npz'.format(kx, ky))
            res = xyz[xyz.files[1]]
            beta = np.sum(res == 1) / np.shape(res)[0]
            # beta = np.sum(res[:, 1] == 1) / np.shape(res)[0]
            def slope_fit(x, a):
                return np.power(a, x) + (1 - np.power(a, x - 1)) * (beta)
            # if k%2:
            #     'odd'
            #     p0 = np.array([(k-2.)/k])
            # else:
            #     'even'
            #     p0 = np.array([(k-1.)/k])
            slope_opt, slope_cov = curve_fit(slope_fit, np.arange(1, Nsteps+1), p_CtoC)
            alphas.append([slope_opt, slope_cov])
            betas.append(beta)
            # f2, ax2 = plt.subplots()
            # ax2.plot(np.arange(1, Nsteps+1), p_PtoP, '.-')
            # ax2.plot(np.linspace(1, Nsteps+1, 100), slope_fit(np.linspace(1, Nsteps+1, 100), *slope_opt))
            # plt.show()
            # plt.close()
            # p_analPtoP = (numerator/denumerator)**x + (1-(numerator/denumerator)**(x-1))*beta
            # ax.plot(x, p_analPtoP, c='b', label='MC')
            # ax.plot(np.arange(1, Nsteps+1), p_OtoP, c='r', label='O to P')
            if ky%2:
                ax.plot(np.linspace(1, Nsteps+1, 100), slope_fit(np.linspace(1, Nsteps+1, 100), *slope_opt),
                        c=cma.colors[int(i / 2)])
                ax.plot(np.arange(1, Nsteps+1), p_CtoC, '^', c=cma.colors[int(i / 2)], label=ky)
            else:
                ax.plot(np.linspace(1, Nsteps + 1, 100), slope_fit(np.linspace(1, Nsteps + 1, 100), *slope_opt),
                        c=cma.colors[4 + int(i / 2)])
                ax.plot(np.arange(1, Nsteps + 1), p_CtoC, 's', c=cma.colors[4 + int(i/2)], label=ky)
        ax.set_xlabel('$s$')
        ax.set_ylabel('$p(X_s = C | X_0 = C)$')
        ax.set_ylim([-0.05, 1.05])
        plt.legend(loc='best')
        ax.set_xscale('log')

        # ax.set_yscale('log')
        plt.tight_layout()
        plt.savefig('.\\smiley cube\\p_CtoC_kx{:d}_kylist_logx_fit.pdf'.format(kx),
                    facecolor=f.get_facecolor())
        plt.savefig('.\\smiley cube\\p_CtoC_kx{:d}_kylist_logx_fit.svg'.format(kx),
                    facecolor=f.get_facecolor())
        plt.savefig('.\\smiley cube\\p_CtoC_kx{:d}_kylist_logx_fit.png'.format(kx),
                    dpi=400, facecolor=f.get_facecolor())
        plt.show()
        plt.close()
        alphas = np.array(alphas)
        alphas = alphas[:, :, 0].astype(float)
        betas = np.array(betas)
        f, ax = plt.subplots(figsize=(0.4*(4/3)*27/8, 0.4*(27/8)))
        matplotlib.rcParams.update({'font.size': 8})
        ax.fill_between(kylist, 1-alphas[:, 0]+ np.sqrt(alphas[:, 1]), 1-alphas[:, 0] - np.sqrt(alphas[:, 1]),
                        color=cma.colors[0], alpha=0.2)
        # ax.fill_between(np.arange(3, 5)[1::2], 1-alphas[:, 0][1::2] + np.sqrt(alphas[:, 1][1::2]),
        #                 1-alphas[:, 0][1::2] - np.sqrt(alphas[:, 1][1::2]),
        #                 color=cma.colors[1], alpha=0.2)
        ax.plot(kylist, 1-alphas[:, 0], '^-', c=cma.colors[0], label=r'$\alpha$')
        # ax.plot(np.arange(3, 5)[1::2], 1-alphas[:, 0][1::2], 's-', c=cma.colors[0])
        ax.plot(kylist, betas, '^-', c=cma.colors[4], label=r'$\beta$')
        # ax.plot(np.arange(3, 5)[1::2], betas[1::2], 's-', c=cma.colors[4])
        ax.set_xlabel('$k_y$')
        # ax.add_ylabel()
        plt.legend()
        plt.tight_layout()
        plt.savefig('.\\smiley cube\\p_CtoC_kx{:d}_kylist_slopefit_asymptote.pdf'.format(kx),
                    facecolor=f.get_facecolor())
        plt.savefig('.\\smiley cube\\p_CtoC_kx{:d}_kylist_slopefit_asymptote.svg'.format(kx),
                    facecolor=f.get_facecolor())
        plt.savefig('.\\smiley cube\\p_CtoC_kx{:d}_kylist_slopefit_asymptote.png'.format(kx),
                    dpi=400, facecolor=f.get_facecolor())
        plt.show()
        plt.close()

        f, ax = plt.subplots(figsize=(cm_to_inch(4.5), cm_to_inch(3.5)))
        matplotlib.rcParams.update({'font.size': 8})
        ax.fill_between(kylist, 1 - alphas[:, 0] + np.sqrt(alphas[:, 1]),
                        1 - alphas[:, 0] - np.sqrt(alphas[:, 1]),
                        color=cma.colors[0], alpha=0.2)
        # ax.fill_between(np.arange(3, 5)[1::2], 1 - alphas[:, 0][1::2] + np.sqrt(alphas[:, 1][1::2]),
        #                 1 - alphas[:, 0][1::2] - np.sqrt(alphas[:, 1][1::2]),
        #                 color=cma.colors[1], alpha=0.2)
        ax.plot(kylist, 1 - alphas[:, 0], '^-', c=cma.colors[0], label=r'$\alpha$')
        # ax.plot(np.arange(3, 5)[1::2], 1 - alphas[:, 0][1::2], 's-', c=cma.colors[0])
        # ax.plot(np.arange(3, 9)[0::2], betas[0::2], '^-', c=cma.colors[4], label=r'$\beta$')
        # ax.plot(np.arange(3, 9)[1::2], betas[1::2], 's-', c=cma.colors[4])
        ax.set_xlabel('$k$')
        ax.set_ylabel(r'$\alpha$')
        # ax.set_xticks([3, 4], minor=False)
        # ax.set_xticks([4, 6, 8], minor=True)
        # ax.set_yticks([0.25, 0.5], minor=False)
        # ax.set_yticks([0.125, 0.375, 0.625], minor=True)
        # plt.legend()
        plt.tight_layout()
        plt.savefig('.\\smiley cube\\p_CtoC_kx{:d}_kylist_slopefit.pdf'.format(kx),
                    facecolor=f.get_facecolor())
        plt.savefig('.\\smiley cube\\p_CtoC_kx{:d}_kylist_slopefit.svg'.format(kx),
                    facecolor=f.get_facecolor())
        plt.savefig('.\\smiley cube\\p_CtoC_kx{:d}_kylist_slopefit.png'.format(kx),
                    dpi=400, facecolor=f.get_facecolor())
        plt.show()
        plt.close()
    return 0


# for k in range(6, 9):
#     generate_random_samples_class(k)
# generate_all_configs_pdf([5], nopdf=True)
# class_distribution_random_samples(np.array([2, 3, 4, 5, 6, 7, 8]))
# for k in range(3, 7):
#     class_distribution_rectangular_configs(np.array([k]), np.arange(1, 7))

# kx=5
# for ky in range(3, 6):
#     random_walk_testset(kx, ky, Nmoves=kx*kx)
#     CtoC, CtoI, ItoI, ItoC = random_walk_testset(kx, ky, kx*kx)
#     p_ItoI = np.divide(ItoI, np.add(ItoI, ItoC))
#     p_ItoC = 1 - p_ItoI
#     p_CtoC = np.divide(CtoC, np.add(CtoC, CtoI))
#     p_CtoI = 1 - p_CtoC
#     np.savez('.\\smiley cube\\probability_classchange_randomwalk_Smiley_Cube_{:d}x{:d}'
#              '_test.npz'.format(kx, ky), CtoC=CtoC,
#              CtoI=CtoI, ItoI=ItoI, ItoC=ItoC, p_CtoC=p_CtoC, p_CtoI=p_CtoI, p_ItoI=p_ItoI, p_ItoC=p_ItoC)

# plot_probability_fit_smiley_cube(np.array([5]), np.array([3, 4, 5]))
class_distribution_rectangular_configs(np.array([2, 3, 4, 5, 6]), np.array([2, 3, 4, 5, 6]))

