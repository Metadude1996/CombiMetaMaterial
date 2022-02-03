"Aleksi Bossart, 26.2.2019"
"Ryan van Mastrigt, 31.1.2022"
"semi-automated generation of arbitrary aperiodic lattices"


import numpy as np
from matplotlib import pyplot as plt

from mechadef import drawlattice
from mechadef import posadj
from mechadef import compatibility
from mechadef import compatibility_pbc
from mechadef import kernel
from mechadef import rank
from Generate_Configs import gen_config
from mechadef import twist
from mechadef import rotate_adj

"Calculate zero modes for pentodal configurations"

def get_modes(sub_mat, vtiles, htiles, displacement=0, draw_lattice=0, draw_modes=0, fix_y0=0, use_qr=0, pbc=0,
              fix_2edge=0, nofix=0):
    "sub_mat: unit cell design. 2D integer array sub_mat[x, y], where the integers correspond to the unit cell "
    "orientation."
    "we characterize the orientation by the location of the missing diagonal: {0, 1, 2, 3} = {LU, RU, RD, LD}"
    "where  LU = left up, RU = right up, RD = right down, LD = left down"
    "vtiles: additional tiling of the unit cell in the vertical direction"
    "htiles: additional tiling of  the unit cell in the horizontal direction"
    "displacement: amount of twist of the lattice"
    "draw_lattice: boolean. Draw (1) or don't draw (0)"
    "draw_modes: boolean. draw modes (1) or don't draw (0)"
    "fix_y0: boolean. Boundary condition: fixes position of all nodes on the line y=0"
    "use_qr: boolean. use rrQR instead of SVD"
    "pbc: boolean. use pbc (1) or not (0)"
    "fix_2dge: boolean. Bouundary condition: fix node on (x, y) = (0, 0) and (x, y) = (2, 0)"
    "no_fix: boolean. Boundary condition: no nodes are fixed (all trivial modes will be present)"

    "load the aperiodicity matrix"
    aperio = gen_config(sub_mat, vtiles, htiles)
    fake_aperio = rotate_adj(aperio) #Just for plotting

    "load the position file for the primitive cell"
    cellpos = np.genfromtxt('lattices/positions/primitivecell/lieb', delimiter=',')

    "compute posadj"
    positions, adjacency = posadj(aperio, cellpos)
    #fake_positions, fake_adjacency = posadj(fake_aperio, cellpos) #Just for plotting
    new_positions = twist(positions, displacement)
    #new_fake_positions = twist(fake_positions, displacement) #Just for plotting
    "compute compatibility matrix"
    if pbc:
        cmat, index_pbc = compatibility_pbc(new_positions, adjacency)
    else:
        cmat = compatibility(new_positions, adjacency)
    "all points on y=0 are fixed and thus removed from the compatibility matrix"
    if fix_y0==1:
        fix=0
        "compute indices of points on y=0"
        indices_y0=np.where(new_positions[:, 1]==0)[0]
        "sort from small to large so deletion and insertion goes correctly"
        indices_y0=np.sort(indices_y0)
        size_Nx0=np.size(indices_y0)
        "delete the columns corresponding to the fixed points from C"
        for i in range(size_Nx0):
            cmat=np.delete(cmat,2*indices_y0[i]-2*i,1)
            cmat=np.delete(cmat,2*indices_y0[i]-2*i,1)
        nullvec=kernel(cmat)
        "Insert 0-vectors in the place of the fixed points so the modes can be drawn"
        for i in range(size_Nx0):
            nullvec=np.insert(nullvec, 2*indices_y0[i], 0, axis=1)
            nullvec=np.insert(nullvec, 2*indices_y0[i], 0, axis=1)
        n1, n2 = np.shape(nullvec)
        modes = n1
        "The first two points are set as fixed as to remove translation + rotation modes"
    elif fix_2edge:
        fix = 0
        ind1 = np.argwhere(np.all(new_positions==np.array([0, 0]), axis=1))
        ind2 = np.argwhere(np.all(new_positions==np.array([2, 0]), axis=1))
        inds = np.append(ind1[0, :], ind2[0, :])
        indsx = 2*inds
        indsy = 2*inds + 1
        indsxy = np.sort(np.append(indsx, indsy))
        cmat = np.delete(cmat, indsxy, axis=1)
        nullvec = kernel(cmat)
        for i in range(np.shape(indsxy)[0]):
            nullvec = np.insert(nullvec, indsxy[i], 0, axis=1)
        # nullvec = np.insert(nullvec, indsxy, 0, axis=1)
        n1, n2 = np.shape(nullvec)
        modes = n1
    elif nofix:
        fix = 0
        if use_qr == 0:
            nullvec = kernel(cmat, numerr=1e-12)
            n1, n2 = np.shape(nullvec)
        else:
            n1 = rank(cmat)
        modes = n1
    else:
        if pbc:
            "four corners get mapped to (0,0), fix (0,0) and rot + trans modes are removed"
            fix = 2
        else:
            "fix two nodes to remove trans + rot modes"
            fix = 3
        if use_qr == 0:
            nullvec = kernel(cmat[:, fix:], numerr=1e-12)
            "the fixed points are removed from the C matrix"
            n1, n2 = np.shape(nullvec)
        else:
            n1 = rank(cmat[:, fix:])
            #n2 = np.shape(cmat)[0]
        modes = n1
    if use_qr == 0:
        zeromodes = np.zeros(n1 * (n2 + fix))
        zeromodes = zeromodes.reshape(n1, n2 + fix)
        zeromodes[:, fix:] += nullvec
        n2 = n2 + fix

        "find size of kernel of cmat, which is no. of modes (+3)"
        #ker = kernel(cmat)
        #modes = np.shape(ker)
        #modes = modes[0] - 3
    "draw the lattice if draw_lattice set to true (1)"
    if draw_lattice == 1:
        plt.rcParams["figure.figsize"] = [18, 10]
        drawlattice(new_positions, adjacency, rayon=0.08)
        plt.show()

    if draw_modes == 1:
        if pbc:
            "sort from small to large for pbc nodes"
            ind = np.argsort(index_pbc[:, 0])
            index_pbc = index_pbc[ind]
            "insert pbc nodes back in the floppy mode with the same deformations as their counterpart"
            for n in range(np.size(index_pbc, 0)):
                zeromodes = np.insert(zeromodes, int(2*index_pbc[n, 0]), zeromodes[:, int(2*index_pbc[n, 1])], axis=1)
                zeromodes = np.insert(zeromodes, int(2*index_pbc[n, 0]+1), zeromodes[:, int(2*index_pbc[n, 1]+1)],
                                      axis=1)
            n2 += int(2*np.size(index_pbc, 0))
        floppy = zeromodes.reshape((n1, int(n2 / 2), 2))
        "rotate out weird modes"
        # ind_uppercorner = np.argwhere(np.all(new_positions == np.array([2, 2]), axis=1))
        # x = floppy[0, ind_uppercorner[0, 0]]/ floppy[1, ind_uppercorner[0, 0]]
        # newfloppy = floppy[0] - x*floppy[1]
        # newfloppy = newfloppy.reshape(n2)
        # newfloppy2 = floppy[1].reshape(n2) - (np.dot(newfloppy, floppy[1].reshape(n2))/np.dot(newfloppy, newfloppy))
        #               * newfloppy
        # floppy[0], floppy[1] = newfloppy.reshape((int(n2 / 2), 2)), newfloppy2.reshape((int(n2 / 2), 2))
        delta = .1

        print("There are %d floppy modes" % n1)

        plt.rcParams["figure.figsize"] = [17, 9]

        for i in range(0, n1, 1):
            modpos = new_positions - delta * floppy[i]
            # print(floppy[i])
            drawlattice(new_positions, adjacency, colour='orange', name="axes%d" % i)
            drawlattice(modpos, adjacency, name="axes%d" % i)

            plt.show()
            plt.close()
    if use_qr == 0:
        return modes, zeromodes.reshape((n1, int(n2/2), 2))
    else:
        return modes

# modes, floppy = get_modes(np.array([[0, 1], [2, 3]]), 1, 1, draw_lattice=0, draw_modes=1, pbc=0, fix_2edge=0)
# print(floppy)

