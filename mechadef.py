"Aleksi Bossart, 20.2.2019"
"Ryan van Mastrigt, 31.01.2022"
"definitions of functions used in the other scripts"

import numpy as np
from scipy import linalg as splinalg
from matplotlib import pyplot as plt
import math
import sympy as sym


def drawlattice(pos, adj, rayon = 0.08, colour = 'b', nodecol = 'r', name = "axes"):

	"draws a 2D lattice based on a list of nodes and a (upper-triangular!) adjacency matrix"

	plt.axes(label = name)

	for i, row in enumerate(adj):

		for j, link in enumerate(row):

			if link:
				
				line = plt.Line2D((pos[i,0], pos[j,0]), (pos[i,1], pos[j,1]), lw=2.5, color = colour)
				plt.gca().add_line(line)

	for x in pos:

		circle = plt.Circle(x, radius = rayon, fc = nodecol)
		plt.gca().add_patch(circle)

	plt.axis('scaled')
	
	return;


def compatibility(pos, adj):
	
	"computes the compatibility matrix (relating node displacements to bond strains) from the adjacency matrix and the node positions"
	"the columns correspond to dx0 dy0 ... dxN dyN, while the lines correspond to eb0 ... ebM"

	N, d = np.shape(pos)
	M = int(np.sum(np.sum(adj)))
	
	cmat = np.zeros(2*N*M)
	cmat = cmat.reshape((M, 2*N))

	k = 0

	for i, row in enumerate(adj):

		for j, link in enumerate(row):

			if link:

				"update cmat to account for the bond elongation relation under consideration"
				"no need for a global direction convention, just match the bond direction and the displacement vector"
				
				"add the four contributions to this bond strain in the compatibility matrix"
				cmat[k, 2*i] = (pos[i,0] - pos[j,0])
				cmat[k, 2*i+1] = (pos[i,1] - pos[j,1])
				cmat[k, 2*j] = -(pos[i,0] - pos[j,0])
				cmat[k, 2*j+1] = -(pos[i,1] - pos[j,1])

				k+=1

	return cmat;


def compatibility_pbc(pos, adj):
	"computes the compatibility matrix (relating node displacements to bond strains) from the adjacency matrix and the"
	"node positions with periodic boundary conditions"
	"the columns correspond to dx0 dy0 ... dxN dyN, while the lines correspond to eb0 ... ebM"

	N, d = np.shape(pos)
	M = int(np.sum(np.sum(adj)))
	Nx = np.amax(pos[:, 0])
	Ny = np.amax(pos[:, 1])

	cmat = np.zeros(2 * N * M)
	cmat = cmat.reshape((M, 2 * N))

	k = 0

	"create a list keeping track of the indices of doubly counted nodes (so the ones that get identified with others bc pbc)"
	index_pbc = np.zeros((M, 2))
	count = 0

	for i, row in enumerate(adj):

		for j, link in enumerate(row):

			if link:
				"update cmat to account for the bond elongation relation under consideration"
				"no need for a global direction convention, just match the bond direction and the displacement vector"

				"add the four contributions to this bond strain in the compatibility matrix"
				i_pbc = i
				j_pbc = j
				if pos[i, 0] == Nx:
					if pos[i, 1] == Ny:
						i_pbc = np.where((pos[:, 0] == 0) & (pos[:, 1] == 0))
						i_pbc = i_pbc[0]

					else:
						i_pbc = np.where((pos[:, 0] == 0) & (pos[:, 1] == pos[i, 1]))
						i_pbc = i_pbc[0]
					index_pbc[count, 0] = i
					index_pbc[count, 1] = i_pbc
					count += 1
				if pos[i, 1] == Ny:
					if pos[i, 0] == Nx:
						i_pbc = np.where((pos[:, 0] == 0) & (pos[:, 1] == 0))
						i_pbc = i_pbc[0]
					else:
						i_pbc = np.where((pos[:, 0] == pos[i, 0]) & (pos[:, 1] == 0))
						i_pbc = i_pbc[0]
					index_pbc[count, 0] = i
					index_pbc[count, 1] = i_pbc
					count += 1
				if pos[j, 0] == Nx:
					if pos[j, 1] == Ny:
						j_pbc = np.where((pos[:, 0] == 0) & (pos[:, 1] == 0))
						j_pbc = j_pbc[0]
					else:
						j_pbc = np.where((pos[:, 0] == 0) & (pos[:, 1] == pos[j, 1]))
						j_pbc = j_pbc[0]
					index_pbc[count, 0] = j
					index_pbc[count, 1] = j_pbc
					count += 1
				if pos[j, 1] == Ny:
					if pos[j, 0] == Nx:
						j_pbc = np.where((pos[:, 0] == 0) & (pos[:, 1] == 0))
						j_pbc = j_pbc[0]
					else:
						j_pbc = np.where((pos[:, 0] == pos[j, 0]) & (pos[:, 1] == 0))
						j_pbc = j_pbc[0]
					index_pbc[count, 0] = j
					index_pbc[count, 1] = j_pbc
					count += 1
				i_pbc = int(i_pbc)
				j_pbc = int(j_pbc)
				cmat[k, 2 * i_pbc] = (pos[i, 0] - pos[j, 0])
				cmat[k, 2 * i_pbc + 1] = (pos[i, 1] - pos[j, 1])
				cmat[k, 2 * j_pbc] = -(pos[i, 0] - pos[j, 0])
				cmat[k, 2 * j_pbc + 1] = -(pos[i, 1] - pos[j, 1])

				k += 1
	"remove redundant zero columns"
	cmat = np.delete(cmat, np.where(~cmat.any(axis=0))[0], axis=1)
	"remove extra 0's and duplicates index_pbc"
	index_pbc = index_pbc[:count]
	index_pbc = np.unique(index_pbc, axis=0)
	return cmat, index_pbc


def kernel(mat, numerr = 1e-12):
	
	"compute the kernel of a matrix using singular value decomposition"

	#u, s, vh = np.linalg.svd(mat)
	u, s, vh = splinalg.svd(mat)
	matdims = np.shape(mat)
	sdim = np.shape(s)
	sdim = sdim[0]

	z = np.zeros(matdims[1])
	z[:sdim] += s

	iskernel = (z < numerr)
	
	return np.compress(iskernel, vh, axis = 0);

def rank(mat, numerr = 1e-12):
	"compute the rank of a matrix using RRQR factorization"
	R, P = splinalg.qr(mat, mode='r', pivoting=True)
	s = np.diagonal(R)
	rank_bool = (np.abs(s) > numerr)
	return np.size(mat, 1) - np.sum(rank_bool)


def gramaway(basis = np.array([[2,0,0],[0,2,0],[0,0,2]]), vector = np.array([1,1,1])):

	"project away the contribution of vector to basis"
	"useful to remove known modes, such as counter-rotations"
	"assuming the basisvectors are the rows"

	newbas = basis - ( np.diag( (basis @ vector) / np.sum(vector * vector) ) @ np.tile(vector, (np.shape(basis)[0],1)) )

	return newbas;


def hourglass(pos, target):
	
	"given a set of target points, use the 4 closest adjacent points to compute the deformation of an hourglass shape"

	"obtain the four adjacent points"

	n1, n2 = np.shape(target)
	vicinity = np.zeros(shape = (n1, 4))

	for i, center in enumerate(target):

		dist = np.linalg.norm(pos - center, axis = 1)
		dist[np.argmin(dist)] += 1000000000

		vicinity[i, 0] = np.argmin(dist)
		dist[np.argmin(dist)] += 1000000000

		vicinity[i, 1] = np.argmin(dist)
		dist[np.argmin(dist)] += 1000000000

		vicinity[i, 2] = np.argmin(dist)
		dist[np.argmin(dist)] += 1000000000

		vicinity[i, 3] = np.argmin(dist)


	"Comput and sort the four angles, then take the difference of the two extrema"

	rawangle = 0 * vicinity

	for i, neighbours in enumerate(vicinity):

		for j, label in enumerate(neighbours):
			
			x = pos[int(label), 0] - target[i, 0]
			y = pos[int(label), 1] - target[i, 1]
			rawangle[i, j] = np.angle(x + 1j*y)

	rawangle = np.sort(rawangle)
	angle = np.sort((rawangle - np.roll(rawangle, 1, axis = 1)) % np.pi)

	deformation = angle[:,-1] - angle[:,0]

	return deformation;


def colorcode(points, defo, rayon = 0.5, nodecol = 'r', name = "axes"):

	"colors points according to some code"
	"used to locate the hourglass modes, initially"

	plt.axes(label = name)

	opa = defo / np.max(defo)

	for i, x in enumerate(points):

		circle = plt.Circle(x, radius = rayon, fc = nodecol, alpha = opa[i])
		plt.gca().add_patch(circle)

	plt.axis('scaled')
	
	return;


def rotatebasis(basis, angle):

	"reshuffle two floppy modes for visual insights"

	rot = np.array( [[np.cos(angle),-np.sin(angle)], [np.sin(angle), np.cos(angle)]] )
	rotbas = rot @ basis

	return rotbas;


def drawstress(pos, adj, selfstress, name = "axes"):

	"draw a state of self-stress"

	colour = {-1:'blue', 0:'white', 1:'red'}
	tension = np.sign((np.abs(selfstress)>1e-15)*selfstress)

	plt.axes(label = name)

	k = 0

	for i, row in enumerate(adj):

		for j, link in enumerate(row):

			if link:
				
				line = plt.Line2D((pos[i,0], pos[j,0]), (pos[i,1], pos[j,1]), lw=2.5, color = colour[tension[k]])
				plt.gca().add_line(line)
				k += 1

	plt.axis('scaled')
	
	return;


def posadj(aperio, cellpos):

	bulkdens = 3
	bounddens = 2
	n, m = np.shape(aperio)

	posdim = bulkdens * n * m + bounddens * (n + m) + 1

	positions = np.zeros(shape = (posdim, 2), dtype=float)

	for i, row in enumerate(aperio):

		for j, orient in enumerate(row):

			vec = bounddens * np.array([i,j])

			positions[bulkdens * (m * i + j), :] = cellpos[0] + vec
			positions[bulkdens * (m * i + j) + 1, :] = cellpos[1] + vec
			positions[bulkdens * (m * i + j) + 2, :] = cellpos[3] + vec

	"add the boundary points at the end of the file"

	positions[bulkdens * n * m : bulkdens * n * m + bounddens * n, 0] = np.arange(0, bounddens * n)
	positions[bulkdens * n * m : bulkdens * n * m + bounddens * n, 1] = bounddens * m * np.ones(bounddens * n)
	positions[bulkdens * n * m + bounddens * n : bulkdens * n * m + bounddens * (n + m), 0] = bounddens * n * np.ones(bounddens * m)
	positions[bulkdens * n * m + bounddens * n : bulkdens * n * m + bounddens * (n + m), 1] = np.arange(0, bounddens * m)
	positions[bulkdens * n * m + bounddens * (n + m), :] = np.array([bounddens * n, bounddens * m])


	"save the resulting positions file"

	np.savetxt('lattices/positions/gen/pentagone', positions, fmt='%d', delimiter=',')


	"initialize the full adjacency matrix to appropriate dimensions"

	adjacency = np.zeros(shape = (posdim, posdim))


	"update the adjacency matrix and save it"

	for i, row in enumerate(aperio):

		for j, orient in enumerate(row):

			adjacency[bulkdens * (m * i + j), bulkdens * (m * i + j) + 1] = 1
			adjacency[bulkdens * (m * i + j), bulkdens * (m * i + j) + 2] = 1

			if (j + 1) % m == 0:
				adjacency[bulkdens * (m * i + j) + 1, bulkdens * n * m + bounddens * i] = 1
	
			else:
				adjacency[bulkdens * (m * i + j) + 1, bulkdens * (m * i + j + 1)] = 1

			if (i + 1) % n == 0:
				adjacency[bulkdens * (m * i + j) + 2, bulkdens * n * m + bounddens * n + bounddens * j] = 1

			else:
				adjacency[bulkdens * (m * i + j) + 2, bulkdens * (m * (i + 1) + j)] = 1
	
			if orient == 8:
				continue

			if orient != 0:
			
				if (j + 1) % m == 0:
			
					adjacency[bulkdens * (m * i + j) + 1, bulkdens * n * m + bounddens * i + 1] = 1

				else:

					adjacency[bulkdens * (m * i + j) + 1, bulkdens * (m * i + j + 1) + 2] = 1

			if orient != 1:
			
				if (j + 1) % m == 0 and (i + 1) % n == 0:
			
					adjacency[bulkdens * n * m + bounddens * i + 1, bulkdens * n * m + bounddens * n + bounddens * j + 1] = 1

				elif (j + 1) % m == 0 and (i + 1) % n != 0:

					adjacency[bulkdens * n * m + bounddens * i + 1, bulkdens * (m * (i + 1) + j) + 1] = 1

				elif (j + 1) % m != 0 and (i + 1) % n == 0:

					adjacency[bulkdens * (m * i + j + 1) + 2, bulkdens * n * m + bounddens * n + bounddens * j + 1] = 1
	
				else:
	
					adjacency[bulkdens * (m * i + j + 1) + 2, bulkdens * (m * (i + 1) + j) + 1] = 1

			if orient != 2:
			
				if (i + 1) % n == 0:
			
					adjacency[bulkdens * (m * i + j) + 2, bulkdens * n * m + bounddens * n + bounddens * j + 1] = 1

				else:

					adjacency[bulkdens * (m * i + j) + 2, bulkdens * (m * (i + 1) + j) + 1] = 1
	
			if orient != 3:
				
				adjacency[bulkdens * (m * i + j) + 1, bulkdens * (m * i + j) + 2] = 1


	for i in np.arange(bounddens * n - 1):

		adjacency[bulkdens * n * m + i, bulkdens * n * m + i + 1] = 1

	adjacency[bulkdens * n * m + bounddens * n - 1, bulkdens * n * m + bounddens * (n + m)] = 1


	for i in np.arange(bounddens * m - 1):

		adjacency[bulkdens * n * m + bounddens * n + i, bulkdens * n * m + bounddens * n + i + 1] = 1

	adjacency[bulkdens * n * m + bounddens * n + bounddens * m - 1, bulkdens * n * m + bounddens * (n + m)] = 1


	np.savetxt('lattices/adjacency/gen/pentagone', adjacency, fmt='%d', delimiter=',')

	return positions, adjacency;

def rotate_adj(pos):
    pos = np.flipud(pos)
    pos = np.transpose(pos)
    return pos;

def twist(pos, disp):
    dim1, dim2 = np.shape(pos)
    for i in range(dim1):
        tempx = pos[i,0]
        tempy = pos[i,1]
        if tempx%2==0 and tempy%2!=0: pos[i,0] = pos[i,0] - disp*math.sin((math.pi*(tempx+tempy)/2))
        if tempx%2!=0 and tempy%2==0: pos[i,1] = pos[i,1] + disp*math.sin((math.pi*(tempx+tempy))/2)
    return pos;
