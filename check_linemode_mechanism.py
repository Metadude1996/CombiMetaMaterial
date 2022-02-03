"Ryan van Mastrigt, 31.1.2022"
"check the line mode rules"

import numpy as np
import os
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib import pyplot as plt
from autopenta import get_modes
import itertools as iter
import pickle
import matplotlib.backends.backend_pdf

def cm_to_inch(x):
    return x / 2.54

def det_modes(config, Nx, Ny, qr=0, twist=0, pbc=0):
    # input_configuration = np.reshape(config, (1, Nx*Ny))
    number_of_modes = get_modes(config, 1, 1, displacement=twist, use_qr=qr, pbc=pbc)
    return number_of_modes

def Pixel_transform(lat):
    Nx, Ny = np.shape(lat)
    Mrep = np.zeros((2*Nx, 2*Ny), dtype=int)
    for x in range(Nx):
        for y in range(Ny):
            uc = lat[x, y]
            if uc == 0:
                temp = np.array([[0, 1], [0, 0]])
                Mrep[2*x:2*x+2, 2*y:2*y+2] = temp
            elif uc == 1:
                temp = np.array([[0, 0], [0, 1]])
                Mrep[2*x:2*x+2, 2*y:2*y+2] = temp
            elif uc == 2:
                temp = np.array([[0, 0], [1, 0]])
                Mrep[2*x:2*x+2, 2*y:2*y+2] = temp
            elif uc == 3:
                temp = np.array([[1, 0], [0, 0]])
                Mrep[2*x:2*x+2, 2*y:2*y+2] = temp
            else:
                print('invalid uc')
                return -1
    return Mrep

def Pixel_rep(data, kx, ky):
    dataMr = np.zeros((np.size(data, 0), kx*2, ky*2), dtype=int)
    for i, lat in enumerate(data):
        config = np.reshape(lat, (kx, ky))
        dataMr[i] = Pixel_transform(config)
    return dataMr

def connectmatrix_hor(plaquette):
    "inputs: plaquette: 2x4 matrix of 0's and 1's"
    "sum middle plaquette"
    summidplaq = np.sum(plaquette[:, 1:3])
    if summidplaq==1 or summidplaq==3:
        return -1
    "top & bottom most row"
    top=0
    bot=0
    "middle plaquette connections"
    up = 0
    down = 0
    left=0
    right=0
    diagtop=0
    diagbot=0
    block=0
    "check top connection"
    if plaquette[0, 0]==1:
        if plaquette[1, 0]==1:
            top = 1
        else:
            return -1
    "check bot connection"
    if plaquette[0, -1]==1:
        if plaquette[1, -1]==1:
            bot =1
        else:
            return -1
    if summidplaq==2:
        "check middle connections"
        if plaquette[0, 1]==1:
            if plaquette[0, 2]==1:
                left=1
            elif plaquette[1, 1]==1:
                up=1
            elif plaquette[1, 2]==1:
                diagtop=1
        elif plaquette[0, 2]==1:
            if plaquette[1, 2]==1:
                down=1
            elif plaquette[1, 1]==1:
                diagbot=1
        elif plaquette[1, 1] == 1:
            if plaquette[1, 2]==1:
                right=1
    elif summidplaq==4:
        "middle plaquette has block of 4 black squares"
        block=1
    "check for connection matrix to return, o=0, x=1"
    trueconnect = 0
    connectmat = np.zeros((2, 2), dtype=int)
    if top or up:
        connectmat[0, 0] = 1
        connectmat[1, 0] = 1
        trueconnect=1
    if bot or down:
        connectmat[0, 1] = 1
        connectmat[1, 1] = 1
        trueconnect=1
    if down:
        connectmat[0, 1] = 1
        connectmat[1, 1] = 1
    if left:
        connectmat[0, 0] = 1
        connectmat[0, 1] = 1
    if right:
        connectmat[1, 0] = 1
        connectmat[1, 1] = 1
    if diagbot:
        connectmat[0, 1] = 1
        connectmat[1, 0] = 1
    if diagtop:
        connectmat[0, 0] = 1
        connectmat[1, 1] = 1
    if block:
        connectmat[0, 0] = 1
        connectmat[0, 1] = 1
        connectmat[1, 0] = 1
        connectmat[1, 1] = 1
        trueconnect=1
    if summidplaq==2:
        trueconnect=1
    if trueconnect:
        return connectmat
    else:
        if np.sum(plaquette)==0:
            return connectmat
        else:
            return -1
    return -1

def check_uniqs_counts_line(line):
    "insert: line[k, linewidth]"
    k = np.shape(line)[0]
    lwidth = np.shape(line)[1]
    unique, counts = np.unique(line, return_counts=True)
    if np.all(unique != 0):
        "no zeros in the line: all unit cells in line connected. Check if all connected within line"
        if np.all(counts%2==0):
            "all connected to others in the line: linemode!"
            "check no connections in horizontal direction (no true line mode)"
            if lwidth==k:
                for i in range(k):
                    if np.sum(line[:, 0]==line[i, -1])==1:
                        "connection in horizontal direction: no lm"
                        return 0
            return 1
    return 0

def check_prodmat_ver(ProdMat, k):
    modelist = []
    linemodes=0
    ProdMat = np.tile(ProdMat.copy(), (1, 1, 2))
    for rows in range(1, k):
        for lines in range(k):
            line = ProdMat[:, :, lines:lines+rows].copy()
            "remove duplicates"
            dellist = []
            for i in range(np.shape(line)[0]):
                arg = np.argwhere(np.all(line[i] == line[:], axis=(1, 2)))
                for j in range(np.shape(arg)[0]):
                    if arg[j, 0] != i and arg[j, 0] > i:
                        dellist.append(arg[j, 0])
            if dellist:
                line = np.delete(line, dellist, axis=0)
            for ul in range(np.shape(line)[0]):
                linmode = check_uniqs_counts_line(line[ul])
                if linmode:
                    "could be two smaller ones: check"
                    smalllinmode_1row = 0
                    smalllinmode_2row = 0
                    smalllinmode_3row = 0
                    if rows>1:
                        for smalllines in range(rows):
                            smallline = line[ul, :, smalllines:smalllines+1]
                            smalllinmode_1row += check_uniqs_counts_line(smallline)
                    if rows>2:
                        for smalllines in range(rows-1):
                            smallline = line[ul, :, smalllines:smalllines+2]
                            smalllinmode_2row += check_uniqs_counts_line(smallline)
                    if rows>3:
                        for smalllines in range(rows-2):
                            smallline = line[ul, :, smalllines:smalllines+3]
                            smalllinmode_3row += check_uniqs_counts_line(smallline)
                    if smalllinmode_1row==rows:
                        "mode is actually just bidomain rows"
                        linmode=0
                    if smalllinmode_2row==1 and smalllinmode_1row==1 and rows==3:
                        "mode is actually one linemode and one bidomain row"
                        linmode=0
                    if smalllinmode_2row==2 and rows==4:
                        "mode is actually two linemodes"
                        linmode=0
                    if smalllinmode_2row==1 and smalllinmode_1row==2 and rows==4:
                        "mode is actually one linemode and 2 bidomain rows"
                        linmode=0
                "check if not found earlier (some linemodes are counted double)"
                if modelist:
                    if [rows, lines] in modelist:
                        "mode found before"
                        linmode=0
                linemodes += linmode
                if linmode:
                    modelist.append([rows, lines])
    return linemodes

def check_type_connections(line):
    "input line[k, width]"
    k = np.shape(line)[0]
    lwidth = np.shape(line)[1]
    line_straightdiag = np.zeros((lwidth-1, 2))
    for l in range(lwidth-1):
        "look in [k, 2] line for diagonal or straight connections to other k-line"
        small = line[:, l:l+2]
        for x in range(k):
            "go over all unitcells in the first k-line"
            if small[x, 0] == small[x, 1]:
                "connected to x-neighbour -> straight connection"
                line_straightdiag[l, 0] += 1
            elif small[x, 0] == small[(x+1) % k, 1] or small[x, 0] == small[(x-1) % k, 1]:
                "connected to next nearest neighbour -> diag connection"
                line_straightdiag[l, 1] += 1
    "check if number of straight and diagonal connections per 2 neighbouring k-lines meet the requirements"
    "requirements: 2 or more connections of same type (diag or straight) in (width-1) pairs of neighbouring k-lines"
    # count = 0
    # for l in range(lwidth-1):
    #     if np.sum(line_straightdiag[l, 0])>=2 or np.sum(line_straightdiag[l, 1])>=2:
    #         count+=1
    # if lwidth%2:
    #     "uneven linemode width"
    #     if count >= lwidth-1:
    #         return 1
    # else:
    #     if count >= 1:
    #         return 1

    "alternative condition: in adjacent rows only vertical or diagonal connections allowed"
    count = 0
    for l in range(lwidth-1):
        if line_straightdiag[l, 0]>0 and line_straightdiag[l, 1]>0:
            count += 1
    if count == 0:
        return 1
    return 0

def check_prodmat_ver_easy(ProdMat, k):
    modelist = []
    linemodes=0
    ProdMat = np.tile(ProdMat.copy(), (1, 2))
    for rows in range(1, k+1):
        for lines in range(k):
            line = ProdMat[:, lines:lines+rows].copy()

            linmode = check_uniqs_counts_line(line)
            if linmode:
                "could be two smaller ones: check"
                smalllinmode_1row = 0
                smalllinmode_2row = 0
                smalllinmode_3row = 0
                if rows>1:
                    for smalllines in range(rows):
                        smallline = line[:, smalllines:smalllines+1]
                        smalllinmode_1row += check_uniqs_counts_line(smallline)
                if rows>2:
                    for smalllines in range(rows-1):
                        smallline = line[:, smalllines:smalllines+2]
                        smalllinmode_2row += check_uniqs_counts_line(smallline)
                if rows>3:
                    for smalllines in range(rows-2):
                        smallline = line[:, smalllines:smalllines+3]
                        smalllinmode_3row += check_uniqs_counts_line(smallline)
                if smalllinmode_1row==rows:
                    "mode is actually just bidomain rows"
                    linmode=0
                if smalllinmode_2row==1 and smalllinmode_1row==1 and rows==3:
                    "mode is actually one linemode and one bidomain row"
                    linmode=0
                if smalllinmode_2row==2 and rows==4:
                    "mode is actually two linemodes"
                    linmode=0
                if smalllinmode_2row==1 and smalllinmode_1row==2 and rows==4:
                    "mode is actually one linemode and 2 bidomain rows"
                    linmode=0
                if smalllinmode_3row==1 and smalllinmode_1row==1 and rows==4:
                    "mode is actually one 3-wide linemode and 1 bidomain row"
                    linmode=0
            "check if not found earlier (some linemodes are counted double)"
            if modelist:
                if [rows, lines] in modelist:
                    "mode found before"
                    linmode=0
            if linmode:
                if rows>1:
                    linmode = check_type_connections(line)
            linemodes += linmode
            if linmode:
                modelist.append([rows, lines])
            # if rows==k:
            #     "no need to go to all the lines"
            #     break
    return linemodes, modelist

def check_prodmat_ver_easy_kxky(ProdMat, kx, ky):
    modelist = []
    linemodes=0
    ProdMat = np.tile(ProdMat.copy(), (1, 2))
    for rows in range(1, ky+1):
        for lines in range(ky):
            line = ProdMat[:, lines:lines+rows].copy()

            linmode = check_uniqs_counts_line(line)
            if linmode:
                "could be two smaller ones: check"
                smalllinmode_1row = 0
                smalllinmode_2row = 0
                smalllinmode_3row = 0
                if rows>1:
                    for smalllines in range(rows):
                        smallline = line[:, smalllines:smalllines+1]
                        smalllinmode_1row += check_uniqs_counts_line(smallline)
                if rows>2:
                    for smalllines in range(rows-1):
                        smallline = line[:, smalllines:smalllines+2]
                        smalllinmode_2row += check_uniqs_counts_line(smallline)
                if rows>3:
                    for smalllines in range(rows-2):
                        smallline = line[:, smalllines:smalllines+3]
                        smalllinmode_3row += check_uniqs_counts_line(smallline)
                if smalllinmode_1row==rows:
                    "mode is actually just bidomain rows"
                    linmode=0
                if smalllinmode_2row==1 and smalllinmode_1row==1 and rows==3:
                    "mode is actually one linemode and one bidomain row"
                    linmode=0
                if smalllinmode_2row==2 and rows==4:
                    "mode is actually two linemodes"
                    linmode=0
                if smalllinmode_2row==1 and smalllinmode_1row==2 and rows==4:
                    "mode is actually one linemode and 2 bidomain rows"
                    linmode=0
                if smalllinmode_3row==1 and smalllinmode_1row==1 and rows==4:
                    "mode is actually one 3-wide linemode and 1 bidomain row"
                    linmode=0
            "check if not found earlier (some linemodes are counted double)"
            if modelist:
                if [rows, lines] in modelist:
                    "mode found before"
                    linmode=0
            if linmode:
                if rows>1:
                    linmode = check_type_connections(line)
            linemodes += linmode
            if linmode:
                modelist.append([rows, lines])
            # if rows==k:
            #     "no need to go to all the lines"
            #     break
    return linemodes, modelist

def check_horizontal_connections(config, k):
    "inputs: config[4*kx, 4*ky], already tiled (2,2). assume square config, i.e. k=kx=ky"
    "initiate neighbour list"
    # NeighbourList = np.full([k, k, 3, 2], -1, dtype=int)
    # Nneighbours = np.full((k, k), 0, dtype=int)
    # "fill in neighbourlist"
    # for x in range(k):
    #     for y in range(k):
    #         # neighbourcount=0
    #         "find which plaquette uc has the black square"
    #         # arguc = np.argwhere(config[x:x+2, y:y+2]==1)
    #         # xplaq = (2*(x+arguc[0, 0])-1)%2*k
    #         # yplaq = (2*(x+arguc[0, 1])-1)%2*k
    #         # plaq = config[xplaq:xplaq+2, yplaq:yplaq+2]
    #         "go over all plaquettes"
    #         plaq = config[2*x+1:2*x+3, 2*y+1:2*y+3]
    #         if np.sum(plaq)>1:
    #             blacksquares = np.argwhere(plaq==1)
    #             for i in range(np.shape(blacksquares)[0]):
    #                 for j in range(i+1, np.shape(blacksquares)[0]):
    #                     N1x = (x+blacksquares[i, 0])%k
    #                     N1y = (y+blacksquares[i, 1])%k
    #                     N2x = (x+blacksquares[j, 0])%k
    #                     N2y = (y+blacksquares[j, 1])%k
    #                     nind1=0
    #                     nind2=0
    #                     while np.all(NeighbourList[N1x, N1y, nind1]!=-1):
    #                         nind1 += 1
    #                     while np.all(NeighbourList[N2x, N2y, nind2]!=-1):
    #                         nind2 += 1
    #                     NeighbourList[N1x, N1y, nind1] = np.array([N2x, N2y])
    #                     NeighbourList[N2x, N2y, nind2] = np.array([N1x, N1y])
    #                     Nneighbours[N1x, N1y] += 1
    #                     Nneighbours[N2x, N2y] += 1
    # "got neighbourlist[x1, y1, 3, (x2,y2)]"
    # "use neighbourlist to build up pair-matrix"
    # # print('whack')
    # uniq = 0
    # # Nmatrices = int(np.prod(Nneighbours[Nneighbours != 0]))
    # Nmatrices = np.power(2, int(np.sum(Nneighbours[Nneighbours != 0]-1)))
    # ProdMat = np.zeros((Nmatrices, k, k))
    # matcount = 1
    # for x in range(k):
    #     for y in range(k):
    #         for n in range(Nneighbours[x, y]):
    #             neighbour = NeighbourList[x, y, n]
    #             "these two sits have never been linked"
    #             if np.all(ProdMat[:, x, y] == 0) and np.all(ProdMat[:, neighbour[0], neighbour[1]]==0):
    #                 "give unique value and fill in uniq for (x, y) and the neighbour "
    #                 uniq+=1
    #                 for mat in range(matcount):
    #                     ProdMat[mat, x, y] = uniq
    #                     ProdMat[mat, neighbour[0], neighbour[1]] = uniq
    #             else:
    #                 "these two sites have been linked before (not necessarily to each other), check the prodmat if (x, y) and neighbour link has been made before"
    #                 ProdMatxy = ProdMat[:, x, y]
    #                 ProdMatn = ProdMat[:, neighbour[0], neighbour[1]]
    #                 cond1 = ProdMatxy == ProdMatn
    #                 cond2 = ProdMatxy != 0
    #                 ind = np.argwhere(np.logical_and(cond1, cond2))
    #                 if np.shape(ind)[0]==0:
    #                     "link has not been made before"
    #                     "copy all previously filled in prodmat and fill in the new uniqs"
    #                     uniq += 1
    #                     for mat in range(matcount, matcount+matcount):
    #                         ProdMat[mat] = ProdMat[mat-matcount].copy()
    #                         "remove previous connections to (x, y) and (nx, ny)"
    #                         prevxy = np.argwhere(ProdMat[mat, :, :]==ProdMat[mat, x, y])
    #                         prevn = np.argwhere(ProdMat[mat, :, :] == ProdMat[mat, neighbour[0], neighbour[1]])
    #                         for prevind in range(np.shape(prevxy)[0]):
    #                             ProdMat[mat, prevxy[prevind, 0], prevxy[prevind, 1]] = 0
    #                         for prevind in range(np.shape(prevn)[0]):
    #                             ProdMat[mat, prevn[prevind, 0], prevn[prevind, 1]] = 0
    #                         ProdMat[mat, x, y] = uniq
    #                         ProdMat[mat, neighbour[0], neighbour[1]] = uniq
    #                     matcount += matcount
    # "remove zeros and duplicates"
    # indzero = np.argwhere(np.all(ProdMat[:, :, :]==0, axis=(1, 2)))
    # ProdMat = np.delete(ProdMat, indzero[:, 0], axis=0)
    # dellist = []
    # for i in range(np.shape(ProdMat)[0]):
    #     arg = np.argwhere(np.all(ProdMat[i] == ProdMat[:], axis=(1, 2)))
    #     for j in range(np.shape(arg)[0]):
    #         if arg[j, 0] != i and arg[j, 0] > i:
    #             dellist.append(arg[j, 0])
    # if dellist:
    #     ProdMat = np.delete(ProdMat, dellist, axis=0)
    # print(ProdMat)
    "alternative ProdMat calculation"
    uniq = 0
    ProdMat = np.zeros((k, k))
    for x in range(k):
        for y in range(k):
            plaq = config[2*x+1: 2*x+3, 2*y+1: 2*y+3]
            if np.sum(plaq)>1:
                uniq += 1
                blackinds = np.argwhere(plaq==1)
                for i in range(np.shape(blackinds)[0]):
                    xind = (x + blackinds[i, 0])%k
                    yind = (y + blackinds[i, 1])%k
                    ProdMat[xind, yind] = uniq
    "check for linemodes"
    linemodeshor=0
    linemodesver=0
    "small to big number of rows"
    linemodeshor, modelisthor = check_prodmat_ver_easy(ProdMat, k)
    # print(ProdMat)
    ProdMatRot = np.rot90(ProdMat, axes=(0, 1))
    # print(ProdMatRot)
    linemodesver, modelistver = check_prodmat_ver_easy(ProdMatRot, k)
    linemodes = linemodeshor + linemodesver
    return linemodes, modelisthor, modelistver

def check_horizontal_connections_kxky(config, kx, ky):
    "inputs: config[4*kx, 4*ky], already tiled (2,2). assume square config, i.e. k=kx=ky"
    "initiate neighbour list"
    # NeighbourList = np.full([k, k, 3, 2], -1, dtype=int)
    # Nneighbours = np.full((k, k), 0, dtype=int)
    # "fill in neighbourlist"
    # for x in range(k):
    #     for y in range(k):
    #         # neighbourcount=0
    #         "find which plaquette uc has the black square"
    #         # arguc = np.argwhere(config[x:x+2, y:y+2]==1)
    #         # xplaq = (2*(x+arguc[0, 0])-1)%2*k
    #         # yplaq = (2*(x+arguc[0, 1])-1)%2*k
    #         # plaq = config[xplaq:xplaq+2, yplaq:yplaq+2]
    #         "go over all plaquettes"
    #         plaq = config[2*x+1:2*x+3, 2*y+1:2*y+3]
    #         if np.sum(plaq)>1:
    #             blacksquares = np.argwhere(plaq==1)
    #             for i in range(np.shape(blacksquares)[0]):
    #                 for j in range(i+1, np.shape(blacksquares)[0]):
    #                     N1x = (x+blacksquares[i, 0])%k
    #                     N1y = (y+blacksquares[i, 1])%k
    #                     N2x = (x+blacksquares[j, 0])%k
    #                     N2y = (y+blacksquares[j, 1])%k
    #                     nind1=0
    #                     nind2=0
    #                     while np.all(NeighbourList[N1x, N1y, nind1]!=-1):
    #                         nind1 += 1
    #                     while np.all(NeighbourList[N2x, N2y, nind2]!=-1):
    #                         nind2 += 1
    #                     NeighbourList[N1x, N1y, nind1] = np.array([N2x, N2y])
    #                     NeighbourList[N2x, N2y, nind2] = np.array([N1x, N1y])
    #                     Nneighbours[N1x, N1y] += 1
    #                     Nneighbours[N2x, N2y] += 1
    # "got neighbourlist[x1, y1, 3, (x2,y2)]"
    # "use neighbourlist to build up pair-matrix"
    # # print('whack')
    # uniq = 0
    # # Nmatrices = int(np.prod(Nneighbours[Nneighbours != 0]))
    # Nmatrices = np.power(2, int(np.sum(Nneighbours[Nneighbours != 0]-1)))
    # ProdMat = np.zeros((Nmatrices, k, k))
    # matcount = 1
    # for x in range(k):
    #     for y in range(k):
    #         for n in range(Nneighbours[x, y]):
    #             neighbour = NeighbourList[x, y, n]
    #             "these two sits have never been linked"
    #             if np.all(ProdMat[:, x, y] == 0) and np.all(ProdMat[:, neighbour[0], neighbour[1]]==0):
    #                 "give unique value and fill in uniq for (x, y) and the neighbour "
    #                 uniq+=1
    #                 for mat in range(matcount):
    #                     ProdMat[mat, x, y] = uniq
    #                     ProdMat[mat, neighbour[0], neighbour[1]] = uniq
    #             else:
    #                 "these two sites have been linked before (not necessarily to each other), check the prodmat if (x, y) and neighbour link has been made before"
    #                 ProdMatxy = ProdMat[:, x, y]
    #                 ProdMatn = ProdMat[:, neighbour[0], neighbour[1]]
    #                 cond1 = ProdMatxy == ProdMatn
    #                 cond2 = ProdMatxy != 0
    #                 ind = np.argwhere(np.logical_and(cond1, cond2))
    #                 if np.shape(ind)[0]==0:
    #                     "link has not been made before"
    #                     "copy all previously filled in prodmat and fill in the new uniqs"
    #                     uniq += 1
    #                     for mat in range(matcount, matcount+matcount):
    #                         ProdMat[mat] = ProdMat[mat-matcount].copy()
    #                         "remove previous connections to (x, y) and (nx, ny)"
    #                         prevxy = np.argwhere(ProdMat[mat, :, :]==ProdMat[mat, x, y])
    #                         prevn = np.argwhere(ProdMat[mat, :, :] == ProdMat[mat, neighbour[0], neighbour[1]])
    #                         for prevind in range(np.shape(prevxy)[0]):
    #                             ProdMat[mat, prevxy[prevind, 0], prevxy[prevind, 1]] = 0
    #                         for prevind in range(np.shape(prevn)[0]):
    #                             ProdMat[mat, prevn[prevind, 0], prevn[prevind, 1]] = 0
    #                         ProdMat[mat, x, y] = uniq
    #                         ProdMat[mat, neighbour[0], neighbour[1]] = uniq
    #                     matcount += matcount
    # "remove zeros and duplicates"
    # indzero = np.argwhere(np.all(ProdMat[:, :, :]==0, axis=(1, 2)))
    # ProdMat = np.delete(ProdMat, indzero[:, 0], axis=0)
    # dellist = []
    # for i in range(np.shape(ProdMat)[0]):
    #     arg = np.argwhere(np.all(ProdMat[i] == ProdMat[:], axis=(1, 2)))
    #     for j in range(np.shape(arg)[0]):
    #         if arg[j, 0] != i and arg[j, 0] > i:
    #             dellist.append(arg[j, 0])
    # if dellist:
    #     ProdMat = np.delete(ProdMat, dellist, axis=0)
    # print(ProdMat)
    "alternative ProdMat calculation"
    uniq = 0
    ProdMat = np.zeros((kx, ky))
    for x in range(kx):
        for y in range(ky):
            plaq = config[2*x+1: 2*x+3, 2*y+1: 2*y+3]
            if np.sum(plaq)>1:
                uniq += 1
                blackinds = np.argwhere(plaq==1)
                for i in range(np.shape(blackinds)[0]):
                    xind = (x + blackinds[i, 0])%kx
                    yind = (y + blackinds[i, 1])%ky
                    ProdMat[xind, yind] = uniq
    "check for linemodes"
    linemodeshor=0
    linemodesver=0
    "small to big number of rows"
    linemodeshor, modelisthor = check_prodmat_ver_easy_kxky(ProdMat, kx, ky)
    # print(ProdMat)
    # ProdMatRot = np.rot90(ProdMat, axes=(0, 1))
    # # print(ProdMatRot)
    # linemodesver, modelistver = check_prodmat_ver_easy_kxky(ProdMatRot, ky, kx)
    # linemodes = linemodeshor + linemodesver
    return linemodeshor, modelisthor



def check_horizontal_linemode(config, line, k):
    "inputs: config (2*k, 2*k) matrix of 0's and 1's, line (1) integer \in {0, 1, 2}"
    verindex = line*2
    # testconfig1 = np.array([[0, 1, 1, 0], [0, 0, 0, 0]])
    # testconfig2 = np.array([[0, 0, 0, 0], [0, 1, 1, 0]])
    # testconfig3 = np.array([[0, 0, 1, 0], [0, 0, 1, 0]])
    # testconfig4 = np.array([[0, 1, 0, 0], [0, 1, 0, 0]])
    # testconfig5 = np.array([[0, 0, 0, 0], [0, 0, 0, 0]])
    # testconfig6 = np.array([[0, 1, 0, 0], [0, 0, 1, 0]])
    # testconfig7 = np.array([[0, 0, 1, 0], [0, 1, 0, 0]])
    # testconfig8 = np.array([[0, 1, 0, 1], [0, 1, 0, 1]])
    # testconfig9 = np.array([[1, 0, 1, 0], [1, 0, 1, 0]])
    # testconfig10 = np.array([[0, 1, 1, 0], [0, 1, 1, 0]])
    # testconfig11 = np.array([[1, 0, 0, 1], [1, 0, 0, 1]])
    # testconfigs = (testconfig1, testconfig2, testconfig3, testconfig4, testconfig5, testconfig6, testconfig7, testconfig8, testconfig9, testconfig10, testconfig11)
    # for i, testconfig in enumerate(testconfigs):
    #     connectmat = connectmatrix_hor(testconfig)

    connectmats = np.zeros((k, 2, 2), dtype=int)
    for horplaq in range(k):
        horindex = (1+horplaq*2)
        connectmat = connectmatrix_hor(config[horindex:horindex+2, verindex:(verindex+4)])
        if np.size(connectmat)==1:
            return 0
        else:
            connectmats[horplaq] = connectmat
        if horplaq>0:
            "check compatibility to previous plaquette"
            if connectmats[horplaq, 0, 0] + connectmats[horplaq-1, 1, 0] ==1 & connectmats[horplaq, 0, 1] + connectmats[horplaq-1, 1, 1] ==1:
                "compatible"
            else:
                return 0
    "check periodic compatibility"
    if connectmats[0, 0, 0] + connectmats[-1, 1, 0] ==1 & connectmats[0, 0, 1] + connectmats[-1, 1, 1] == 1:
        "compatible"
        topdom = 0
        botdom = 0
        for plaqs in range(k):
            if np.sum(connectmats[plaqs, 0:2, 0])==2:
                botdom += 1
            if np.sum(connectmats[plaqs, 0:2, 1])==2:
                topdom += 1
        if botdom==np.ceil(k/2) and topdom==np.ceil(k/2):
            "two bi-domains, to avoid counting double set lm to 0"
            return 0
        return 1

    return 0

def check_horizontal_linemode_3x2(config, line, kx):
    "inputs: config (6, 4) matrix of 0's and 1's, line (1) integer \in {0, 1, 2}"
    verindex = line*2
    # testconfig1 = np.array([[0, 1, 1, 0], [0, 0, 0, 0]])
    # testconfig2 = np.array([[0, 0, 0, 0], [0, 1, 1, 0]])
    # testconfig3 = np.array([[0, 0, 1, 0], [0, 0, 1, 0]])
    # testconfig4 = np.array([[0, 1, 0, 0], [0, 1, 0, 0]])
    # testconfig5 = np.array([[0, 0, 0, 0], [0, 0, 0, 0]])
    # testconfig6 = np.array([[0, 1, 0, 0], [0, 0, 1, 0]])
    # testconfig7 = np.array([[0, 0, 1, 0], [0, 1, 0, 0]])
    # testconfig8 = np.array([[0, 1, 0, 1], [0, 1, 0, 1]])
    # testconfig9 = np.array([[1, 0, 1, 0], [1, 0, 1, 0]])
    # testconfig10 = np.array([[0, 1, 1, 0], [0, 1, 1, 0]])
    # testconfig11 = np.array([[1, 0, 0, 1], [1, 0, 0, 1]])
    # testconfigs = (testconfig1, testconfig2, testconfig3, testconfig4, testconfig5, testconfig6, testconfig7, testconfig8, testconfig9, testconfig10, testconfig11)
    # for i, testconfig in enumerate(testconfigs):
    #     connectmat = connectmatrix_hor(testconfig)

    connectmats = np.zeros((kx, 2, 2), dtype=int)
    for horplaq in range(kx):
        horindex = (1+horplaq*2)
        connectmat = connectmatrix_hor(config[horindex:horindex+2, verindex:(verindex+4)])
        if np.size(connectmat)==1:
            return 0
        else:
            connectmats[horplaq] = connectmat
        if horplaq>0:
            "check compatibility to previous plaquette"
            if connectmats[horplaq, 0, 0] + connectmats[horplaq-1, 1, 0] ==1 & connectmats[horplaq, 0, 1] + connectmats[horplaq-1, 1, 1] ==1:
                "compatible"
            else:
                return 0
    "check periodic compatibility"
    if connectmats[0, 0, 0] + connectmats[-1, 1, 0] ==1 & connectmats[0, 0, 1] + connectmats[-1, 1, 1] == 1:
        "compatible"
        return 1

    return 0

def check_horizontal_bidomain(config, line, k):
    "inputs: config (2*k, 2*k) matrix of 0's and 1's, line (1) integer \in {0, 1, 2}"
    verindex = line * 2
    # config = np.tile(config, (2, 1))
    connectedges = np.zeros(k, dtype=int)
    for horplaq in range(k):
        horindex = (1 + horplaq * 2)
        edgeplaq = config[horindex:horindex + 2, verindex:(verindex + 2)]
        if np.sum(edgeplaq[0:2, 0]) == 2 or np.sum(edgeplaq[0:2, 1])==2:
            connectedges[horplaq] = 1
    if np.sum(connectedges)==2:
        return 1
    return 0

def check_all_configs():
    "load configs + labels"
    k = 4
    datapath = os.path.dirname('C:\\Users\\ryanv\\PycharmProjects\\metacombi\\results\\modescaling\\')
    respath = os.path.dirname('C:\\Users\\ryanv\\PycharmProjects\\metacombi\\results\\modescaling\\')

    raw_data = np.loadtxt(datapath + '\\PixelRep_{:d}x{:d}.txt'.format(k, k), delimiter=',')
    raw_results = np.loadtxt(respath + '\\results_analysis_new_rrQR_i_Scen_slope_offset_M1k_{:d}x{:d}.txt'.format(k, k),
                             delimiter=',')
    y_total = raw_results[:, 1].astype(int)
    x_total = raw_data.reshape(-1, 2 * k, 2 * k)
    indB = np.argwhere(y_total==1)
    indA = np.argwhere(y_total==0)
    "check for horizontal & vertical linemodes"
    ind_lines_slope = np.zeros((np.shape(indB)[0], 5), dtype=int)
    for i, ind in enumerate(indB[:, 0]):
        config = x_total[ind]
        config = np.tile(config, (2, 2))
        configrot = np.tile(np.rot90(x_total[ind]), (2, 2))
        linemode=0
        bidomain=0
        for lines in range(k):
            linemode += check_horizontal_linemode(config, lines, k)
            linemode += check_horizontal_linemode(configrot, lines, k)
            bidomain += check_horizontal_bidomain(config, lines, k)
            bidomain += check_horizontal_bidomain(configrot, lines, k)
        print('i: {:d} \t lm: {:d} \t bidom: {:d} \t sum: {:d} \t slope: {:d} \n'.format(i, linemode, bidomain, linemode+bidomain, raw_results[ind, 2].astype(int)))
        ind_lines_slope[i] = np.array([ind, linemode, bidomain, linemode+bidomain, raw_results[ind, 2].astype(int)])
    np.save(r'.//results//modescaling//npdata_linecountB_ind_lines_slope.npy', ind_lines_slope)
    print('check linemode + bidomain = slope B')
    print(np.all(ind_lines_slope[:, 3] == raw_results[:, 2].astype(int)))
    ind_lines_slope = np.zeros((np.shape(indA)[0], 5), dtype=int)
    for i, ind in enumerate(indA[:, 0]):
        config = x_total[ind]
        config = np.tile(config, (2, 2))
        configrot = np.tile(np.rot90(x_total[ind]), (2, 2))
        linemode=0
        bidomain=0
        for lines in range(3):
            linemode += check_horizontal_linemode(config, lines, k)
            linemode += check_horizontal_linemode(configrot, lines, k)
            bidomain += check_horizontal_bidomain(config, lines, k)
            bidomain += check_horizontal_bidomain(configrot, lines, k)
        print('i: {:d} \t lm: {:d} \t bidom: {:d} \t sum: {:d} \t  slope: {:d} \n'.format(i, linemode, bidomain, linemode+bidomain, raw_results[ind, 2].astype(int)))
        ind_lines_slope[i] = np.array([ind, linemode, bidomain, linemode+bidomain, raw_results[ind, 2].astype(int)])
    np.save(r'.//results//modescaling//npdata_linecountA_ind_lines_slope.npy', ind_lines_slope)
    print('check linemode + bidomain = slope A')
    print(np.all(ind_lines_slope[:, 3] == raw_results[:, 2].astype(int)))


    return 0

def check3x3_config3x2(config3x3, config3x2):
    indlist = []
    conf3x3 = config3x3[:, 0:2*3, 0:2*2]
    conf3x3 = np.reshape(conf3x3, (-1, 2*3*2*2))
    conf3x2 = np.reshape(config3x2, (-1, 2*3*2*2))
    for i in range(np.shape(config3x3)[0]):
        for j in range(np.shape(config3x2)[0]):
            if np.all(conf3x3[i] == conf3x2[j]):
                indlist.append(i)
    return indlist

def oldmain():
    kx = 3
    ky = 2
    k=3
    combinations = iter.product(range(4), repeat=kx*ky)
    combinations = np.fromiter(iter.chain(*combinations), int).reshape(-1, kx*ky)
    PixelConfigs = Pixel_rep(combinations, kx, ky)
    # ind_lines = np.zeros((np.shape(PixelConfigs)[0], 2), dtype=int)
    # for i, config in enumerate(PixelConfigs):
    #     config = np.tile(config, (2, 1))
    #     #configrot = np.tile(np.rot90(config), (2, 2))
    #     linemode = 0
    #     for lines in range(1):
    #         linemode += check_horizontal_linemode_3x2(config, lines)
    #         #linemode += check_horizontal_linemode_3x2(configrot, lines)
    #     print('i: {:d} \t lines: {:d} \n'.format(i, linemode))
    #     ind_lines[i] = np.array([i, linemode])
    # np.save(r'.//results//modescaling//3x2_linemode_count', ind_lines)
    ind_lines = np.load(r'.//results//modescaling//3x2_linemode_count.npy')
    print(np.count_nonzero(ind_lines[:, 1] > 0))
    print(np.shape(ind_lines)[0])
    datapath = os.path.dirname('C:\\Users\\ryanv\\PycharmProjects\\metacombi\\results\\modescaling\\')
    respath = os.path.dirname('C:\\Users\\ryanv\\PycharmProjects\\metacombi\\results\\modescaling\\')
    ind_B = np.argwhere(ind_lines[:, 1]>0)
    raw_data = np.loadtxt(datapath + '\\PixelRep_{:d}x{:d}.txt'.format(k, k), delimiter=',')
    raw_results = np.loadtxt(respath + '\\results_analysis_new_rrQR_i_Scen_slope_offset_M1k_{:d}x{:d}.txt'.format(k, k),
                             delimiter=',')
    y_total = raw_results[:, 1].astype(int)
    x_total = raw_data.reshape(-1, 2 * k, 2 * k).astype(int)
    # indlist = check3x3_config3x2(x_total, PixelConfigs[ind_B[:, 0]])
    # np.save(r'.//results//modescaling//indlist_3x2linemodecount_B', indlist)
    indlist = np.load(r'.//results//modescaling//indlist_3x2linemodecount_B.npy')
    print(np.count_nonzero(y_total[indlist[:]]>0))
    return 0

def check_misclassified_rules_label(k):
    if k==7:
        dat = np.load(r'.//results//modescaling//{:d}x{:d}_codecount_vs_rawresults_simple_onlyDiagOrVer_fix_2SCWidth.npz'.format(k, k))
    else:
        dat = np.load(
            r'.//results//modescaling//{:d}x{:d}_codecount_vs_rawresults_simple_onlyDiagOrVer.npz'.format(
                k, k))
    datapath = os.path.dirname('C:\\Users\\ryanv\\PycharmProjects\\metacombi\\results\\modescaling\\')
    raw_data = np.loadtxt(datapath + '\\PixelRep_{:d}x{:d}.txt'.format(k, k), delimiter=',')

    x_total = raw_data.reshape(-1, 2 * k, 2 * k)

    rules_vs_slopeA = dat['A']
    rules_vs_slopeB = dat['B']
    indneqA = np.argwhere(rules_vs_slopeA[:, 1] != rules_vs_slopeA[:, 2])
    indneqB = np.argwhere(rules_vs_slopeB[:, 1] != rules_vs_slopeB[:, 2])
    indneqs = [indneqA, indneqB]
    rules_vs_slopes = [rules_vs_slopeA, rules_vs_slopeB]
    if k==7:
        pdf = matplotlib.backends.backend_pdf.PdfPages(u".//results//modescaling//non_lm_or_bidomain_{:d}x{:d}_configs_onlyDiagOrVer_fix_2SCWidth.pdf".format(k, k))
    else:
        pdf = matplotlib.backends.backend_pdf.PdfPages(
            u".//results//modescaling//non_lm_or_bidomain_{:d}x{:d}_configs_onlyDiagOrVer.pdf".format(k, k))
    label = ['A', 'B']
    for a in range(len(indneqs)):
        for i, ind in enumerate(indneqs[a][:, 0]):
            f, ax = plt.subplots()
            resind = rules_vs_slopes[a][ind, 0]
            ax.imshow(x_total[resind], cmap='Greys', vmin=0, vmax=1)
            for x in range(1, 2*k, 2):
                ax.axvline(x-0.5, color='grey')
            for y in range(1, 2*k, 2):
                ax.axhline(y-0.5, color='grey')
            ax.set_title('True label:'+label[a]+'\t Code slope: {:d}'.format(rules_vs_slopes[a][ind, 1]))
            pdf.savefig(f)
            # plt.show()
            plt.close()
    pdf.close()
    return 0

def check_lm_mechanism(k, extended=False):
    datapath = os.path.dirname('C:\\Users\\ryanv\\PycharmProjects\\metacombi\\results\\modescaling\\')
    respath = os.path.dirname('C:\\Users\\ryanv\\PycharmProjects\\metacombi\\results\\modescaling\\')
    # x_total = np.load(u".\\results\\modescaling\\x_total_4x4_first100.npy")
    # raw_results = np.load(u".\\results\\modescaling\\results_4x4_first100.npy")
    if extended:
        raw_data = np.loadtxt(datapath + '\\PixelRep_{:d}x{:d}_extended.txt'.format(k, k), delimiter=',')
    else:
        raw_data = np.loadtxt(datapath + '\\PixelRep_{:d}x{:d}.txt'.format(k, k), delimiter=',')
    if k == 7:
        if extended:
            raw_results = np.loadtxt(
                respath + '\\results_analysis_new_rrQR_i_Scen_slope_M1k_{:d}x{:d}_extended.txt'.format(k, k),
                delimiter=',')
        else:
            raw_results = np.loadtxt(
                respath + '\\results_analysis_new_rrQR_i_Scen_slope_M1k_{:d}x{:d}.txt'.format(k, k),
                delimiter=',')
    elif k == 8:
        if extended:
            raw_results = np.loadtxt(
                respath + '\\results_analysis_new_rrQR_i_Scen_slope_M1k_{:d}x{:d}_extended.txt'.format(k, k),
                delimiter=',')
        else:
            raw_results = np.loadtxt(
                respath + '\\results_analysis_new_rrQR_i_Scen_slope_M1k_{:d}x{:d}.txt'.format(k, k),
                delimiter=',')

    elif k==6:
        raw_results = np.loadtxt(respath + '\\results_analysis_new_rrQR_i_Scen_slope_M1k_{:d}x{:d}.txt'.format(k, k),
                                delimiter=',')
    elif k <= 5 and k >= 3:
        raw_results = np.loadtxt(respath + r'\\results_analysis_i_Scen_slope_offset_M1k_{:d}x{:d}_fixn4.txt'
                                 .format(k, k), delimiter=',')
    else:
        raw_results = np.loadtxt(respath + '\\results_analysis_new_rrQR_i_Scen_slope_offset_M1k_{:d}x{:d}.txt'.format(k, k),
                                delimiter=',')
    "remove nan data"
    # nanind = np.argwhere(np.isnan(raw_results[:, 2]))
    # raw_data = np.delete(raw_data, nanind[:, 0], axis=0)
    # raw_results = np.delete(raw_results, nanind[:, 0], axis=0)
    y_total = raw_results[:, 1].astype(int)
    x_total = raw_data.reshape(-1, 2 * k, 2 * k)
    indB = np.argwhere(y_total==1)
    indA = np.argwhere(y_total==0)
    Code_True_A = np.zeros((np.shape(indA)[0], 3), dtype=int)
    Code_True_B = np.zeros((np.shape(indB)[0], 3), dtype=int)
    modelist_B_hor = []
    modelist_B_ver = []
    Bcount = 0
    Acount = 0
    # wrong = 0
    for i in range(0, np.shape(x_total)[0]):
        linemodes=0
        config = x_total[i]
        linemodes, modelisthor, modelistver = check_horizontal_connections(np.tile(config, (2, 2)), k)
        if y_total[i]==1:
            "B"
            Code_True_B[Bcount] = np.array([i, linemodes, raw_results[i, 2].astype(int)])
            modelist_B_hor.append(modelisthor)
            modelist_B_ver.append(modelistver)
            Bcount+=1
        elif y_total[i]==0:
            "A"
            Code_True_A[Acount] = np.array([i, linemodes, raw_results[i, 2].astype(int)])
            Acount+=1
        # if linemodes != int(raw_results[i, 2]):
        #     wrong+=1

        # linemodes += check_horizontal_connections(np.tile(np.rot90(config), (2, 2)), k)
        # print('code:')
        # print(linemodes)
        # print('actual:')
        # print(raw_results[indA[i, 0], 2])
        # print("index:")
        # print(i)
        print("index: {:d} \t code: {:d} \t actual: {:d}".format(i, linemodes, raw_results[i, 2].astype(int)))
        # f, ax = plt.subplots()
        # ax.imshow(config, cmap='Greys', vmin=0, vmax=1)
        # for x in range(1, 2 * k, 2):
        #     ax.axvline(x - 0.5, color='grey')
        # for y in range(1, 2 * k, 2):
        #     ax.axhline(y - 0.5, color='grey')
        # plt.show()
        # plt.close()
    if k==7:
        if extended:
            np.savez(
                u'.\\results\\modescaling\\{:d}x{:d}_codecount_vs_rawresults_simple_onlyDiagOrVer_extended.npz'.format(
                    k, k),
                A=Code_True_A, B=Code_True_B)
        else:
            np.savez(
                u'.\\results\\modescaling\\{:d}x{:d}_codecount_vs_rawresults_simple_onlyDiagOrVer_fix_2SCWidth.npz'.
                    format(k, k),
                A=Code_True_A, B=Code_True_B)
    else:
        if extended:
            np.savez(
                u'.\\results\\modescaling\\{:d}x{:d}_codecount_vs_rawresults_simple_onlyDiagOrVer_extended.npz'
                    .format(k, k),
                A=Code_True_A, B=Code_True_B)
        else:
            np.savez(u'.\\results\\modescaling\\{:d}x{:d}_codecount_vs_rawresults_simple_onlyDiagOrVer.npz'.format(k, k)
                     , A=Code_True_A, B=Code_True_B)
    # print('Wrong:')
    # print(wrong)
    if extended:
        with open(u'.\\results\\modescaling\\{:d}x{:d}_modelistB_hor_row_line_onlyDiagOrVer_extended.data'.format(k, k), 'wb') as filehandle:
            # save the data as binary data stream
            pickle.dump(modelist_B_hor, filehandle)
        with open(u'.\\results\\modescaling\\{:d}x{:d}_modelistB_ver_row_line_onlyDiagOrVer_extended.data'.format(k, k), 'wb') as filehandle:
            # save the data as binary data stream
            pickle.dump(modelist_B_ver, filehandle)
    else:
        with open(u'.\\results\\modescaling\\{:d}x{:d}_modelistB_hor_row_line_onlyDiagOrVer.data'.format(k, k), 'wb') as filehandle:
            # save the data as binary data stream
            pickle.dump(modelist_B_hor, filehandle)
        with open(u'.\\results\\modescaling\\{:d}x{:d}_modelistB_ver_row_line_onlyDiagOrVer.data'.format(k, k), 'wb') as filehandle:
            # save the data as binary data stream
            pickle.dump(modelist_B_ver, filehandle)
    return 0

def check_lm_mechanism_C(k):
    datapath = os.path.dirname('C:\\Users\\ryanv\\PycharmProjects\\metacombi\\results\\modescaling\\')
    respath = os.path.dirname('C:\\Users\\ryanv\\PycharmProjects\\metacombi\\results\\modescaling\\')
    # x_total = np.load(u".\\results\\modescaling\\x_total_4x4_first100.npy")
    # raw_results = np.load(u".\\results\\modescaling\\results_4x4_first100.npy")
    raw_data = np.loadtxt(datapath + '\\PixelRep_{:d}x{:d}.txt'.format(k, k), delimiter=',')
    if k == 7 or k == 8:
        raw_results = np.loadtxt(
            respath + '\\results_analysis_new_rrQR_i_Scen_slope_M1k_{:d}x{:d}_extended.txt'.format(k, k),
                                delimiter=',')
    elif k==6:
        raw_results = np.loadtxt(respath + '\\results_analysis_new_rrQR_i_Scen_slope_M1k_{:d}x{:d}.txt'.format(k, k),
                                delimiter=',')
    elif k<=5 and k>=3:
        raw_results = np.loadtxt(respath + r'\\results_analysis_new_rrQR_i_Scen_slope_offset_M1k_{:d}x{:d}_fixn4.txt'
                                 .format(k, k), delimiter=',')
    else:
        raw_results = np.loadtxt(respath + '\\results_analysis_new_rrQR_i_Scen_slope_offset_M1k_{:d}x{:d}.txt'.format(k, k),
                                delimiter=',')
    "remove nan data"
    # nanind = np.argwhere(np.isnan(raw_results[:, 2]))
    # raw_data = np.delete(raw_data, nanind[:, 0], axis=0)
    # raw_results = np.delete(raw_results, nanind[:, 0], axis=0)
    y_total = raw_results[:, 1].astype(int)
    x_total = raw_data.reshape(-1, 2 * k, 2 * k)
    # indB = np.argwhere(y_total==1)
    # indA = np.argwhere(y_total==0)
    indC = np.argwhere(y_total == 2)
    # Code_True_A = np.zeros((np.shape(indA)[0], 3), dtype=int)
    # Code_True_B = np.zeros((np.shape(indB)[0], 3), dtype=int)
    Code_True_C = np.zeros((np.shape(indC)[0], 3), dtype=int)
    modelist_C_hor = []
    modelist_C_ver = []
    Ccount = 0
    # wrong = 0
    for i in range(0, np.shape(indC)[0]):
        linemodes=0
        config = x_total[indC[i, 0]]
        linemodes, modelisthor, modelistver = check_horizontal_connections(np.tile(config, (2, 2)), k)
        Code_True_C[i] = np.array([indC[i, 0], linemodes, raw_results[indC[i, 0], 2].astype(int)])
        modelist_C_hor.append(modelisthor)
        modelist_C_ver.append(modelistver)
        print("index: {:d} \t code: {:d} \t actual: {:d}".format(indC[i, 0], linemodes, raw_results[indC[i, 0], 2].astype(int)))
    if k==7 or k == 8:
        np.save(u'.\\results\\modescaling\\{:d}x{:d}_codecount_vs_rawresults_simple_onlyDiagOrVer_classC_extended.npy'.format(k, k),
                 Code_True_C)
    else:
        np.save(u'.\\results\\modescaling\\{:d}x{:d}_codecount_vs_rawresults_simple_onlyDiagOrVer_classC.npy'.format(k, k),
                Code_True_C)
    # print('Wrong:')
    # print(wrong)
    if k == 7 or k == 8:
        with open(u'.\\results\\modescaling\\{:d}x{:d}_modelistC_hor_row_line_onlyDiagOrVer_extended.data'.format(k, k),
                  'wb') as filehandle:
            # save the data as binary data stream
            pickle.dump(modelist_C_hor, filehandle)
        with open(u'.\\results\\modescaling\\{:d}x{:d}_modelistC_ver_row_line_onlyDiagOrVer_extended.data'.format(k, k),
                  'wb') as filehandle:
            # save the data as binary data stream
            pickle.dump(modelist_C_ver, filehandle)
    else:
        with open(u'.\\results\\modescaling\\{:d}x{:d}_modelistC_hor_row_line_onlyDiagOrVer.data'.format(k, k), 'wb') as filehandle:
            # save the data as binary data stream
            pickle.dump(modelist_C_hor, filehandle)
        with open(u'.\\results\\modescaling\\{:d}x{:d}_modelistC_ver_row_line_onlyDiagOrVer.data'.format(k, k), 'wb') as filehandle:
            # save the data as binary data stream
            pickle.dump(modelist_C_ver, filehandle)
    return 0

def print_lm_C(k):
    datapath = os.path.dirname('C:\\Users\\ryanv\\PycharmProjects\\metacombi\\results\\modescaling\\')
    "load in data [index, Number of line modes, class (C = nan)]"
    dat = np.load(u'.\\results\\modescaling\\{:d}x{:d}_codecount_vs_rawresults_simple_onlyDiagOrVer_classC.npy'
                  .format(k, k))
    "load in raw results"
    raw_data = np.loadtxt(datapath + u'\\PixelRep_{:d}x{:d}.txt'.format(k, k), delimiter=',')
    if k <= 4:
        raw_results = np.loadtxt(datapath + u'\\data_new_rrQR_i_n_M_{:d}x{:d}.txt'.format(k, k), delimiter=',')
    elif k == 5:
        raw_results = np.load(datapath + u'\\data_new_rrQR_i_n_M_{:d}x{:d}_fixn4.npy'.format(k, k))

    with PdfPages(datapath + u'\\classC_PixRep_ModeScaling_{:d}x{:d}.pdf'.format(k, k)) as pdf:
        for i, ind in enumerate(dat[:, 0]):
            f, ax = plt.subplots(1, 2)
            ax[0].imshow(np.reshape(raw_data[ind], (2*k, 2*k)), cmap='Greys', vmin=0, vmax=1)
            for x in range(1, 2*k, 2):
                ax[0].axvline(x-0.5, color='grey')
            for y in range(1, 2*k, 2):
                ax[0].axhline(y-0.5, color='grey')
            ax[1].plot(np.arange(1, 5), raw_results[ind, k*k+1: k*k+1+4], '.-')
            ax[1].set_xlabel('n')
            ax[1].set_ylabel('$M_{:d}(n)$'.format(k))
            plt.title('Class C, lm: {:d}'.format(dat[i, 1]))
            pdf.savefig(f)
            plt.close()
        d = pdf.infodict()
        d['Title'] = 'Class C Pixel Representation and Mode Scaling'
        d['Author'] = 'Ryan van Mastrigt'
    return 0

def lm_correction_C(k):
    datapath = os.path.dirname('C:\\Users\\ryanv\\PycharmProjects\\metacombi\\results\\modescaling\\')
    "load in data [index, Number of line modes, class (C = nan)]"
    dat = np.load(u'.\\results\\modescaling\\{:d}x{:d}_codecount_vs_rawresults_simple_onlyDiagOrVer_classC.npy'
                  .format(k, k))
    "load in raw results"
    raw_data = np.loadtxt(datapath + u'\\PixelRep_{:d}x{:d}.txt'.format(k, k), delimiter=',')
    if k <= 4:
        raw_results = np.loadtxt(datapath + u'\\data_new_rrQR_i_n_M_{:d}x{:d}.txt'.format(k, k), delimiter=',')
    elif k == 5:
        raw_results = np.load(datapath + u'\\data_new_rrQR_i_n_M_{:d}x{:d}_fixn4.npy'.format(k, k))

    ModeScaling = raw_results[dat[:, 0], k*k+1: k*k+1+4]
    ind_nolm = np.argwhere(dat[:, 1] == 0)
    ind_lm = np.argwhere(dat[:, 1] >= 1)
    for i in range(np.shape(ModeScaling)[0]):
        ModeScaling[i] = ModeScaling[i] - np.multiply(np.arange(1, 5), dat[i, 1])
    Modes_nolm, counts_nolm = np.unique(ModeScaling[ind_nolm[:, 0]], axis=0, return_counts=True)
    Modes_lm, counts_lm = np.unique(ModeScaling[ind_lm[:, 0]], axis=0, return_counts=True)
    print('no line mode:')
    print(Modes_nolm)
    print(counts_nolm)
    print('line modes:')
    print(Modes_lm)
    print(counts_lm)
    return 0

def check_lm_mechanism_C_extended(k):
    datapath = os.path.dirname('C:\\Users\\ryanv\\PycharmProjects\\metacombi\\results\\modescaling\\')
    respath = os.path.dirname('C:\\Users\\ryanv\\PycharmProjects\\metacombi\\results\\modescaling\\')
    # x_total = np.load(u".\\results\\modescaling\\x_total_4x4_first100.npy")
    # raw_results = np.load(u".\\results\\modescaling\\results_4x4_first100.npy")
    raw_data = np.loadtxt(datapath + '\\PixelRep_{:d}x{:d}.txt'.format(k, k), delimiter=',')
    if k == 7 or k == 8:
        raw_results = np.loadtxt(
            respath + '\\results_analysis_new_rrQR_i_Scen_slope_M1k_{:d}x{:d}_extended_classC_extend.txt'.format(k, k),
                                delimiter=',')
    elif k==6:
        raw_results = np.loadtxt(respath + '\\results_analysis_new_rrQR_i_Scen_slope_M1k_{:d}x{:d}_classC_extend.txt'.
                                 format(k, k),
                                delimiter=',')
    elif k<=5 and k>=3:
        raw_results = np.loadtxt(respath + r'\\results_analysis_new_rrQR_i_Scen_slope_offset_M1k_{:d}x{:d}_fixn4_'
                                           r'classC_extend.txt'
                                 .format(k, k), delimiter=',')
    else:
        raw_results = np.loadtxt(respath + '\\results_analysis_new_rrQR_i_Scen_slope_offset_M1k_{:d}x{:d}_classC_extend'
                                           '.txt'.format(k, k),
                                delimiter=',')
    "remove nan data"
    # nanind = np.argwhere(np.isnan(raw_results[:, 2]))
    # raw_data = np.delete(raw_data, nanind[:, 0], axis=0)
    # raw_results = np.delete(raw_results, nanind[:, 0], axis=0)
    y_total = raw_results[:, 1].astype(int)
    x_total = raw_data.reshape(-1, 2 * k, 2 * k)
    x_total = x_total[raw_results[:, 0].astype(int), :]
    # indB = np.argwhere(y_total==1)
    # indA = np.argwhere(y_total==0)
    # indC = np.argwhere(y_total == 2)
    # Code_True_A = np.zeros((np.shape(indA)[0], 3), dtype=int)
    # Code_True_B = np.zeros((np.shape(indB)[0], 3), dtype=int)
    Code_True_C = np.zeros((np.shape(y_total)[0], 3), dtype=int)
    modelist_C_hor = []
    modelist_C_ver = []
    Ccount = 0
    # wrong = 0
    for i in range(0, np.shape(y_total)[0]):
        linemodes=0
        config = x_total[i]
        linemodes, modelisthor, modelistver = check_horizontal_connections(np.tile(config, (2, 2)), k)
        Code_True_C[i] = np.array([raw_results[i, 0].astype(int), linemodes, raw_results[i, 2].astype(int)])
        modelist_C_hor.append(modelisthor)
        modelist_C_ver.append(modelistver)
        print("index: {:d} \t code: {:d} \t actual: {:d}".format(raw_results[i, 0].astype(int), linemodes,
                                                                 raw_results[i, 2].astype(int)))
    if k==7 or k == 8:
        np.save(u'.\\results\\modescaling\\{:d}x{:d}_codecount_vs_rawresults_simple_onlyDiagOrVer_classC_extended_n5n6.npy'.format(k, k),
                 Code_True_C)
    else:
        np.save(u'.\\results\\modescaling\\{:d}x{:d}_codecount_vs_rawresults_simple_onlyDiagOrVer_classC_n5n6.npy'.format(k, k),
                Code_True_C)
    # print('Wrong:')
    # print(wrong)
    if k == 7 or k == 8:
        with open(u'.\\results\\modescaling\\{:d}x{:d}_modelistC_hor_row_line_onlyDiagOrVer_extended_n5n6.data'.format(k, k),
                  'wb') as filehandle:
            # save the data as binary data stream
            pickle.dump(modelist_C_hor, filehandle)
        with open(u'.\\results\\modescaling\\{:d}x{:d}_modelistC_ver_row_line_onlyDiagOrVer_extended_n5n6.data'.format(k, k),
                  'wb') as filehandle:
            # save the data as binary data stream
            pickle.dump(modelist_C_ver, filehandle)
    else:
        with open(u'.\\results\\modescaling\\{:d}x{:d}_modelistC_hor_row_line_onlyDiagOrVer_n5n6.data'.format(k, k), 'wb') as filehandle:
            # save the data as binary data stream
            pickle.dump(modelist_C_hor, filehandle)
        with open(u'.\\results\\modescaling\\{:d}x{:d}_modelistC_ver_row_line_onlyDiagOrVer_n5n6.data'.format(k, k), 'wb') as filehandle:
            # save the data as binary data stream
            pickle.dump(modelist_C_ver, filehandle)
    return 0

def print_lm_C_extended(k):
    datapath = os.path.dirname('C:\\Users\\ryanv\\PycharmProjects\\metacombi\\results\\modescaling\\')
    "load in data [index, Number of line modes, class (C = nan)]"
    dat = np.load(u'.\\results\\modescaling\\{:d}x{:d}_codecount_vs_rawresults_simple_onlyDiagOrVer_classC_n5n6.npy'
                  .format(k, k))
    "load in raw results"
    raw_data = np.loadtxt(datapath + u'\\PixelRep_{:d}x{:d}.txt'.format(k, k), delimiter=',')
    raw_results = np.load(datapath + u'\\data_new_rrQR_i_n_M_{:d}x{:d}_fixn4_classC_extend.npy'.format(k, k))

    with PdfPages(datapath + u'\\classC_PixRep_ModeScaling_{:d}x{:d}_n5n6.pdf'.format(k, k)) as pdf:
        for i, ind in enumerate(dat[:, 0]):
            f, ax = plt.subplots(1, 2)
            ax[0].imshow(np.reshape(raw_data[ind], (2*k, 2*k)), cmap='Greys', vmin=0, vmax=1)
            for x in range(1, 2*k, 2):
                ax[0].axvline(x-0.5, color='grey')
            for y in range(1, 2*k, 2):
                ax[0].axhline(y-0.5, color='grey')
            ax[1].plot(np.arange(1, 7), raw_results[i, k*k+1: k*k+1+6], '.-')
            ax[1].set_xlabel('$n$')
            ax[1].set_ylabel('$M_{:d}(n)$'.format(k))
            if dat[i, 2] == 0:
                plt.title('Class I, lm: {:d}'.format(dat[i, 1]))
            elif dat[i, 2] == 1:
                plt.title('Class C, lm: {:d}'.format(dat[i, 1]))
            else:
                plt.title('Class X, lm: {:d}'.format(dat[i, 1]))
            pdf.savefig(f)
            plt.close()
        d = pdf.infodict()
        d['Title'] = 'Class X Pixel Representation and Mode Scaling extended to n=5 and n=6'
        d['Author'] = 'Ryan van Mastrigt'
    return 0

def confusion_matrices_rules():
    datapath = os.path.dirname('C:\\Users\\ryanv\\PycharmProjects\\metacombi\\results\\modescaling\\')
    respath = os.path.dirname('C:\\Users\\ryanv\\PycharmProjects\\metacombi\\results\\modescaling\\')
    plt.style.use(r'C:\\Users\\ryanv\\PycharmProjects\\Matplotlib styles\\paper-onehalf.mplstyle')
    f = plt.figure(1, figsize=(cm_to_inch(8.6), cm_to_inch(6.1)))
    xlength = 3.32
    ylength = 0.3
    # f = plt.figure(1, figsize=(cm_to_inch(4.45), cm_to_inch(4.15)))
    xoffset = 0.2 * (xlength / 8.6)
    yoffset = 0.2 * (xlength / 6.1)
    # xoffset = 0.25
    # yoffset =
    frac_pad_x = 0.07 * (2./3.)
    frac_pad_y = frac_pad_x*8.6/6.1
    figfracx = (8.5 - xoffset * 8.6 - 2*frac_pad_x*8.6) / 8.6
    figfracy = figfracx * 8.6 / (6.1)
    for k in range(3, 9):
        if k == 7:
            raw_results = np.loadtxt(
                respath + '\\results_analysis_new_rrQR_i_Scen_slope_M1k_{:d}x{:d}_extended.txt'.format(k, k),
                delimiter=',')
        elif k == 8:
            raw_results = np.loadtxt(
                respath + '\\results_analysis_new_rrQR_i_Scen_slope_M1k_{:d}x{:d}_extended.txt'.format(k, k),
                delimiter=',')

        elif k == 6:
            raw_results = np.loadtxt(
                respath + '\\results_analysis_new_rrQR_i_Scen_slope_M1k_{:d}x{:d}.txt'.format(k, k),
                delimiter=',')
        elif k <= 5 and k >= 3:
            raw_results = np.loadtxt(respath + r'\\results_analysis_new_rrQR_i_Scen_slope_offset_M1k_{:d}x{:d}_'
                                               r'fixn4.txt'
                                     .format(k, k), delimiter=',')
            results_X_extended = np.load(datapath + u'\\data_new_rrQR_i_n_M_{:d}x{:d}_fixn4_classC_extend.npy'.
                                         format(k, k))
            dat_X_extended = np.load(
                u'.\\results\\modescaling\\{:d}x{:d}_codecount_vs_rawresults_simple_onlyDiagOrVer_classC_n5n6.npy'
                .format(k, k))
        else:
            raw_results = np.loadtxt(
                respath + '\\results_analysis_new_rrQR_i_Scen_slope_offset_M1k_{:d}x{:d}.txt'.format(k, k),
                delimiter=',')
        if k >= 7:
            dat = np.load(u'.\\results\\modescaling\\{:d}x{:d}_codecount_vs_rawresults_simple_onlyDiagOrVer'
                          u'_extended.npz'
                          .format(k, k))
        else:
            dat = np.load(u'.\\results\\modescaling\\{:d}x{:d}_codecount_vs_rawresults_simple_onlyDiagOrVer.npz'
                          .format(k, k))
        datI = dat['A']
        datC = dat['B']
        y_total = raw_results[:, 1].astype(int)
        indI = np.argwhere(y_total==0)[:, 0]
        indC = np.argwhere(y_total==1)[:, 0]
        indX = np.argwhere(y_total==2)[:, 0]
        tIpI = np.shape(np.argwhere(datI[:, 1] == 0))[0]
        tIpC = np.shape(np.argwhere(datI[:, 1] != 0))[0]
        tCpC = np.shape(np.argwhere(datC[:, 1] != 0))[0]
        tCpI = np.shape(np.argwhere(datC[:, 1] == 0))[0]
        if np.shape(indX)[0]>0:
            tIpI += np.shape(np.argwhere(np.logical_and(dat_X_extended[:, 1] == 0, dat_X_extended[:, 2] == 0)))[0]
            tIpC += np.shape(np.argwhere(np.logical_and(dat_X_extended[:, 1] != 0, dat_X_extended[:, 2] == 0)))[0]
            tCpC += np.shape(np.argwhere(np.logical_and(dat_X_extended[:, 1] != 0, dat_X_extended[:, 2] != 0)))[0]
            tCpI += np.shape(np.argwhere(np.logical_and(dat_X_extended[:, 1] == 0, dat_X_extended[:, 2] != 0)))[0]
        CM = np.zeros((2, 2), dtype=int)
        CM[0, 0] = tIpI
        CM[1, 0] = tIpC
        CM[0, 1] = tCpI
        CM[1, 1] = tCpC
        ax = f.add_axes([xoffset + int((k-3)/2)*(frac_pad_x+figfracx/3.), yoffset+(k % 2) * (frac_pad_y + figfracy/3.),
                         figfracx/3., figfracy/3.])
        ax.imshow(CM, cmap='Blues', vmin=0, vmax=1, origin='lower')
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        # ax.text(-0.44, -0.1, np.format_float_scientific(CM[0, 0], precision=2, unique=True, exp_digits=1, min_digits=2),
        #         c='white', fontsize=8)
        ax.text(-0, 0, '{:d}'.format(CM[0, 0]), c='white', fontsize=7, ha='center', va='center')
        ax.text(-0, 1, '{:d}'.format(CM[1, 0]), c='black', fontsize=7, ha='center', va='center')
        ax.text(1, -0, '{:d}'.format(CM[0, 1]), c='black', fontsize=7, ha='center', va='center')
        ax.text(1, 1, '{:d}'.format(CM[1, 1]), c='white', fontsize=7, ha='center', va='center')
        # ax.text(0.56, 0.9, np.format_float_scientific(CM[1, 1], precision=2, unique=True, exp_digits=1, min_digits=2),
        #         c='white', fontsize=8)
        ax.set_title(r'${:d} \times {:d}$'.format(k, k), fontsize=8, pad=0.01)
        if k%2 == 0:
            ax.set_xlabel('mode scaling')

            ax.set_xticklabels(['I', 'C'])
        else:
            ax.set_xticklabels([])
        if int((k-3)/2) == 0:
            ax.set_ylabel('rules')

            ax.set_yticklabels(['I', 'C'])
        else:
            ax.set_yticklabels([])
    f.savefig('.\\results\\modescaling\\figures\\modescaling_vs_rules_ConfusionMatrices.pdf',
              facecolor=f.get_facecolor())
    f.savefig('.\\results\\modescaling\\figures\\modescaling_vs_rules_ConfusionMatrices.svg',
              facecolor=f.get_facecolor())
    f.savefig('.\\results\\modescaling\\figures\\modescaling_vs_rules_ConfusionMatrices.png',
              facecolor=f.get_facecolor(), dpi=400)
    plt.show()
    plt.close()
    return 0



def main():
    confusion_matrices_rules()
    # for k in range(5, 6):
    #     # check_lm_mechanism(k)
    #     # check_lm_mechanism_C(k)
    #     check_lm_mechanism_C_extended(k)
    #     print_lm_C_extended(k)
    # for k in  range(3,  9):
    #     check_lm_mechanism_C(k)
    # check_lm_mechanism(k, extended=True)
    # check_misclassified_rules_label(k)

    # ind_line_slope_A = np.load(r'.//results//modescaling//npdata_linecountA_ind_lines_slope.npy')
    # ind_line_slope_B = np.load(r'.//results//modescaling//npdata_linecountB_ind_lines_slope.npy')
    # indneqA = np.argwhere(ind_line_slope_A[:, -1] != ind_line_slope_A[:, 3])
    # indneqB = np.argwhere(ind_line_slope_B[:, -1] != ind_line_slope_B[:, 3])
    # indneqs = np.append(indneqA, indneqB)
    # pdf = matplotlib.backends.backend_pdf.PdfPages(u".//results//modescaling//non_lm_or_bidomain_4x4_configs.pdf")
    # for i, ind in enumerate(ind_line_slope_B[indneqB[:, 0], 0]):
    #     f, ax = plt.subplots()
    #     ax.imshow(x_total[ind], cmap='Greys', vmin=0, vmax=1)
    #     for x in range(1, 2*k, 2):
    #         ax.axvline(x-0.5, color='grey')
    #     for y in range(1, 2*k, 2):
    #         ax.axhline(y-0.5, color='grey')
    #     pdf.savefig(f)
    #     # plt.show()
    #     plt.close()
    # pdf.close()
    return 0

if __name__ == '__main__':
    main()