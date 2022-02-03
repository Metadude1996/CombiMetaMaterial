import numpy as np
import matplotlib.pyplot as plt
# from matplotlib import cm
from matplotlib import animation
from autopenta import get_modes
from check_linemode_mechanism import Pixel_rep
import matplotlib as mpl
import os

def categorical_cmap(nc, nsc, cmap="tab10", continuous=False, offset=0):
    if nc > plt.get_cmap(cmap).N:
        raise ValueError("Too many categories for colormap.")
    if continuous:
        ccolors = plt.get_cmap(cmap)(np.linspace(0,1,nc))
    else:
        ccolors = plt.get_cmap(cmap)(np.arange(offset, offset+nc, dtype=int))
    cols = np.zeros((nc*nsc, 3))
    for i, c in enumerate(ccolors):
        chsv = mpl.colors.rgb_to_hsv(c[:3])
        arhsv = np.tile(chsv,nsc).reshape(nsc,3)
        arhsv[:,1] = np.linspace(chsv[1],0.25,nsc)
        arhsv[:,2] = np.linspace(chsv[2],1,nsc)
        rgb = mpl.colors.hsv_to_rgb(arhsv)
        cols[i*nsc:(i+1)*nsc,:] = rgb
    cmap = mpl.colors.ListedColormap(cols)
    return cmap

def get_triangles(x0, y0, c, t1, t2, t3, t4):
    triangles = [None] * 4
    count = 0
    if t1:
        triangles[count] = plt.Polygon([[x0, y0], [x0 + 1, y0], [x0, y0 + 1]], fc=c, ec='black', fill=True)
        # triangles[count] = plt.Polygon([[x0, y0], [x0 + 1, y0], [x0, y0 + 1]], fc=c, ec=c, fill=True)
        count += 1
    if t2:
        triangles[count] = plt.Polygon([[x0 + 1, y0], [x0 + 2, y0], [x0 + 2, y0 + 1]], fc=c, ec='black', fill=True)
        # triangles[count] = plt.Polygon([[x0 + 1, y0], [x0 + 2, y0], [x0 + 2, y0 + 1]], fc=c, ec=c, fill=True)
        count += 1
    if t3:
        triangles[count] = plt.Polygon([[x0 + 2, y0 + 1], [x0 + 2, y0 + 2], [x0 + 1, y0 + 2]], fc=c, ec='black',
                                       fill=True)
        # triangles[count] = plt.Polygon([[x0+2, y0+1], [x0+2, y0+2], [x0+1, y0+2]], fc=c, ec=c, fill=True)
        count += 1
    if t4:
        triangles[count] = plt.Polygon([[x0, y0 + 1], [x0, y0 + 2], [x0 + 1, y0 + 2]], fc=c, ec='black', fill=True)
        # triangles[count] = plt.Polygon([[x0, y0 + 1], [x0, y0 + 2], [x0 + 1, y0 + 2]], fc=c, ec=c, fill=True)
        count += 1
    return triangles[0], triangles[1], triangles[2], triangles[3]


def draw_schematic(lattice, ax):
    # f = plt.figure()
    # ax = plt.axes()
    # circle = plt.Circle((0, 0), radius=0.75, fc='y')
    # plt.gca().add_patch(circle)
    # c1 = (51/255, 255/255, 255/255)
    # c2 = (29/255, 143/255, 254/255)
    # c3 = (255/255, 204/255, 51/255)
    # c4 = (254/255, 127/255, 0/255)
    cp = (191 / 255, 191 / 255, 255 / 255)
    cl = 'black'
    cb = (0 / 255, 0 / 255, 139 / 255)
    cb2 = (124 / 255, 124 / 255, 255 / 255)
    c1, c2, c3, c4 = cp, cp, cp, cp
    cma = plt.cm.get_cmap('tab20c')
    cm_skip = [cma.colors[i] for i in np.arange(0, 4)]
    # cm_skip = [cma.colors[i] for i in np.arange(0, 4)]
    # c1, c2, c3, c4 = cm_skip[0], cm_skip[1], cm_skip[2], cm_skip[3]
    cmp = categorical_cmap(1, 4, cmap='tab20', offset=12)
    c1, c2, c3, c4 = cmp(0), cmp(1), cmp(2), cmp(3)
    linewidth = 1
    for i, row in enumerate(lattice):
        for j, cell in enumerate(row):
            x0 = 2 * i
            y0 = 2 * j
            if cell == 0:
                triangles = get_triangles(x0, y0, c1, 1, 1, 1, 0)
                "lines are given with (x0, x1), (y0, y1)"
                line1 = plt.Line2D((x0, x0), (y0 + 1, y0 + 2), lw=linewidth, color=cl)
                line2 = plt.Line2D((x0, x0 + 1), (y0 + 2, y0 + 2), lw=linewidth, color=cl)
                # line1 = plt.Line2D((x0, x0), (y0 + 1, y0 + 2), lw=linewidth, color=c1)
                # line2 = plt.Line2D((x0, x0 + 1), (y0 + 2, y0 + 2), lw=linewidth, color=c1)
                ax.add_patch(triangles[0])
                ax.add_patch(triangles[1])
                ax.add_patch(triangles[2])
                ax.add_line(line1)
                ax.add_line(line2)
            elif cell == 1:
                triangles = get_triangles(x0, y0, c2, 1, 1, 0, 1)
                "lines are given with (x0, x1), (y0, y1)"
                line1 = plt.Line2D((x0 + 1, x0 + 2), (y0 + 2, y0 + 2), lw=linewidth, color=cl)
                line2 = plt.Line2D((x0 + 2, x0 + 2), (y0 + 1, y0 + 2), lw=linewidth, color=cl)
                # line1 = plt.Line2D((x0 + 1, x0 + 2), (y0 + 2, y0 + 2), lw=linewidth, color=c2)
                # line2 = plt.Line2D((x0 + 2, x0 + 2), (y0 + 1, y0 + 2), lw=linewidth, color=c2)
                ax.add_patch(triangles[0])
                ax.add_patch(triangles[1])
                ax.add_patch(triangles[2])
                ax.add_line(line1)
                ax.add_line(line2)
            elif cell == 2:
                triangles = get_triangles(x0, y0, c3, 1, 0, 1, 1)
                "lines are given with (x0, x1), (y0, y1)"
                line1 = plt.Line2D((x0 + 1, x0 + 2), (y0, y0), lw=linewidth, color=cl)
                line2 = plt.Line2D((x0 + 2, x0 + 2), (y0 + 1, y0), lw=linewidth, color=cl)
                # line1 = plt.Line2D((x0 + 1, x0 + 2), (y0, y0), lw=linewidth, color=c3)
                # line2 = plt.Line2D((x0 + 2, x0 + 2), (y0 + 1, y0), lw=linewidth, color=c3)
                ax.add_patch(triangles[0])
                ax.add_patch(triangles[1])
                ax.add_patch(triangles[2])
                ax.add_line(line1)
                ax.add_line(line2)
            elif cell == 3:
                triangles = get_triangles(x0, y0, c4, 0, 1, 1, 1)
                "lines are given with (x0, x1), (y0, y1)"
                line1 = plt.Line2D((x0, x0 + 1), (y0, y0), lw=linewidth, color=cl)
                line2 = plt.Line2D((x0, x0), (y0, y0 + 1), lw=linewidth, color=cl)
                # line1 = plt.Line2D((x0, x0 + 1), (y0, y0), lw=linewidth, color=c4)
                # line2 = plt.Line2D((x0, x0), (y0, y0 + 1), lw=linewidth, color=c4)
                ax.add_patch(triangles[0])
                ax.add_patch(triangles[1])
                ax.add_patch(triangles[2])
                ax.add_line(line1)
                ax.add_line(line2)
            elif cell == -1:
                triangles = get_triangles(x0, y0, c4, 1, 1, 1, 1)
                ax.add_patch(triangles[0])
                ax.add_patch(triangles[1])
                ax.add_patch(triangles[2])
                ax.add_patch(triangles[3])
            else:
                print('invalid cell value given, space will be left empty')
    plt.axis('scaled')
    plt.axis('off')
    # modes = get_modes(np.array(lattice), 1, 1, draw_lattice=1)


def draw_pixel_rep(lattice, ax, plaquette=True):
    # plt.axes()
    for i, row in enumerate(lattice):
        for j, cell in enumerate(row):
            x0 = 2 * i
            y0 = 2 * j
            if cell == 0:
                rectangle = plt.Rectangle((x0, y0 + 1), 1, 1, fc='0')
                if plaquette:
                    line1 = plt.Line2D((x0 + 1, x0 + 1), (y0, y0 + 2), lw=1, color='0.5')
                    line2 = plt.Line2D((x0, x0 + 2), (y0 + 1, y0 + 1), lw=1, color='0.5')
                    lines = [line1, line2]
                else:
                    line1 = plt.Line2D((x0, x0), (y0, y0 + 2), lw=1, color='0.5')
                    line2 = plt.Line2D((x0 + 2, x0 + 2), (y0, y0 + 2), lw=1, color='0.5')
                    line3 = plt.Line2D((x0, x0 + 2), (y0, y0), lw=1, color='0.5')
                    line4 = plt.Line2D((x0, x0 + 2), (y0 + 2, y0 + 2), lw=1, color='0.5')
                    lines = [line1, line2, line3, line4]
                ax.add_patch(rectangle)
                for line in lines:
                    ax.add_line(line)
            elif cell == 1:
                rectangle = plt.Rectangle((x0 + 1, y0 + 1), 1, 1, fc='0')
                if plaquette:
                    line1 = plt.Line2D((x0 + 1, x0 + 1), (y0, y0 + 2), lw=1, color='0.5')
                    line2 = plt.Line2D((x0, x0 + 2), (y0 + 1, y0 + 1), lw=1, color='0.5')
                    lines = [line1, line2]
                else:
                    line1 = plt.Line2D((x0, x0), (y0, y0 + 2), lw=1, color='0.5')
                    line2 = plt.Line2D((x0 + 2, x0 + 2), (y0, y0 + 2), lw=1, color='0.5')
                    line3 = plt.Line2D((x0, x0 + 2), (y0, y0), lw=1, color='0.5')
                    line4 = plt.Line2D((x0, x0 + 2), (y0 + 2, y0 + 2), lw=1, color='0.5')
                    lines = [line1, line2, line3, line4]
                ax.add_patch(rectangle)
                for line in lines:
                    ax.add_line(line)
            elif cell == 2:
                rectangle = plt.Rectangle((x0 + 1, y0), 1, 1, fc='0')
                if plaquette:
                    line1 = plt.Line2D((x0 + 1, x0 + 1), (y0, y0 + 2), lw=1, color='0.5')
                    line2 = plt.Line2D((x0, x0 + 2), (y0 + 1, y0 + 1), lw=1, color='0.5')
                    lines = [line1, line2]
                else:
                    line1 = plt.Line2D((x0, x0), (y0, y0 + 2), lw=1, color='0.5')
                    line2 = plt.Line2D((x0 + 2, x0 + 2), (y0, y0 + 2), lw=1, color='0.5')
                    line3 = plt.Line2D((x0, x0 + 2), (y0, y0), lw=1, color='0.5')
                    line4 = plt.Line2D((x0, x0 + 2), (y0 + 2, y0 + 2), lw=1, color='0.5')
                    lines = [line1, line2, line3, line4]
                ax.add_patch(rectangle)
                for line in lines:
                    ax.add_line(line)
            elif cell == 3:
                rectangle = plt.Rectangle((x0, y0), 1, 1, fc='0')
                if plaquette:
                    line1 = plt.Line2D((x0 + 1, x0 + 1), (y0, y0 + 2), lw=1, color='0.5')
                    line2 = plt.Line2D((x0, x0 + 2), (y0 + 1, y0 + 1), lw=1, color='0.5')
                    lines = [line1, line2]
                else:
                    line1 = plt.Line2D((x0, x0), (y0, y0 + 2), lw=1, color='0.5')
                    line2 = plt.Line2D((x0 + 2, x0 + 2), (y0, y0 + 2), lw=1, color='0.5')
                    line3 = plt.Line2D((x0, x0 + 2), (y0, y0), lw=1, color='0.5')
                    line4 = plt.Line2D((x0, x0 + 2), (y0 + 2, y0 + 2), lw=1, color='0.5')
                    lines = [line1, line2, line3, line4]
                ax.add_patch(rectangle)
                for line in lines:
                    ax.add_line(line)
            elif cell == -1:
                if plaquette:
                    line1 = plt.Line2D((x0 + 1, x0 + 1), (y0, y0 + 2), lw=1, color='0.5')
                    line2 = plt.Line2D((x0, x0 + 2), (y0 + 1, y0 + 1), lw=1, color='0.5')
                    lines = [line1, line2]
                else:
                    line1 = plt.Line2D((x0, x0), (y0, y0 + 2), lw=1, color='0.5')
                    line2 = plt.Line2D((x0 + 2, x0 + 2), (y0, y0 + 2), lw=1, color='0.5')
                    line3 = plt.Line2D((x0, x0 + 2), (y0, y0), lw=1, color='0.5')
                    line4 = plt.Line2D((x0, x0 + 2), (y0 + 2, y0 + 2), lw=1, color='0.5')
                    lines = [line1, line2, line3, line4]
                for line in lines:
                    ax.add_line(line)
            else:
                print('invalid cell number, left blank')
    plt.axis('scaled')
    plt.axis('off')
    # plt.savefig("test.png", bbox_inches='tight')
    # plt.show()


def draw_vertex_rep(lattice, ax):
    "draw square grid"
    Nx, Ny = np.shape(lattice)
    lc = 'black'
    for i, row in enumerate(lattice):
        x0 = 2 * i
        verline = plt.Line2D((x0 + 1, x0 + 1), (0, 2 * Ny), lw=1, color=lc)
        ax.add_line(verline)
        for j, cell in enumerate(row):
            y0 = 2 * j
            if i == 0:
                horline = plt.Line2D((0, 2 * Nx), (y0 + 1, y0 + 1), lw=1, color=lc)
                ax.add_line(horline)
            if cell == 0:
                line1 = plt.Line2D((x0 + 1, x0), (y0 + 1, y0 + 2), lw=1, color=lc)
                ax.add_line(line1)
            elif cell == 1:
                line1 = plt.Line2D((x0 + 1, x0 + 2), (y0 + 1, y0 + 2), lw=1, color=lc)
                ax.add_line(line1)
            elif cell == 2:
                line1 = plt.Line2D((x0 + 1, x0 + 2), (y0 + 1, y0), lw=1, color=lc)
                ax.add_line(line1)
            elif cell == 3:
                line1 = plt.Line2D((x0 + 1, x0), (y0 + 1, y0), lw=1, color=lc)
                ax.add_line(line1)
            elif cell == -1:
                "no line"
            else:
                print('invalid cell number, left blank. Cell coordinates: ({:d}, {:d})'.format(i, j))
    plt.axis('scaled')
    plt.axis('off')


def draw_lines(config, modes, tiling, ax):
    Nx, Ny = np.shape(config)
    for cols in range(tiling[0]):
        xoffset = 2 * cols * np.shape(config)[0]
        for rows in range(tiling[1]):
            yoffset = 2 * rows * np.shape(config)[1]
            for i in range(np.shape(modes)[0]):
                x0 = 2 * (i - 1) + 1 + xoffset
                configind = (i - 1) % Nx
                if modes[i] == 'inside':
                    line = plt.Line2D((np.maximum(x0, 0), np.minimum(x0 + 2, Nx * 2 * tiling[0])),
                                      (2 + yoffset, 2 + yoffset), lw=14, color='yellow')
                    ax.add_line(line)
                if modes[i] == 'inedge':
                    for y in range(Ny):
                        if config[configind, y] == (y + 1):
                            line = plt.Line2D((np.maximum(x0, 0), np.minimum(x0 + 2, Nx * 2 * tiling[0])),
                                              (1 + yoffset + 2 * y, 1 + yoffset + 2 * y), lw=14, color='purple')
                            ax.add_line(line)
                        # if config[configind, y]==2:
                        #     line = plt.Line2D((np.maximum(x0, 0), np.minimum(x0+2, Nx*2*tiling[0])), (3+yoffset, 3+yoffset), lw=14, color='purple')
                        #     ax.add_line(line)
                if modes[i] == 'inoutedge':
                    if config[configind, 0]:
                        line = plt.Line2D((np.maximum(x0, 0), np.minimum(x0 + 2, Nx * 2 * tiling[0])),
                                          (1 + yoffset, 1 + yoffset), lw=14, color='purple')
                        ax.add_line(line)
                        line2 = plt.Line2D((np.maximum(x0, 0), np.minimum(x0 + 2, Nx * 2 * tiling[0])),
                                           (3 + yoffset, 3 + yoffset), lw=14, color='pink')
                        ax.add_line(line2)
                    else:
                        line = plt.Line2D((np.maximum(x0, 0), np.minimum(x0 + 2, Nx * 2 * tiling[0])),
                                          (3 + yoffset, 3 + yoffset), lw=14, color='purple')
                        ax.add_line(line)
                        line2 = plt.Line2D((np.maximum(x0, 0), np.minimum(x0 + 2, Nx * 2 * tiling[0])),
                                           (1 + yoffset, 1 + yoffset), lw=14, color='pink')
                        ax.add_line(line2)
                if modes[i] == 'outedge':
                    for y in range(Ny):
                        # if config[configind, 1]==1:
                        #     line = plt.Line2D((np.maximum(x0, 0), np.minimum(x0+2, Nx*2*tiling[0])), (3+yoffset, 3+yoffset), lw=14, color='pink')
                        #     ax.add_line(line)
                        if config[configind, y] == 2 - y:
                            line = plt.Line2D((np.maximum(x0, 0), np.minimum(x0 + 2, Nx * 2 * tiling[0])),
                                              (1 + yoffset + 2 * y, 1 + yoffset + 2 * y), lw=14, color='pink')
                            ax.add_line(line)
                if modes[i] == 'diag':
                    line = plt.Line2D((np.maximum(x0, 0), np.minimum(x0 + 2, Nx * 2 * tiling[0])),
                                      (2 + yoffset, 2 + yoffset), lw=14, color='skyblue')
                    ax.add_line(line)
                if modes[i] == 'blank':
                    "keep empty"
    return


def dxdy_arrow_edge(edge):
    if edge == 1:
        'left'
        dx = -1
        dy = 0
    elif edge == 2:
        'up'
        dx = 0
        dy = 1
    elif edge == 3:
        'right'
        dx = 1
        dy = 0
    elif edge == 4:
        'down'
        dx = 0
        dy = -1
    elif edge == 5:
        'leftup'
        dx = -1
        dy = 1
    elif edge == 6:
        'rightup'
        dx = 1
        dy = 1
    elif edge == 7:
        'rightdown'
        dx = 1
        dy = -1
    elif edge == 8:
        'leftdown'
        dx = -1
        dy = -1
    else:
        print('invalid edge')
    return dx, dy


def draw_arrow(x0, y0, diaghorvert, edge, ax, inout=0, n_arrows=1):
    'draw arrow for cell with origin at (x0, y0), if diaghorvert is 0, +1, +2 draw arrows diag, hor or vert respectively'
    'edge tells which edge to draw on {1, 2, 3, 4, 5, 6, 7, 8} = {left, up, right, down, leftup, rightup, rightdown, leftdown}'
    'if inout is 0 draw arrows outgoing, if 1 draw ingoing, n_arrows gives number of arrows to draw'
    headlength = 0.3
    headwidth = 0.15
    fc = 'yellow'
    ec = 'black'
    if diaghorvert == 0:
        "draw diagonal"
        dx0, dy0 = dxdy_arrow_edge(edge)
        x0t = x0 + inout * 2 * dx0
        y0t = y0 + inout * 2 * dy0
        for n in range(n_arrows):
            arr = n_arrows - n - 1
            dx = (1 - 2 * inout) * (dx0 + dx0 * ((-n_arrows * 0.25 * 0.5 * headlength) + arr * 0.25 * headlength))
            dy = (1 - 2 * inout) * (dy0 + dy0 * ((-n_arrows * 0.25 * 0.5 * headlength) + arr * 0.25 * headlength))
            ax.arrow(x0t, y0t, dx, dy, head_width=headwidth, head_length=headlength, fc=fc, ec=ec)
    elif diaghorvert == 1:
        "draw horizontal"
        dx0, dy0 = dxdy_arrow_edge(edge)
        x0t = x0 + inout * 2 * dx0
        y0t = y0 + inout * 2 * dy0
        for n in range(n_arrows):
            arr = n_arrows - n - 1
            dx = (1 - 2 * inout) * (dx0 + dx0 * ((-n_arrows * 0.5 * headlength) + arr * headlength))
            dy = (1 - 2 * inout) * dy0
            ax.arrow(x0t, y0t, dx, dy, head_width=headwidth, head_length=headlength, fc=fc, ec=ec)
    elif diaghorvert == 2:
        "draw vertical"
        dx0, dy0 = dxdy_arrow_edge(edge)
        x0t = x0 + inout * 2 * dx0
        y0t = y0 + inout * 2 * dy0
        for n in range(n_arrows):
            arr = n_arrows - n - 1
            dx = (1 - 2 * inout) * dx0
            dy = (1 - 2 * inout) * (dy0 + dy0 * ((-n_arrows * 0.5 * headlength) + arr * headlength))
            ax.arrow(x0t, y0t, dx, dy, head_width=headwidth, head_length=headlength, fc=fc, ec=ec)
    return


def draw_triangle(x0, y0, diaghorvert, edge, ax, inout, n_arrows, zorder):
    'draw arrow for cell with origin at (x0, y0), if diaghorvert is 0, +1, +2 draw arrows diag, hor or vert respectively'
    'edge tells which edge to draw on {1, 2, 3, 4, 5, 6, 7, 8} = {left, up, right, down, leftup, rightup, rightdown, leftdown}'
    'if inout is 0 draw arrows outgoing, if 1 draw ingoing, n_arrows gives number of arrows to draw'
    headlength = 0.3
    headwidth = 0.15
    fc = 'yellow'
    ec = 'black'
    if diaghorvert == 0:
        "draw diagonal"
        dx0, dy0 = dxdy_arrow_edge(edge)
        for n in range(n_arrows):
            c0 = np.array([x0, y0]) + np.array([dx0, dy0]) - (n_arrows - 2 * n) * 0.5 * (
                        1 / np.sqrt(2)) * headlength * np.array([dx0, dy0])
            c1 = np.array([x0, y0]) + np.array([dx0, dy0]) - (n_arrows - 2 - 2 * n) * 0.5 * (
                        1 / np.sqrt(2)) * headlength * np.array([dx0, dy0])
            if inout == 0:
                "outgoing, c0 is closest to centre"
                x1 = c0 + np.array([dx0, -dy0]) * (1 / np.sqrt(2)) * 0.5 * headwidth
                x2 = c0 + np.array([-dx0, dy0]) * (1 / np.sqrt(2)) * 0.5 * headwidth
                t1 = plt.Polygon([x1, x2, c1], fc=fc, ec=ec, zorder=zorder)
                ax.add_patch(t1)
            if inout == 1:
                x1 = c1 + np.array([dx0, -dy0]) * (1 / np.sqrt(2)) * 0.5 * headwidth
                x2 = c1 + np.array([-dx0, dy0]) * (1 / np.sqrt(2)) * 0.5 * headwidth
                t1 = plt.Polygon((x1, x2, c0), fc=fc, ec=ec, zorder=zorder)
                ax.add_patch(t1)
    elif diaghorvert == 1:
        "draw horizontal"
        dx0, dy0 = dxdy_arrow_edge(edge)
        for n in range(n_arrows):
            c0 = np.array([x0, y0]) + np.array([dx0, dy0]) - (n_arrows - 2 * n) * 0.5 * headlength * np.array(
                [dx0, 0])
            c1 = np.array([x0, y0]) + np.array([dx0, dy0]) - (
                    n_arrows - 2 - 2 * n) * 0.5 * headlength * np.array([dx0, 0])
            if inout == 0:
                "outgoing, c0 is closest to centre"
                x1 = c0 + np.array([0, -dx0]) * 0.5 * headwidth
                x2 = c0 + np.array([-0, dx0]) * 0.5 * headwidth
                t1 = plt.Polygon([x1, x2, c1], fc=fc, ec=ec, zorder=zorder)
                ax.add_patch(t1)
            if inout == 1:
                x1 = c1 + np.array([0, -dx0]) * 0.5 * headwidth
                x2 = c1 + np.array([-0, dx0]) * 0.5 * headwidth
                t1 = plt.Polygon((x1, x2, c0), fc=fc, ec=ec, zorder=zorder)
                ax.add_patch(t1)
    elif diaghorvert == 2:
        "draw vertical"
        dx0, dy0 = dxdy_arrow_edge(edge)
        for n in range(n_arrows):
            c0 = np.array([x0, y0]) + np.array([dx0, dy0]) - (n_arrows - 2 * n) * 0.5 * headlength * np.array(
                [0, dy0])
            c1 = np.array([x0, y0]) + np.array([dx0, dy0]) - (
                    n_arrows - 2 - 2 * n) * 0.5 * headlength * np.array([0, dy0])
            if inout == 0:
                "outgoing, c0 is closest to centre"
                x1 = c0 + np.array([dy0, -0]) * 0.5 * headwidth
                x2 = c0 + np.array([-dy0, 0]) * 0.5 * headwidth
                t1 = plt.Polygon([x1, x2, c1], fc=fc, ec=ec, zorder=zorder)
                ax.add_patch(t1)
            if inout == 1:
                x1 = c1 + np.array([dy0, -0]) * 0.5 * headwidth
                x2 = c1 + np.array([-dy0, 0]) * 0.5 * headwidth
                t1 = plt.Polygon((x1, x2, c0), fc=fc, ec=ec, zorder=zorder)
                ax.add_patch(t1)
    return


def draw_horizontal_connections():
    f, ax = plt.subplots(figsize=(4, 2))
    plt.gca().set_aspect('equal', adjustable='box')
    lattice = np.array([[1], [0]])
    draw_vertex_rep(lattice, ax)
    draw_triangle(1, 1, 1, 1, ax, 1, 1, 5)
    draw_triangle(1, 1, 1, 3, ax, 1, 3, 5)
    draw_triangle(3, 1, 1, 3, ax, 0, 1, 5)
    draw_triangle(1, 1, 2, 4, ax, 0, 2, 5)
    draw_triangle(3, 1, 2, 4, ax, 1, 2, 5)
    draw_triangle(1, 2, 1, 3, ax, 0, 2, 5)
    xlims = ax.get_xlim()
    ylims = ax.get_ylim()
    plt.xlim([xlims[0] - 0.55, xlims[1] + 0.55])
    plt.ylim([ylims[0] - 0.55, ylims[1] + 0.55])
    plt.savefig(r'.\\results\\modescaling\\LineModes\\Connections_Hor_up0.pdf', transparent=True)
    plt.savefig(r'.\\results\\modescaling\\LineModes\\Connections_Hor_up0.png', dpi=400, transparent=True)
    plt.show()
    plt.close()

    f, ax = plt.subplots(figsize=(4, 2))
    plt.gca().set_aspect('equal', adjustable='box')
    lattice = np.array([[1], [0]])
    draw_vertex_rep(lattice, ax)
    "input args draw_triangle: (x0, y0, diaghorver, edge, ax, outin, n_arrows, zorder)"
    "edge: {1, 2, 3, 4, 5, 6, 7, 8} = {left, up, right, down, leftup, rightup, rightdown, leftdown}"
    draw_triangle(1, 1, 1, 1, ax, 0, 1, 5)
    draw_triangle(1, 1, 1, 3, ax, 1, 3, 5)
    draw_triangle(3, 1, 1, 3, ax, 1, 1, 5)
    draw_triangle(1, 1, 2, 2, ax, 1, 3, 5)
    draw_triangle(1, 1, 2, 4, ax, 0, 1, 5)
    draw_triangle(3, 1, 2, 2, ax, 0, 3, 5)
    draw_triangle(3, 1, 2, 4, ax, 1, 1, 5)
    draw_triangle(1, 2, 1, 3, ax, 0, 4, 5)
    xlims = ax.get_xlim()
    ylims = ax.get_ylim()
    plt.xlim([xlims[0] - 0.55, xlims[1] + 0.55])
    plt.ylim([ylims[0] - 0.55, ylims[1] + 0.55])
    plt.savefig(r'.\\results\\modescaling\\LineModes\\Connections_Hor_up_alt.pdf', transparent=True)
    plt.savefig(r'.\\results\\modescaling\\LineModes\\Connections_Hor_up_alt.png', dpi=400, transparent=True)
    plt.show()
    plt.close()

    f, ax = plt.subplots(figsize=(4, 2))
    plt.gca().set_aspect('equal', adjustable='box')
    lattice = np.array([[1], [0]])
    draw_vertex_rep(lattice, ax)
    "input args draw_triangle: (x0, y0, diaghorver, edge, ax, outin, n_arrows, zorder)"
    "edge: {1, 2, 3, 4, 5, 6, 7, 8} = {left, up, right, down, leftup, rightup, rightdown, leftdown}"
    draw_triangle(1, 1, 1, 1, ax, 1, 3, 5)
    draw_triangle(1, 1, 1, 3, ax, 1, 1, 5)
    draw_triangle(3, 1, 1, 3, ax, 0, 3, 5)
    draw_triangle(1, 1, 2, 4, ax, 0, 2, 5)
    draw_triangle(1, 1, 2, 2, ax, 0, 4, 5)
    draw_triangle(3, 1, 2, 4, ax, 1, 2, 5)
    draw_triangle(3, 1, 2, 2, ax, 1, 4, 5)
    draw_triangle(1, 2, 1, 3, ax, 1, 2, 5)
    xlims = ax.get_xlim()
    ylims = ax.get_ylim()
    plt.xlim([xlims[0] - 0.55, xlims[1] + 0.55])
    plt.ylim([ylims[0] - 0.55, ylims[1] + 0.55])
    plt.savefig(r'.\\results\\modescaling\\LineModes\\Connections_Hor_up1.pdf', transparent=True)
    plt.savefig(r'.\\results\\modescaling\\LineModes\\Connections_Hor_up1.png', dpi=400, transparent=True)
    plt.show()
    plt.close()

    f, ax = plt.subplots(figsize=(4, 2))
    plt.gca().set_aspect('equal', adjustable='box')
    lattice = np.array([[1], [0]])
    draw_vertex_rep(lattice, ax)
    "input args draw_triangle: (x0, y0, diaghorver, edge, ax, outin, n_arrows, zorder)"
    "edge: {1, 2, 3, 4, 5, 6, 7, 8} = {left, up, right, down, leftup, rightup, rightdown, leftdown}"
    draw_triangle(1, 1, 1, 1, ax, 0, 1, 5)
    draw_triangle(1, 1, 1, 3, ax, 0, 1, 5)
    draw_triangle(1, 1, 2, 2, ax, 1, 1, 5)
    draw_triangle(1, 1, 2, 4, ax, 1, 1, 5)
    draw_triangle(3, 1, 2, 2, ax, 0, 1, 5)
    draw_triangle(3, 1, 2, 4, ax, 0, 1, 5)
    draw_triangle(3, 1, 1, 3, ax, 1, 1, 5)
    xlims = ax.get_xlim()
    ylims = ax.get_ylim()
    plt.xlim([xlims[0] - 0.55, xlims[1] + 0.55])
    plt.ylim([ylims[0] - 0.55, ylims[1] + 0.55])
    plt.savefig(r'.\\results\\modescaling\\LineModes\\Connections_Hor_up_rotsqr.pdf', transparent=True)
    plt.savefig(r'.\\results\\modescaling\\LineModes\\Connections_Hor_up_rotsqr.png', dpi=400, transparent=True)
    plt.show()
    plt.close()

    f, ax = plt.subplots(figsize=(4, 2))
    plt.gca().set_aspect('equal', adjustable='box')
    lattice = np.array([[2], [3]])
    draw_vertex_rep(lattice, ax)
    "input args draw_triangle: (x0, y0, diaghorver, edge, ax, outin, n_arrows, zorder)"
    "edge: {1, 2, 3, 4, 5, 6, 7, 8} = {left, up, right, down, leftup, rightup, rightdown, leftdown}"
    draw_triangle(1, 1, 1, 1, ax, 1, 1, 5)
    draw_triangle(1, 1, 1, 3, ax, 0, 1, 5)
    draw_triangle(3, 1, 1, 3, ax, 0, 1, 5)
    draw_triangle(1, 1, 2, 4, ax, 0, 2, 5)
    draw_triangle(3, 1, 2, 4, ax, 1, 2, 5)
    draw_triangle(1, 0, 1, 3, ax, 1, 2, 5)
    xlims = ax.get_xlim()
    ylims = ax.get_ylim()
    plt.xlim([xlims[0] - 0.55, xlims[1] + 0.55])
    plt.ylim([ylims[0] - 0.55, ylims[1] + 0.55])
    plt.savefig(r'.\\results\\modescaling\\LineModes\\Connections_Hor_down0.pdf', transparent=True)
    plt.savefig(r'.\\results\\modescaling\\LineModes\\Connections_Hor_down0.png', dpi=400, transparent=True)
    plt.show()
    plt.close()

    f, ax = plt.subplots(figsize=(4, 2))
    plt.gca().set_aspect('equal', adjustable='box')
    lattice = np.array([[2], [3]])
    draw_vertex_rep(lattice, ax)
    "input args draw_triangle: (x0, y0, diaghorver, edge, ax, outin, n_arrows, zorder)"
    "edge: {1, 2, 3, 4, 5, 6, 7, 8} = {left, up, right, down, leftup, rightup, rightdown, leftdown}"
    draw_triangle(1, 1, 1, 1, ax, 0, 1, 5)
    draw_triangle(1, 1, 1, 3, ax, 1, 3, 5)
    draw_triangle(3, 1, 1, 3, ax, 1, 1, 5)
    draw_triangle(1, 1, 2, 4, ax, 1, 3, 5)
    draw_triangle(1, 1, 2, 2, ax, 0, 1, 5)
    draw_triangle(3, 1, 2, 4, ax, 0, 3, 5)
    draw_triangle(3, 1, 2, 2, ax, 1, 1, 5)
    draw_triangle(1, 0, 1, 3, ax, 0, 4, 5)
    xlims = ax.get_xlim()
    ylims = ax.get_ylim()
    plt.xlim([xlims[0] - 0.55, xlims[1] + 0.55])
    plt.ylim([ylims[0] - 0.55, ylims[1] + 0.55])
    plt.savefig(r'.\\results\\modescaling\\LineModes\\Connections_Hor_down_alt.pdf', transparent=True)
    plt.savefig(r'.\\results\\modescaling\\LineModes\\Connections_Hor_down_alt.png', dpi=400, transparent=True)
    plt.show()
    plt.close()

    f, ax = plt.subplots(figsize=(4, 2))
    plt.gca().set_aspect('equal', adjustable='box')
    lattice = np.array([[2], [3]])
    draw_vertex_rep(lattice, ax)
    "input args draw_triangle: (x0, y0, diaghorver, edge, ax, outin, n_arrows, zorder)"
    "edge: {1, 2, 3, 4, 5, 6, 7, 8} = {left, up, right, down, leftup, rightup, rightdown, leftdown}"
    draw_triangle(1, 1, 1, 1, ax, 1, 3, 5)
    draw_triangle(1, 1, 1, 3, ax, 1, 5, 5)
    draw_triangle(3, 1, 1, 3, ax, 0, 3, 5)
    draw_triangle(1, 1, 2, 4, ax, 0, 2, 5)
    draw_triangle(1, 1, 2, 2, ax, 0, 4, 5)
    draw_triangle(3, 1, 2, 4, ax, 1, 2, 5)
    draw_triangle(3, 1, 2, 2, ax, 1, 4, 5)
    draw_triangle(1, 0, 1, 3, ax, 0, 2, 5)
    xlims = ax.get_xlim()
    ylims = ax.get_ylim()
    plt.xlim([xlims[0] - 0.55, xlims[1] + 0.55])
    plt.ylim([ylims[0] - 0.55, ylims[1] + 0.55])
    plt.savefig(r'.\\results\\modescaling\\LineModes\\Connections_Hor_down1.pdf', transparent=True)
    plt.savefig(r'.\\results\\modescaling\\LineModes\\Connections_Hor_down1.png', dpi=400, transparent=True)
    plt.show()
    plt.close()

    f, ax = plt.subplots(figsize=(4, 2))
    plt.gca().set_aspect('equal', adjustable='box')
    lattice = np.array([[2], [3]])
    draw_vertex_rep(lattice, ax)
    "input args draw_triangle: (x0, y0, diaghorver, edge, ax, outin, n_arrows, zorder)"
    "edge: {1, 2, 3, 4, 5, 6, 7, 8} = {left, up, right, down, leftup, rightup, rightdown, leftdown}"
    draw_triangle(1, 1, 1, 1, ax, 0, 1, 5)
    draw_triangle(1, 1, 1, 3, ax, 0, 1, 5)
    draw_triangle(1, 1, 2, 2, ax, 1, 1, 5)
    draw_triangle(1, 1, 2, 4, ax, 1, 1, 5)
    draw_triangle(3, 1, 2, 2, ax, 0, 1, 5)
    draw_triangle(3, 1, 2, 4, ax, 0, 1, 5)
    draw_triangle(3, 1, 1, 3, ax, 1, 1, 5)
    xlims = ax.get_xlim()
    ylims = ax.get_ylim()
    plt.xlim([xlims[0] - 0.55, xlims[1] + 0.55])
    plt.ylim([ylims[0] - 0.55, ylims[1] + 0.55])
    plt.savefig(r'.\\results\\modescaling\\LineModes\\Connections_Hor_down_rotsqr.pdf', transparent=True)
    plt.savefig(r'.\\results\\modescaling\\LineModes\\Connections_Hor_down_rotsqr.png', dpi=400, transparent=True)
    plt.show()
    plt.close()
    return 0


def draw_vertical_connections():
    f, ax = plt.subplots(figsize=(2, 4))
    plt.gca().set_aspect('equal', adjustable='box')
    lattice = np.array([[0, 3]])
    draw_vertex_rep(lattice, ax)
    draw_triangle(1, 1, 1, 1, ax, 0, 1, 5)
    draw_triangle(1, 1, 1, 3, ax, 1, 1, 5)
    draw_triangle(1, 3, 1, 1, ax, 1, 1, 5)
    draw_triangle(1, 3, 1, 3, ax, 0, 1, 5)
    draw_triangle(1, 1, 2, 2, ax, 0, 2, 5)
    draw_triangle(0, 1, 2, 2, ax, 1, 2, 5)
    xlims = ax.get_xlim()
    ylims = ax.get_ylim()
    plt.xlim([xlims[0] - 0.55, xlims[1] + 0.55])
    plt.ylim([ylims[0] - 0.55, ylims[1] + 0.55])
    plt.savefig(r'.\\results\\modescaling\\LineModes\\Connections_Ver_left0.pdf', transparent=True)
    plt.savefig(r'.\\results\\modescaling\\LineModes\\Connections_Ver_left0.png', dpi=400, transparent=True)
    plt.show()
    plt.close()

    f, ax = plt.subplots(figsize=(2, 4))
    plt.gca().set_aspect('equal', adjustable='box')
    lattice = np.array([[0, 3]])
    draw_vertex_rep(lattice, ax)
    "input args draw_triangle: (x0, y0, diaghorver, edge, ax, outin, n_arrows, zorder)"
    "edge: {1, 2, 3, 4, 5, 6, 7, 8} = {left, up, right, down, leftup, rightup, rightdown, leftdown}"
    draw_triangle(1, 1, 1, 1, ax, 0, 1, 5)
    draw_triangle(1, 1, 1, 3, ax, 0, 1, 5)
    draw_triangle(1, 3, 1, 1, ax, 1, 1, 5)
    draw_triangle(1, 3, 1, 3, ax, 1, 1, 5)
    draw_triangle(1, 1, 2, 2, ax, 1, 1, 5)
    draw_triangle(1, 1, 2, 4, ax, 1, 1, 5)
    draw_triangle(1, 3, 2, 2, ax, 0, 1, 5)
    xlims = ax.get_xlim()
    ylims = ax.get_ylim()
    plt.xlim([xlims[0] - 0.55, xlims[1] + 0.55])
    plt.ylim([ylims[0] - 0.55, ylims[1] + 0.55])
    plt.savefig(r'.\\results\\modescaling\\LineModes\\Connections_Ver_left_rotsqr.pdf', transparent=True)
    plt.savefig(r'.\\results\\modescaling\\LineModes\\Connections_Ver_left_rotsqr.png', dpi=400, transparent=True)
    plt.show()
    plt.close()

    f, ax = plt.subplots(figsize=(2, 4))
    plt.gca().set_aspect('equal', adjustable='box')
    lattice = np.array([[0, 3]])
    draw_vertex_rep(lattice, ax)
    "input args draw_triangle: (x0, y0, diaghorver, edge, ax, outin, n_arrows, zorder)"
    "edge: {1, 2, 3, 4, 5, 6, 7, 8} = {left, up, right, down, leftup, rightup, rightdown, leftdown}"
    draw_triangle(1, 1, 1, 3, ax, 0, 1, 5)
    draw_triangle(1, 1, 1, 1, ax, 1, 3, 5)
    draw_triangle(1, 3, 1, 1, ax, 0, 3, 5)
    draw_triangle(1, 3, 1, 3, ax, 1, 1, 5)
    draw_triangle(1, 1, 2, 2, ax, 1, 3, 5)
    draw_triangle(1, 1, 2, 4, ax, 0, 1, 5)
    draw_triangle(1, 3, 2, 2, ax, 1, 1, 5)
    draw_triangle(0, 1, 2, 2, ax, 0, 4, 5)
    xlims = ax.get_xlim()
    ylims = ax.get_ylim()
    plt.xlim([xlims[0] - 0.55, xlims[1] + 0.55])
    plt.ylim([ylims[0] - 0.55, ylims[1] + 0.55])
    plt.savefig(r'.\\results\\modescaling\\LineModes\\Connections_Ver_left_alt.pdf', transparent=True)
    plt.savefig(r'.\\results\\modescaling\\LineModes\\Connections_Ver_left_alt.png', dpi=400, transparent=True)
    plt.show()
    plt.close()

    f, ax = plt.subplots(figsize=(2, 4))
    plt.gca().set_aspect('equal', adjustable='box')
    lattice = np.array([[0, 3]])
    draw_vertex_rep(lattice, ax)
    "input args draw_triangle: (x0, y0, diaghorver, edge, ax, outin, n_arrows, zorder)"
    "edge: {1, 2, 3, 4, 5, 6, 7, 8} = {left, up, right, down, leftup, rightup, rightdown, leftdown}"
    draw_triangle(1, 1, 1, 1, ax, 0, 1, 5)
    draw_triangle(1, 1, 1, 3, ax, 0, 3, 5)
    draw_triangle(1, 3, 1, 1, ax, 1, 1, 5)
    draw_triangle(1, 3, 1, 3, ax, 1, 3, 5)
    draw_triangle(1, 1, 2, 2, ax, 1, 4, 5)
    draw_triangle(1, 1, 2, 4, ax, 1, 2, 5)
    draw_triangle(1, 3, 2, 2, ax, 0, 2, 5)
    draw_triangle(0, 1, 2, 2, ax, 0, 2, 5)
    xlims = ax.get_xlim()
    ylims = ax.get_ylim()
    plt.xlim([xlims[0] - 0.55, xlims[1] + 0.55])
    plt.ylim([ylims[0] - 0.55, ylims[1] + 0.55])
    plt.savefig(r'.\\results\\modescaling\\LineModes\\Connections_Ver_left1.pdf', transparent=True)
    plt.savefig(r'.\\results\\modescaling\\LineModes\\Connections_Ver_left1.png', dpi=400, transparent=True)
    plt.show()
    plt.close()

    f, ax = plt.subplots(figsize=(2, 4))
    plt.gca().set_aspect('equal', adjustable='box')
    lattice = np.array([[1, 2]])
    draw_vertex_rep(lattice, ax)
    "input args draw_triangle: (x0, y0, diaghorver, edge, ax, outin, n_arrows, zorder)"
    "edge: {1, 2, 3, 4, 5, 6, 7, 8} = {left, up, right, down, leftup, rightup, rightdown, leftdown}"
    draw_triangle(1, 1, 1, 1, ax, 0, 1, 5)
    draw_triangle(1, 1, 1, 3, ax, 1, 1, 5)
    draw_triangle(1, 3, 1, 1, ax, 1, 1, 5)
    draw_triangle(1, 3, 1, 3, ax, 0, 1, 5)
    draw_triangle(1, 1, 2, 2, ax, 1, 2, 5)
    draw_triangle(2, 1, 2, 2, ax, 0, 2, 5)
    xlims = ax.get_xlim()
    ylims = ax.get_ylim()
    plt.xlim([xlims[0] - 0.55, xlims[1] + 0.55])
    plt.ylim([ylims[0] - 0.55, ylims[1] + 0.55])
    plt.savefig(r'.\\results\\modescaling\\LineModes\\Connections_Ver_right0.pdf', transparent=True)
    plt.savefig(r'.\\results\\modescaling\\LineModes\\Connections_Ver_right0.png', dpi=400, transparent=True)
    plt.show()
    plt.close()

    f, ax = plt.subplots(figsize=(2, 4))
    plt.gca().set_aspect('equal', adjustable='box')
    lattice = np.array([[1, 2]])
    draw_vertex_rep(lattice, ax)
    "input args draw_triangle: (x0, y0, diaghorver, edge, ax, outin, n_arrows, zorder)"
    "edge: {1, 2, 3, 4, 5, 6, 7, 8} = {left, up, right, down, leftup, rightup, rightdown, leftdown}"
    draw_triangle(1, 1, 1, 1, ax, 0, 1, 5)
    draw_triangle(1, 1, 1, 3, ax, 1, 3, 5)
    draw_triangle(1, 3, 1, 3, ax, 0, 3, 5)
    draw_triangle(1, 3, 1, 1, ax, 1, 1, 5)
    draw_triangle(1, 1, 2, 2, ax, 1, 3, 5)
    draw_triangle(1, 1, 2, 4, ax, 0, 1, 5)
    draw_triangle(1, 3, 2, 2, ax, 1, 1, 5)
    draw_triangle(2, 1, 2, 2, ax, 0, 4, 5)
    xlims = ax.get_xlim()
    ylims = ax.get_ylim()
    plt.xlim([xlims[0] - 0.55, xlims[1] + 0.55])
    plt.ylim([ylims[0] - 0.55, ylims[1] + 0.55])
    plt.savefig(r'.\\results\\modescaling\\LineModes\\Connections_Ver_right_alt.pdf', transparent=True)
    plt.savefig(r'.\\results\\modescaling\\LineModes\\Connections_Ver_right_alt.png', dpi=400, transparent=True)
    plt.show()
    plt.close()

    f, ax = plt.subplots(figsize=(2, 4))
    plt.gca().set_aspect('equal', adjustable='box')
    lattice = np.array([[1, 2]])
    draw_vertex_rep(lattice, ax)
    "input args draw_triangle: (x0, y0, diaghorver, edge, ax, outin, n_arrows, zorder)"
    "edge: {1, 2, 3, 4, 5, 6, 7, 8} = {left, up, right, down, leftup, rightup, rightdown, leftdown}"
    draw_triangle(1, 1, 1, 1, ax, 0, 1, 5)
    draw_triangle(1, 1, 1, 3, ax, 0, 1, 5)
    draw_triangle(1, 3, 1, 1, ax, 1, 1, 5)
    draw_triangle(1, 3, 1, 3, ax, 1, 1, 5)
    draw_triangle(1, 1, 2, 2, ax, 1, 1, 5)
    draw_triangle(1, 1, 2, 4, ax, 1, 1, 5)
    draw_triangle(1, 3, 2, 2, ax, 0, 1, 5)
    xlims = ax.get_xlim()
    ylims = ax.get_ylim()
    plt.xlim([xlims[0] - 0.55, xlims[1] + 0.55])
    plt.ylim([ylims[0] - 0.55, ylims[1] + 0.55])
    plt.savefig(r'.\\results\\modescaling\\LineModes\\Connections_Ver_right_rotsqr.pdf', transparent=True)
    plt.savefig(r'.\\results\\modescaling\\LineModes\\Connections_Ver_right_rotsqr.png', dpi=400, transparent=True)
    plt.show()
    plt.close()

    f, ax = plt.subplots(figsize=(2, 4))
    plt.gca().set_aspect('equal', adjustable='box')
    lattice = np.array([[1, 2]])
    draw_vertex_rep(lattice, ax)
    "input args draw_triangle: (x0, y0, diaghorver, edge, ax, outin, n_arrows, zorder)"
    "edge: {1, 2, 3, 4, 5, 6, 7, 8} = {left, up, right, down, leftup, rightup, rightdown, leftdown}"
    draw_triangle(1, 1, 1, 1, ax, 0, 1, 5)
    draw_triangle(1, 1, 1, 3, ax, 0, 3, 5)
    draw_triangle(1, 3, 1, 1, ax, 1, 1, 5)
    draw_triangle(1, 3, 1, 3, ax, 1, 3, 5)
    draw_triangle(1, 1, 2, 4, ax, 1, 2, 5)
    draw_triangle(1, 3, 2, 2, ax, 0, 2, 5)
    draw_triangle(2, 1, 2, 2, ax, 1, 2, 5)
    xlims = ax.get_xlim()
    ylims = ax.get_ylim()
    plt.xlim([xlims[0] - 0.55, xlims[1] + 0.55])
    plt.ylim([ylims[0] - 0.55, ylims[1] + 0.55])
    plt.savefig(r'.\\results\\modescaling\\LineModes\\Connections_Ver_right1.pdf', transparent=True)
    plt.savefig(r'.\\results\\modescaling\\LineModes\\Connections_Ver_right1.png', dpi=400, transparent=True)
    plt.show()
    plt.close()
    return 0


def draw_diagonal_connections():
    f, ax = plt.subplots(figsize=(4, 4))
    plt.gca().set_aspect('equal', adjustable='box')
    lattice = np.array([[-1, 2], [0, -1]])
    draw_vertex_rep(lattice, ax)
    "input args draw_triangle: (x0, y0, diaghorver, edge, ax, outin, n_arrows, zorder)"
    "edge: {1, 2, 3, 4, 5, 6, 7, 8} = {left, up, right, down, leftup, rightup, rightdown, leftdown}"
    draw_triangle(1, 3, 1, 1, ax, 1, 1, 5)
    draw_triangle(1, 3, 1, 3, ax, 0, 1, 5)
    draw_triangle(3, 1, 1, 1, ax, 1, 1, 5)
    draw_triangle(3, 1, 1, 3, ax, 0, 1, 5)
    draw_triangle(1, 3, 2, 4, ax, 0, 2, 5)
    draw_triangle(3, 1, 2, 2, ax, 1, 2, 5)
    draw_triangle(1, 3, 0, 7, ax, 1, 2, 5)

    xlims = ax.get_xlim()
    ylims = ax.get_ylim()
    plt.xlim([xlims[0] - 0.55, xlims[1] + 0.55])
    plt.ylim([ylims[0] - 0.55, ylims[1] + 0.55])
    plt.savefig(r'.\\results\\modescaling\\LineModes\\Connections_Diag_leftup0.pdf', transparent=True)
    plt.savefig(r'.\\results\\modescaling\\LineModes\\Connections_Diag_leftup0.png', dpi=400, transparent=True)
    # plt.tight_layout()
    plt.show()
    plt.close()

    f, ax = plt.subplots(figsize=(4, 4))
    plt.gca().set_aspect('equal', adjustable='box')
    lattice = np.array([[-1, 2], [0, -1]])
    draw_vertex_rep(lattice, ax)
    "input args draw_triangle: (x0, y0, diaghorver, edge, ax, outin, n_arrows, zorder)"
    "edge: {1, 2, 3, 4, 5, 6, 7, 8} = {left, up, right, down, leftup, rightup, rightdown, leftdown}"
    draw_triangle(1, 3, 1, 1, ax, 1, 1, 5)
    draw_triangle(1, 3, 1, 3, ax, 1, 3, 5)
    draw_triangle(3, 1, 1, 1, ax, 1, 1, 5)
    draw_triangle(3, 1, 1, 3, ax, 1, 3, 5)
    draw_triangle(1, 3, 2, 2, ax, 0, 2, 5)
    draw_triangle(3, 1, 2, 2, ax, 0, 4, 5)
    draw_triangle(3, 1, 2, 4, ax, 0, 2, 5)
    draw_triangle(1, 3, 0, 7, ax, 0, 2, 5)
    xlims = ax.get_xlim()
    ylims = ax.get_ylim()
    plt.xlim([xlims[0] - 0.55, xlims[1] + 0.55])
    plt.ylim([ylims[0] - 0.55, ylims[1] + 0.55])
    # plt.tight_layout()
    plt.savefig(r'.\\results\\modescaling\\LineModes\\Connections_Diag_leftup1.pdf', transparent=True)
    plt.savefig(r'.\\results\\modescaling\\LineModes\\Connections_Diag_leftup1.png', dpi=400, transparent=True)
    plt.show()
    plt.close()

    f, ax = plt.subplots(figsize=(4, 4))
    plt.gca().set_aspect('equal', adjustable='box')
    lattice = np.array([[-1, 2], [0, -1]])
    draw_vertex_rep(lattice, ax)
    "input args draw_triangle: (x0, y0, diaghorver, edge, ax, outin, n_arrows, zorder)"
    "edge: {1, 2, 3, 4, 5, 6, 7, 8} = {left, up, right, down, leftup, rightup, rightdown, leftdown}"
    draw_triangle(1, 3, 1, 1, ax, 0, 1, 5)
    draw_triangle(1, 3, 1, 3, ax, 1, 3, 5)
    draw_triangle(3, 1, 1, 3, ax, 1, 1, 5)
    draw_triangle(3, 1, 1, 1, ax, 0, 3, 5)
    draw_triangle(1, 3, 2, 4, ax, 1, 3, 5)
    draw_triangle(1, 3, 2, 2, ax, 0, 1, 5)
    draw_triangle(3, 1, 2, 2, ax, 0, 3, 5)
    draw_triangle(3, 1, 2, 4, ax, 1, 1, 5)
    draw_triangle(3, 1, 0, 5, ax, 1, 4, 5)
    xlims = ax.get_xlim()
    ylims = ax.get_ylim()
    plt.xlim([xlims[0] - 0.55, xlims[1] + 0.55])
    plt.ylim([ylims[0] - 0.55, ylims[1] + 0.55])
    plt.savefig(r'.\\results\\modescaling\\LineModes\\Connections_Diag_leftup_alt.pdf', transparent=True)
    plt.savefig(r'.\\results\\modescaling\\LineModes\\Connections_Diag_leftup_alt.png', dpi=400, transparent=True)
    plt.show()
    plt.close()

    f, ax = plt.subplots(figsize=(4, 4))
    plt.gca().set_aspect('equal', adjustable='box')
    lattice = np.array([[-1, 2], [0, -1]])
    draw_vertex_rep(lattice, ax)
    "input args draw_triangle: (x0, y0, diaghorver, edge, ax, outin, n_arrows, zorder)"
    "edge: {1, 2, 3, 4, 5, 6, 7, 8} = {left, up, right, down, leftup, rightup, rightdown, leftdown}"

    draw_triangle(1, 3, 1, 1, ax, 1, 1, 5)
    draw_triangle(1, 3, 1, 3, ax, 1, 1, 5)
    draw_triangle(1, 3, 2, 4, ax, 1, 1, 5)
    draw_triangle(1, 3, 2, 2, ax, 0, 1, 5)
    xlims = ax.get_xlim()
    ylims = ax.get_ylim()
    plt.xlim([xlims[0] - 0.55, xlims[1] + 0.55])
    plt.ylim([ylims[0] - 0.55, ylims[1] + 0.55])
    # plt.tight_layout()
    plt.savefig(r'.\\results\\modescaling\\LineModes\\Connections_Diag_leftup_rotsqr.pdf', transparent=True)
    plt.savefig(r'.\\results\\modescaling\\LineModes\\Connections_Diag_leftup_rotsqr.png', dpi=400, transparent=True)
    plt.show()
    plt.close()

    f, ax = plt.subplots(figsize=(4, 4))
    plt.gca().set_aspect('equal', adjustable='box')
    lattice = np.array([[-1, 2], [0, -1]])
    draw_vertex_rep(lattice, ax)
    "input args draw_triangle: (x0, y0, diaghorver, edge, ax, outin, n_arrows, zorder)"
    "edge: {1, 2, 3, 4, 5, 6, 7, 8} = {left, up, right, down, leftup, rightup, rightdown, leftdown}"

    draw_triangle(3, 1, 1, 1, ax, 1, 1, 5)
    draw_triangle(3, 1, 1, 3, ax, 1, 1, 5)
    draw_triangle(3, 1, 2, 4, ax, 1, 1, 5)
    draw_triangle(3, 1, 2, 2, ax, 0, 1, 5)
    xlims = ax.get_xlim()
    ylims = ax.get_ylim()
    plt.xlim([xlims[0] - 0.55, xlims[1] + 0.55])
    plt.ylim([ylims[0] - 0.55, ylims[1] + 0.55])
    # plt.tight_layout()
    plt.savefig(r'.\\results\\modescaling\\LineModes\\Connections_Diag_rightdown_rotsqr.pdf', transparent=True)
    plt.savefig(r'.\\results\\modescaling\\LineModes\\Connections_Diag_rightdown_rotsqr.png', dpi=400, transparent=True)
    plt.show()
    plt.close()

    f, ax = plt.subplots(figsize=(4, 4))
    plt.gca().set_aspect('equal', adjustable='box')
    lattice = np.array([[1, -1], [-1, 3]])
    draw_vertex_rep(lattice, ax)
    "input args draw_triangle: (x0, y0, diaghorver, edge, ax, outin, n_arrows, zorder)"
    "edge: {1, 2, 3, 4, 5, 6, 7, 8} = {left, up, right, down, leftup, rightup, rightdown, leftdown}"
    draw_triangle(1, 1, 1, 1, ax, 1, 1, 5)
    draw_triangle(1, 1, 1, 3, ax, 0, 1, 5)
    draw_triangle(3, 3, 1, 1, ax, 1, 1, 5)
    draw_triangle(3, 3, 1, 3, ax, 0, 1, 5)
    draw_triangle(1, 1, 2, 2, ax, 0, 2, 5)
    draw_triangle(3, 3, 2, 4, ax, 1, 2, 5)
    draw_triangle(1, 1, 0, 6, ax, 1, 2, 5)
    xlims = ax.get_xlim()
    ylims = ax.get_ylim()
    plt.xlim([xlims[0] - 0.55, xlims[1] + 0.55])
    plt.ylim([ylims[0] - 0.55, ylims[1] + 0.55])
    # plt.tight_layout()
    plt.savefig(r'.\\results\\modescaling\\LineModes\\Connections_Diag_rightdown0.pdf', transparent=True)
    plt.savefig(r'.\\results\\modescaling\\LineModes\\Connections_Diag_righdown0.png', dpi=400, transparent=True)
    plt.show()
    plt.close()

    f, ax = plt.subplots(figsize=(4, 4))
    plt.gca().set_aspect('equal', adjustable='box')
    lattice = np.array([[1, -1], [-1, 3]])
    draw_vertex_rep(lattice, ax)
    "input args draw_triangle: (x0, y0, diaghorver, edge, ax, outin, n_arrows, zorder)"
    "edge: {1, 2, 3, 4, 5, 6, 7, 8} = {left, up, right, down, leftup, rightup, rightdown, leftdown}"
    draw_triangle(1, 1, 1, 1, ax, 1, 1, 5)
    draw_triangle(1, 1, 1, 3, ax, 1, 3, 5)
    draw_triangle(3, 3, 1, 1, ax, 1, 1, 5)
    draw_triangle(3, 3, 1, 3, ax, 1, 3, 5)
    draw_triangle(1, 1, 2, 4, ax, 0, 2, 5)
    draw_triangle(3, 3, 2, 2, ax, 0, 2, 5)
    draw_triangle(3, 3, 2, 4, ax, 0, 4, 5)
    draw_triangle(1, 1, 0, 6, ax, 0, 2, 5)
    xlims = ax.get_xlim()
    ylims = ax.get_ylim()
    plt.xlim([xlims[0] - 0.55, xlims[1] + 0.55])
    plt.ylim([ylims[0] - 0.55, ylims[1] + 0.55])
    plt.savefig(r'.\\results\\modescaling\\LineModes\\Connections_Diag_rightdown1.pdf', transparent=True)
    plt.savefig(r'.\\results\\modescaling\\LineModes\\Connections_Diag_righdown1.png', dpi=400, transparent=True)
    # plt.tight_layout()
    plt.show()
    plt.close()

    f, ax = plt.subplots(figsize=(4, 4))
    plt.gca().set_aspect('equal', adjustable='box')
    lattice = np.array([[1, -1], [-1, 3]])
    draw_vertex_rep(lattice, ax)
    "input args draw_triangle: (x0, y0, diaghorver, edge, ax, outin, n_arrows, zorder)"
    "edge: {1, 2, 3, 4, 5, 6, 7, 8} = {left, up, right, down, leftup, rightup, rightdown, leftdown}"
    draw_triangle(1, 1, 1, 1, ax, 0, 1, 5)
    draw_triangle(1, 1, 1, 3, ax, 1, 3, 5)
    draw_triangle(3, 3, 1, 3, ax, 1, 1, 5)
    draw_triangle(3, 3, 1, 1, ax, 0, 3, 5)
    draw_triangle(1, 1, 2, 2, ax, 1, 3, 5)
    draw_triangle(1, 1, 2, 4, ax, 0, 1, 5)
    draw_triangle(3, 3, 2, 4, ax, 0, 3, 5)
    draw_triangle(3, 3, 2, 2, ax, 1, 1, 5)
    draw_triangle(1, 1, 0, 6, ax, 0, 4, 5)
    xlims = ax.get_xlim()
    ylims = ax.get_ylim()
    plt.xlim([xlims[0] - 0.55, xlims[1] + 0.55])
    plt.ylim([ylims[0] - 0.55, ylims[1] + 0.55])
    plt.savefig(r'.\\results\\modescaling\\LineModes\\Connections_Diag_leftdown_alt.pdf', transparent=True)
    plt.savefig(r'.\\results\\modescaling\\LineModes\\Connections_Diag_leftdown_alt.png', dpi=400, transparent=True)
    plt.show()
    plt.close()

    f, ax = plt.subplots(figsize=(4, 4))
    plt.gca().set_aspect('equal', adjustable='box')
    lattice = np.array([[1, -1], [-1, 3]])
    draw_vertex_rep(lattice, ax)
    "input args draw_triangle: (x0, y0, diaghorver, edge, ax, outin, n_arrows, zorder)"
    "edge: {1, 2, 3, 4, 5, 6, 7, 8} = {left, up, right, down, leftup, rightup, rightdown, leftdown}"

    draw_triangle(1, 1, 1, 1, ax, 1, 1, 5)
    draw_triangle(1, 1, 1, 3, ax, 1, 1, 5)
    draw_triangle(1, 1, 2, 4, ax, 1, 1, 5)
    draw_triangle(1, 1, 2, 2, ax, 0, 1, 5)
    xlims = ax.get_xlim()
    ylims = ax.get_ylim()
    plt.xlim([xlims[0] - 0.55, xlims[1] + 0.55])
    plt.ylim([ylims[0] - 0.55, ylims[1] + 0.55])
    # plt.tight_layout()
    plt.savefig(r'.\\results\\modescaling\\LineModes\\Connections_Diag_leftdown_rotsqr.pdf', transparent=True)
    plt.savefig(r'.\\results\\modescaling\\LineModes\\Connections_Diag_leftdown_rotsqr.png', dpi=400, transparent=True)
    plt.show()
    plt.close()

    f, ax = plt.subplots(figsize=(4, 4))
    plt.gca().set_aspect('equal', adjustable='box')
    lattice = np.array([[1, -1], [-1, 3]])
    draw_vertex_rep(lattice, ax)
    "input args draw_triangle: (x0, y0, diaghorver, edge, ax, outin, n_arrows, zorder)"
    "edge: {1, 2, 3, 4, 5, 6, 7, 8} = {left, up, right, down, leftup, rightup, rightdown, leftdown}"

    draw_triangle(3, 3, 1, 1, ax, 1, 1, 5)
    draw_triangle(3, 3, 1, 3, ax, 1, 1, 5)
    draw_triangle(3, 3, 2, 4, ax, 1, 1, 5)
    draw_triangle(3, 3, 2, 2, ax, 0, 1, 5)
    xlims = ax.get_xlim()
    ylims = ax.get_ylim()
    plt.xlim([xlims[0] - 0.55, xlims[1] + 0.55])
    plt.ylim([ylims[0] - 0.55, ylims[1] + 0.55])
    # plt.tight_layout()
    plt.savefig(r'.\\results\\modescaling\\LineModes\\Connections_Diag_rightup_rotsqr.pdf', transparent=True)
    plt.savefig(r'.\\results\\modescaling\\LineModes\\Connections_Diag_rightup_rotsqr.png', dpi=400, transparent=True)
    plt.show()
    plt.close()
    return 0


def hierarchyfigure():
    saveschem = 'results//modescaling//UC3_schematic'
    savepix = 'results//modescaling//UC3_pixrep'
    savevert = 'results//modescaling//UC3_vertrep'
    config = np.array([[3]])
    # config = np.tile(config, (3, 3))
    f, ax = plt.subplots(1, 1)
    draw_schematic(config, ax)
    plt.tight_layout()
    plt.savefig(saveschem + '.pdf', transparent=True, bbox_inches='tight')
    plt.savefig(saveschem + '.png', transparent=True, bbox_inches='tight')
    # plt.show()
    plt.close()
    f, ax = plt.subplots(1, 1)
    draw_pixel_rep(config, ax)
    plt.tight_layout()
    plt.savefig(savepix + '.pdf', transparent=True, bbox_inches='tight')
    plt.savefig(savepix + '.png', transparent=True, bbox_inches='tight')
    # plt.show()
    plt.close()
    f, ax = plt.subplots(1, 1)
    draw_vertex_rep(config, ax)
    plt.tight_layout()
    plt.savefig(savevert + '.pdf', transparent=True, bbox_inches='tight')
    plt.savefig(savevert + '.png', transparent=True, bbox_inches='tight')
    # plt.show()
    plt.close()


def draw_unitcell_coordinates(cell, r, ax):
    # f = plt.figure()
    # ax = plt.axes()
    # circle = plt.Circle((0, 0), radius=0.75, fc='y')
    # plt.gca().add_patch(circle)
    # c1 = (51/255, 255/255, 255/255)
    # c2 = (29/255, 143/255, 254/255)
    # c3 = (255/255, 204/255, 51/255)
    # c4 = (254/255, 127/255, 0/255)
    cp = (191 / 255, 191 / 255, 255 / 255)
    cl = 'black'
    cb = (0 / 255, 0 / 255, 139 / 255)
    cb2 = (124 / 255, 124 / 255, 255 / 255)
    c1, c2, c3, c4 = cp, cp, cp, cp
    cma = plt.cm.get_cmap('tab20c')
    cm_skip = [cma.colors[i] for i in np.arange(8, 12)]
    # cm_skip = [cma.colors[i] for i in np.arange(0, 4)]
    c1, c2, c3, c4 = cm_skip[0], cm_skip[1], cm_skip[2], cm_skip[3]
    linewidth = 1
    x0 = r[0, 0]
    y0 = r[0, 1]
    x1 = r[1, 0]
    y1 = r[1, 1]
    x2 = r[2, 0]
    y2 = r[2, 1]
    x3 = r[3, 0]
    y3 = r[3, 1]
    x4 = r[4, 0]
    y4 = r[4, 1]
    x5 = r[5, 0]
    y5 = r[5, 1]
    x6 = r[6, 0]
    y6 = r[6, 1]
    x7 = r[7, 0]
    y7 = r[7, 1]
    if cell == 0:
        tr1 = plt.Polygon([[x0, y0], [x1, y1], [x3, y3]], fc=c1, ec='black', fill=True)
        tr2 = plt.Polygon([[x2, y2], [x1, y1], [x4, y4]], fc=c1, ec='black', fill=True)
        tr3 = plt.Polygon([[x7, y7], [x4, y4], [x6, y6]], fc=c1, ec='black', fill=True)
        "lines are given with (x0, x1), (y0, y1)"
        line1 = plt.Line2D((x3, x5), (y3, y5), lw=linewidth, color=cl)
        line2 = plt.Line2D((x5, x6), (y5, y6), lw=linewidth, color=cl)
        # line1 = plt.Line2D((x0, x0), (y0 + 1, y0 + 2), lw=linewidth, color=c1)
        # line2 = plt.Line2D((x0, x0 + 1), (y0 + 2, y0 + 2), lw=linewidth, color=c1)
        ax.add_patch(tr1)
        ax.add_patch(tr2)
        ax.add_patch(tr3)
        ax.add_line(line1)
        ax.add_line(line2)
    elif cell == 1:
        tr1 = plt.Polygon([[x0, y0], [x1, y1], [x3, y3]], fc=c2, ec='black', fill=True)
        tr2 = plt.Polygon([[x2, y2], [x1, y1], [x4, y4]], fc=c2, ec='black', fill=True)
        tr3 = plt.Polygon([[x5, y5], [x3, y3], [x6, y6]], fc=c2, ec='black', fill=True)
        "lines are given with (x0, x1), (y0, y1)"
        line1 = plt.Line2D((x4, x7), (y4, y7), lw=linewidth, color=cl)
        line2 = plt.Line2D((x6, x7), (y6, y7), lw=linewidth, color=cl)
        # line1 = plt.Line2D((x0 + 1, x0 + 2), (y0 + 2, y0 + 2), lw=linewidth, color=c2)
        # line2 = plt.Line2D((x0 + 2, x0 + 2), (y0 + 1, y0 + 2), lw=linewidth, color=c2)
        ax.add_patch(tr1)
        ax.add_patch(tr2)
        ax.add_patch(tr3)
        ax.add_line(line1)
        ax.add_line(line2)
    elif cell == 2:
        tr1 = plt.Polygon([[x0, y0], [x1, y1], [x3, y3]], fc=c3, ec='black', fill=True)
        tr2 = plt.Polygon([[x7, y7], [x6, y6], [x4, y4]], fc=c3, ec='black', fill=True)
        tr3 = plt.Polygon([[x5, y5], [x3, y3], [x6, y6]], fc=c3, ec='black', fill=True)
        "lines are given with (x0, x1), (y0, y1)"
        line1 = plt.Line2D((x1, x2), (y1, y2), lw=linewidth, color=cl)
        line2 = plt.Line2D((x2, x4), (y2, y4), lw=linewidth, color=cl)
        # line1 = plt.Line2D((x0 + 1, x0 + 2), (y0, y0), lw=linewidth, color=c3)
        # line2 = plt.Line2D((x0 + 2, x0 + 2), (y0 + 1, y0), lw=linewidth, color=c3)
        ax.add_patch(tr1)
        ax.add_patch(tr2)
        ax.add_patch(tr3)
        ax.add_line(line1)
        ax.add_line(line2)
    elif cell == 3:
        tr1 = plt.Polygon([[x2, y2], [x1, y1], [x4, y4]], fc=c4, ec='black', fill=True)
        tr2 = plt.Polygon([[x7, y7], [x6, y6], [x4, y4]], fc=c4, ec='black', fill=True)
        tr3 = plt.Polygon([[x5, y5], [x3, y3], [x6, y6]], fc=c4, ec='black', fill=True)
        "lines are given with (x0, x1), (y0, y1)"
        line1 = plt.Line2D((x0, x1), (y0, y1), lw=linewidth, color=cl)
        line2 = plt.Line2D((x0, x3), (y0, y3), lw=linewidth, color=cl)
        # line1 = plt.Line2D((x0, x0 + 1), (y0, y0), lw=linewidth, color=c4)
        # line2 = plt.Line2D((x0, x0), (y0, y0 + 1), lw=linewidth, color=c4)
        ax.add_patch(tr1)
        ax.add_patch(tr2)
        ax.add_patch(tr3)
        ax.add_line(line1)
        ax.add_line(line2)
    elif cell == -1:
        tr1 = plt.Polygon([[x2, y2], [x1, y1], [x4, y4]], fc=c1, ec='black', fill=True)
        tr2 = plt.Polygon([[x7, y7], [x6, y6], [x4, y4]], fc=c1, ec='black', fill=True)
        tr3 = plt.Polygon([[x5, y5], [x3, y3], [x6, y6]], fc=c1, ec='black', fill=True)
        tr4 = plt.Polygon([[x0, y0], [x1, y1], [x3, y3]], fc=c1, ec='black', fill=True)
        ax.add_patch(tr1)
        ax.add_patch(tr2)
        ax.add_patch(tr3)
        ax.add_patch(tr4)
    else:
        print('invalid cell value given, space will be left empty')
    plt.axis('scaled')
    plt.axis('off')
    # modes = get_modes(np.array(lattice), 1, 1, draw_lattice=1)
    return


def gram_schmidt(modes):
    v1, v2 = modes[0].flatten(), modes[1].flatten()
    u1 = np.array([[0, 0], [1, 0], [0, -1], [0, 0], [0, 1], [0, 0], [-1, 0], [0, 0]]).flatten()
    u2 = v2 - np.dot(v2, u1) / (np.dot(u1, u1)) * u1
    return u1, u2


def animate_unitcell(cell, alpha, fps, time, mode):
    modes, floppy = get_modes(np.array([[cell]]), 1, 1, draw_lattice=0, draw_modes=0, pbc=0, fix_2edge=1)
    u1, u2 = gram_schmidt(floppy)
    print(u1, u2)
    MCRS = np.array([[0, 0], [0, -1], [0, 0], [1, 0], [-1, 0], [0, 0], [0, 1], [0, 0]])
    MCRS = np.array([[u1[0], u1[1]], [u1[4], u1[5]], [u1[10], u1[11]], [u1[2], u1[3]], [u1[12], u1[13]], [u1[6], u1[7]],
                     [u1[8], u1[9]], [u1[14], u1[15]]])
    MD = np.array([[u2[0], u2[1]], [u2[4], u2[5]], [u2[10], u2[11]], [u2[2], u2[3]], [u2[12], u2[13]], [u2[6], u2[7]],
                   [u2[8], u2[9]], [u2[14], u2[15]]])

    Nframes = fps * time
    framems = 1000 / fps
    r = np.array([[0, 0], [1, 0], [2, 0], [0, 1], [2, 1], [0, 2], [1, 2], [2, 2]])
    f, ax = plt.subplots()
    # cp = (191 / 255, 191 / 255, 255 / 255)
    cl = 'black'
    # cb = (0 / 255, 0 / 255, 139 / 255)
    # cb2 = (124 / 255, 124 / 255, 255 / 255)
    # c1, c2, c3, c4 = cp, cp, cp, cp
    cma = plt.cm.get_cmap('tab20c')
    cm_skip = [cma.colors[i] for i in np.arange(8, 12)]
    # cm_skip = [cma.colors[i] for i in np.arange(0, 4)]
    c1, c2, c3, c4 = cm_skip[0], cm_skip[1], cm_skip[2], cm_skip[3]
    linewidth = 1
    x0 = r[0, 0]
    y0 = r[0, 1]
    x1 = r[1, 0]
    y1 = r[1, 1]
    x2 = r[2, 0]
    y2 = r[2, 1]
    x3 = r[3, 0]
    y3 = r[3, 1]
    x4 = r[4, 0]
    y4 = r[4, 1]
    x5 = r[5, 0]
    y5 = r[5, 1]
    x6 = r[6, 0]
    y6 = r[6, 1]
    x7 = r[7, 0]
    y7 = r[7, 1]
    if cell == 0:
        tr1 = plt.Polygon([[x0, y0], [x1, y1], [x3, y3]], fc=c1, ec='black', fill=True)
        tr2 = plt.Polygon([[x2, y2], [x1, y1], [x4, y4]], fc=c1, ec='black', fill=True)
        tr3 = plt.Polygon([[x7, y7], [x4, y4], [x6, y6]], fc=c1, ec='black', fill=True)
        "lines are given with (x0, x1), (y0, y1)"
        line1 = plt.Line2D((x3, x5), (y3, y5), lw=linewidth, color=cl)
        line2 = plt.Line2D((x5, x6), (y5, y6), lw=linewidth, color=cl)
        # line1 = plt.Line2D((x0, x0), (y0 + 1, y0 + 2), lw=linewidth, color=c1)
        # line2 = plt.Line2D((x0, x0 + 1), (y0 + 2, y0 + 2), lw=linewidth, color=c1)
    elif cell == 1:
        tr1 = plt.Polygon([[x0, y0], [x1, y1], [x3, y3]], fc=c2, ec='black', fill=True)
        tr2 = plt.Polygon([[x2, y2], [x1, y1], [x4, y4]], fc=c2, ec='black', fill=True)
        tr3 = plt.Polygon([[x5, y5], [x3, y3], [x6, y6]], fc=c2, ec='black', fill=True)
        "lines are given with (x0, x1), (y0, y1)"
        line1 = plt.Line2D((x4, x7), (y4, y7), lw=linewidth, color=cl)
        line2 = plt.Line2D((x6, x7), (y6, y7), lw=linewidth, color=cl)
        # line1 = plt.Line2D((x0 + 1, x0 + 2), (y0 + 2, y0 + 2), lw=linewidth, color=c2)
        # line2 = plt.Line2D((x0 + 2, x0 + 2), (y0 + 1, y0 + 2), lw=linewidth, color=c2)
    elif cell == 2:
        tr1 = plt.Polygon([[x0, y0], [x1, y1], [x3, y3]], fc=c3, ec='black', fill=True)
        tr2 = plt.Polygon([[x7, y7], [x6, y6], [x4, y4]], fc=c3, ec='black', fill=True)
        tr3 = plt.Polygon([[x5, y5], [x3, y3], [x6, y6]], fc=c3, ec='black', fill=True)
        "lines are given with (x0, x1), (y0, y1)"
        line1 = plt.Line2D((x1, x2), (y1, y2), lw=linewidth, color=cl)
        line2 = plt.Line2D((x2, x4), (y2, y4), lw=linewidth, color=cl)
        # line1 = plt.Line2D((x0 + 1, x0 + 2), (y0, y0), lw=linewidth, color=c3)
        # line2 = plt.Line2D((x0 + 2, x0 + 2), (y0 + 1, y0), lw=linewidth, color=c3)
    elif cell == 3:
        tr1 = plt.Polygon([[x2, y2], [x1, y1], [x4, y4]], fc=c4, ec='black', fill=True)
        tr2 = plt.Polygon([[x7, y7], [x6, y6], [x4, y4]], fc=c4, ec='black', fill=True)
        tr3 = plt.Polygon([[x5, y5], [x3, y3], [x6, y6]], fc=c4, ec='black', fill=True)
        "lines are given with (x0, x1), (y0, y1)"
        line1 = plt.Line2D((x0, x1), (y0, y1), lw=linewidth, color=cl)
        line2 = plt.Line2D((x0, x3), (y0, y3), lw=linewidth, color=cl)
        # line1 = plt.Line2D((x0, x0 + 1), (y0, y0), lw=linewidth, color=c4)
        # line2 = plt.Line2D((x0, x0), (y0, y0 + 1), lw=linewidth, color=c4)
    else:
        tr1 = plt.Polygon([[x2, y2], [x1, y1], [x4, y4]], fc=c1, ec='black', fill=True)
        tr2 = plt.Polygon([[x7, y7], [x6, y6], [x4, y4]], fc=c1, ec='black', fill=True)
        tr3 = plt.Polygon([[x5, y5], [x3, y3], [x6, y6]], fc=c1, ec='black', fill=True)
        tr4 = plt.Polygon([[x0, y0], [x1, y1], [x3, y3]], fc=c1, ec='black', fill=True)
    plt.axis('scaled')
    plt.axis('off')
    ax.set_xlim([-1, 3])
    ax.set_ylim([-1, 3])

    def init():
        if cell != -1:
            ax.add_patch(tr1)
            ax.add_patch(tr2)
            ax.add_patch(tr3)
            ax.add_line(line1)
            ax.add_line(line2)
            return tr1, tr2, tr3, line1, line2
        else:
            ax.add_patch(tr1)
            ax.add_patch(tr2)
            ax.add_patch(tr3)
            ax.add_patch(tr4)
            return tr1, tr2, tr3, tr4

    def animate(i):
        if mode:
            if int(i / (0.5 * Nframes)) % 2:
                dr = r + 2 * alpha * MD - (i / (0.5 * Nframes)) * alpha * MD
            else:
                dr = r + i / (0.5 * Nframes) * alpha * MD
        else:
            if int(i / (0.5 * Nframes)) % 2:
                dr = r - 2 * alpha * MCRS + (i / (0.5 * Nframes)) * alpha * MCRS
            else:
                dr = r - i / (0.5 * Nframes) * alpha * MCRS
        x0 = dr[0, 0]
        y0 = dr[0, 1]
        x1 = dr[1, 0]
        y1 = dr[1, 1]
        x2 = dr[2, 0]
        y2 = dr[2, 1]
        x3 = dr[3, 0]
        y3 = dr[3, 1]
        x4 = dr[4, 0]
        y4 = dr[4, 1]
        x5 = dr[5, 0]
        y5 = dr[5, 1]
        x6 = dr[6, 0]
        y6 = dr[6, 1]
        x7 = dr[7, 0]
        y7 = dr[7, 1]
        if cell == 0:
            tr1.set_xy([[x0, y0], [x1, y1], [x3, y3]])
            tr2.set_xy([[x2, y2], [x1, y1], [x4, y4]])
            tr3.set_xy([[x7, y7], [x4, y4], [x6, y6]])
            "lines are given with (x0, x1), (y0, y1)"
            line1.set_xdata(np.array([x3, x5]))
            line1.set_ydata(np.array([y3, y5]))
            line2.set_xdata(np.array([x5, x6]))
            line2.set_ydata(np.array([y5, y6]))
            # line1 = plt.Line2D((x0, x0), (y0 + 1, y0 + 2), lw=linewidth, color=c1)
            # line2 = plt.Line2D((x0, x0 + 1), (y0 + 2, y0 + 2), lw=linewidth, color=c1)
            return tr1, tr2, tr3, line1, line2
        elif cell == 1:
            tr1.set_xy([[x0, y0], [x1, y1], [x3, y3]])
            tr2.set_xy([[x2, y2], [x1, y1], [x4, y4]])
            tr3.set_xy([[x5, y5], [x3, y3], [x6, y6]])
            "lines are given with (x0, x1), (y0, y1)"
            line1.set_xdata(np.array([x4, x7]))
            line1.set_ydata(np.array([y4, y7]))
            line2.set_xdata([x6, x7])
            line2.set_ydata([y6, y7])
            # line1 = plt.Line2D((x0 + 1, x0 + 2), (y0 + 2, y0 + 2), lw=linewidth, color=c2)
            # line2 = plt.Line2D((x0 + 2, x0 + 2), (y0 + 1, y0 + 2), lw=linewidth, color=c2)
            return tr1, tr2, tr3, line1, line2
        elif cell == 2:
            tr1.set_xy([[x0, y0], [x1, y1], [x3, y3]])
            tr2.set_xy([[x7, y7], [x6, y6], [x4, y4]])
            tr3.set_xy([[x5, y5], [x3, y3], [x6, y6]])
            "lines are given with (x0, x1), (y0, y1)"
            line1.set_xdata([x1, x2])
            line1.set_ydata([y1, y2])
            line2.set_xdata([x2, x4])
            line2.set_ydata([y2, y4])
            # line1 = plt.Line2D((x0 + 1, x0 + 2), (y0, y0), lw=linewidth, color=c3)
            # line2 = plt.Line2D((x0 + 2, x0 + 2), (y0 + 1, y0), lw=linewidth, color=c3)
            return tr1, tr2, tr3, line1, line2
        elif cell == 3:
            tr1.set_xy([[x2, y2], [x1, y1], [x4, y4]])
            tr2.set_xy([[x7, y7], [x6, y6], [x4, y4]])
            tr3.set_xy([[x5, y5], [x3, y3], [x6, y6]])
            "lines are given with (x0, x1), (y0, y1)"
            line1.set_xdata([x0, x1])
            line1.set_ydata([y0, y1])
            line2.set_xdata([x0, x3])
            line2.set_ydata([y0, y3])
            # line1 = plt.Line2D((x0, x0 + 1), (y0, y0), lw=linewidth, color=c4)
            # line2 = plt.Line2D((x0, x0), (y0, y0 + 1), lw=linewidth, color=c4)
            return tr1, tr2, tr3, line1, line2
        elif cell == -1:
            tr1.set_xy([[x2, y2], [x1, y1], [x4, y4]])
            tr2.set_xy([[x7, y7], [x6, y6], [x4, y4]])
            tr3.set_xy([[x5, y5], [x3, y3], [x6, y6]])
            tr4.set_xy([[x0, y0], [x1, y1], [x3, y3]])
            return tr1, tr2, tr3, tr4
        return -1

    # animate(0)
    # animate(10)
    anim = animation.FuncAnimation(f, animate,
                                   init_func=init,
                                   frames=Nframes,
                                   interval=framems,
                                   blit=True)
    plt.show()
    if mode:
        anim.save(r'.\\results\\modescaling\\uc{:d}_D+.mp4'.format(cell), fps=fps)
    else:
        anim.save(r'.\\results\\modescaling\\uc{:d}_CRS+.mp4'.format(cell), fps=fps)
    return 0

    # modes = get_modes(np.array(lattice), 1, 1, draw_lattice=1)

def rotation_mode(r):
    "input: r[Nvertices, (0:x, 1:y)]"
    "determine centre of mass location"
    R = np.sum(r, axis=0) / np.shape(r)[0]
    u = np.zeros((np.shape(r)[0], 2))
    dr = r - R
    u[:, 0] = -dr[:, 1]
    u[:, 1] = dr[:, 0]
    # for i in range(np.shape(r)[0]):
    #     u[i, 0] = -dr[i, 1]
    #     u[i, 1] = dr[i, 0]
    return u

def gram_schmidt_mode(modes, Ncells_x, Ncells_y, trivialmodes = False, r=[]):
    # v = np.reshape(modes, (np.shape(modes)[0], -1))
    if trivialmodes:
        u = np.zeros((4, np.shape(modes)[1], np.shape(modes)[2]))
    else:
        u = np.zeros((1, np.shape(modes)[1], np.shape(modes)[2]))
    Ncells = Ncells_x * Ncells_y
    for i in range(Ncells):
        u[0, 3 * i:3 * (i + 1)] = ((-1) ** (i % Ncells_y + int(i / Ncells_y))) * np.array([[0, 0], [1, 0], [0, -1]])
    for i in range(3 * Ncells, 3 * Ncells + Ncells_x * 2, 2):
        u[0, i: i + 2] = ((-1) ** (Ncells_y - 1 + (i - 3 * Ncells) / 2)) * np.array([[0, 0], [0, 1]])
    for i in range(3 * Ncells + 2 * Ncells_x, 3 * Ncells + 2 * Ncells_x + 2 * Ncells_y, 2):
        u[0, i: i + 2] = ((-1) ** (Ncells_x - 1 + (i - (3 * Ncells + 2 * Ncells_x)) / 2)) * np.array([[0, 0], [-1, 0]])
    u[0, -1] = np.array([0, 0])
    if trivialmodes:
        u[1] = np.zeros_like(modes[0])
        u[2] = np.zeros_like(modes[0])
        u[3] = np.zeros_like(modes[0])
        u[1, :, 0] = np.full_like(u[1, :, 0], 1)
        u[2, :, 1] = np.full_like(u[2, :, 1], 1)
        u[3] = rotation_mode(r)
    u = np.reshape(u, (np.shape(u)[0], -1))
    v = np.reshape(modes, (np.shape(modes)[0], -1))
    if trivialmodes:
        u[0] /= np.sqrt(np.dot(u[0], u[0]))
        u[1] /= np.sqrt(np.dot(u[1], u[1]))
        u[2] /= np.sqrt(np.dot(u[2], u[2]))
        u[3] /= np.sqrt(np.dot(u[3], u[3]))
        startindex = 4
    else:
        u[0] /= np.sqrt(np.dot(u[0], u[0]))
        startindex = 1
    v = np.append(u, v, axis=0)
    g = np.zeros_like(v)
    for i in range(0, np.shape(v)[0]):
        g[i] = v[i]
        for j in range(0, i):
            g[i] -= np.dot(g[i], g[j]) / (np.dot(g[j], g[j])) * g[j]
    inddel = np.argwhere(np.all(np.abs(g) < 1e-12, axis=1))
    g = np.delete(g, inddel[:, 0], axis=0)
    g = np.reshape(g, (np.shape(g)[0], np.shape(modes)[1], np.shape(modes)[2]))
    # print(g)
    return g


def draw_modes(config, alpha, trivialmodes=False, draw_config=[]):
    modes, floppy, positions = get_modes(config, 1, 1, draw_lattice=0, draw_modes=0, pbc=0, use_qr=0, nofix=int(trivialmodes))
    # print(positions)
    # urot = rotation_mode(positions)
    # u = np.zeros((1, np.shape(positions)[0], 2))
    # u[0] = rotation_mode(positions)
    Ncells_x = np.shape(config)[0]
    Ncells_y = np.shape(config)[1]
    Ncells = Ncells_x * Ncells_y
    u = gram_schmidt_mode(floppy, np.shape(config)[0], np.shape(config)[1], trivialmodes=trivialmodes, r=positions)
    # print(u)
    # alpha = 0.05
    # Nframes = fps*time
    # framems = 1000/fps
    print('Number of modes: {:d}'.format(modes))
    if len(draw_config) > 0:
        config = draw_config

    for m in range(np.shape(u)[0]):
        r = positions + alpha * u[m]
        f = plt.figure(1, figsize=(1.69291, 1.69291), frameon=False)
        ax = f.add_axes([0.1, 0.1, 0.8, 0.8])
        # f, ax = plt.subplots(figsize=(1.69291, 1.69291), frameon=False)
        # cp = (191 / 255, 191 / 255, 255 / 255)
        cl = 'black'
        # cb = (0 / 255, 0 / 255, 139 / 255)
        # cb2 = (124 / 255, 124 / 255, 255 / 255)
        # c1, c2, c3, c4 = cp, cp, cp, cp
        cma = plt.cm.get_cmap('tab20c')
        cm_skip = [cma.colors[i] for i in np.arange(0, 4)]
        # cm_skip = [cma.colors[i] for i in np.arange(0, 4)]
        c1, c2, c3, c4 = cm_skip[0], cm_skip[1], cm_skip[2], cm_skip[3]
        "Pink"
        cmp = categorical_cmap(1, 4, cmap='tab20', offset=12)
        c1, c2, c3, c4 = cmp(0), cmp(1), cmp(2), cmp(3)
        linewidth = 1
        for x in range(np.shape(config)[0]):
            for y in range(np.shape(config)[1]):
                cell = config[x, y]
                index_bottomleft = x * Ncells_y * 3 + y * 3
                index_middleleft = index_bottomleft + 1
                index_bottommiddle = index_bottomleft + 2
                # r_bottomleft = r[index_bottomleft: index_bottomleft + 3]
                if y == Ncells_y - 1:
                    index_topleft = 3 * Ncells + 2 * x
                    index_topmiddle = index_topleft + 1
                else:
                    index_topleft = x * Ncells_y * 3 + 3 * (y+1)
                    index_topmiddle = index_topleft + 2
                if x == Ncells_x - 1:
                    index_bottomright = 3 * Ncells + 2 * Ncells_x + 2 * y
                    index_middleright = index_bottomright + 1
                else:
                    index_bottomright = (x + 1) * Ncells_y * 3 + 3 * y
                    index_middleright = index_bottomright + 1
                if x <= Ncells_x - 2 and y == Ncells_y - 1:
                    index_topright = 3 * Ncells + 2 * (x + 1)
                elif x <= Ncells_x - 2 and y <= Ncells_y - 2:
                    index_topright = (x + 1) * Ncells_y * 3 + 3 * (y + 1)
                else:
                    index_topright = 3 * Ncells + 2 * Ncells_y + 2 * (y + 1)
                if cell == 0:
                    tr1 = plt.Polygon([r[index_bottomleft], r[index_middleleft], r[index_bottommiddle]], fc=c1,
                                      ec='black', fill=True)
                    tr2 = plt.Polygon([r[index_bottommiddle], r[index_bottomright], r[index_middleright]], fc=c1,
                                      ec='black', fill=True)
                    tr3 = plt.Polygon([r[index_topmiddle], r[index_topright], r[index_middleright]], fc=c1,
                                      ec='black', fill=True)
                    "lines are given with (x0, x1), (y0, y1)"
                    line1 = plt.Line2D((r[index_middleleft, 0], r[index_topleft, 0]), (r[index_middleleft, 1],
                                                                                       r[index_topleft, 1]),
                                       lw=linewidth, color=cl)
                    line2 = plt.Line2D((r[index_topright, 0], r[index_topmiddle, 0]), (r[index_topright, 1],
                                                                                       r[index_topmiddle, 1]),
                                       lw=linewidth, color=cl)
                elif cell == 1:
                    tr1 = plt.Polygon([r[index_bottomleft], r[index_middleleft], r[index_bottommiddle]], fc=c2,
                                      ec='black', fill=True)
                    tr2 = plt.Polygon([r[index_bottommiddle], r[index_bottomright], r[index_middleright]], fc=c2,
                                      ec='black', fill=True)
                    tr3 = plt.Polygon([r[index_middleleft], r[index_topleft], r[index_topmiddle]], fc=c2,
                                      ec='black', fill=True)
                    "lines are given with (x0, x1), (y0, y1)"
                    line1 = plt.Line2D((r[index_topmiddle, 0], r[index_topright, 0]), (r[index_topmiddle, 1],
                                                                                       r[index_topright, 1]),
                                       lw=linewidth, color=cl)
                    line2 = plt.Line2D((r[index_topright, 0], r[index_middleright, 0]), (r[index_topright, 1],
                                                                                         r[index_middleright, 1]),
                                       lw=linewidth, color=cl)
                elif cell == 2:
                    tr1 = plt.Polygon([r[index_bottomleft], r[index_middleleft], r[index_bottommiddle]], fc=c3,
                                      ec='black', fill=True)
                    tr2 = plt.Polygon([r[index_middleleft], r[index_topleft], r[index_topmiddle]], fc=c3,
                                      ec='black', fill=True)
                    tr3 = plt.Polygon([r[index_topmiddle], r[index_topright], r[index_middleright]], fc=c3,
                                      ec='black', fill=True)
                    "lines are given with (x0, x1), (y0, y1)"
                    line1 = plt.Line2D((r[index_middleright, 0], r[index_bottomright, 0]), (r[index_middleright, 1],
                                                                                            r[index_bottomright, 1]),
                                       lw=linewidth, color=cl)
                    line2 = plt.Line2D((r[index_bottomright, 0], r[index_bottommiddle, 0]), (r[index_bottomright, 1],
                                                                                             r[index_bottommiddle, 1]),
                                       lw=linewidth, color=cl)
                elif cell == 3:
                    tr1 = plt.Polygon([r[index_middleleft], r[index_topleft], r[index_topmiddle]], fc=c4,
                                      ec='black', fill=True)
                    tr2 = plt.Polygon([r[index_bottommiddle], r[index_bottomright], r[index_middleright]], fc=c4,
                                      ec='black', fill=True)
                    tr3 = plt.Polygon([r[index_topmiddle], r[index_topright], r[index_middleright]], fc=c4,
                                      ec='black', fill=True)
                    "lines are given with (x0, x1), (y0, y1)"
                    line1 = plt.Line2D((r[index_middleleft, 0], r[index_bottomleft, 0]), (r[index_middleleft, 1],
                                                                                          r[index_bottomleft, 1]),
                                       lw=linewidth, color=cl)
                    line2 = plt.Line2D((r[index_bottomleft, 0], r[index_bottommiddle, 0]), (r[index_bottomleft, 1],
                                                                                       r[index_bottommiddle, 1]),
                                       lw=linewidth, color=cl)
                else:
                    tr1 = plt.Polygon([r[index_middleleft], r[index_topleft], r[index_topmiddle]], fc=c1,
                                      ec='black', fill=True)
                    tr2 = plt.Polygon([r[index_bottommiddle], r[index_bottomright], r[index_middleright]], fc=c1,
                                      ec='black', fill=True)
                    tr3 = plt.Polygon([r[index_topmiddle], r[index_topright], r[index_middleright]], fc=c1,
                                      ec='black', fill=True)
                    tr4 = plt.Polygon([r[index_bottomleft], r[index_middleleft], r[index_bottommiddle]], fc=c1,
                                      ec='black', fill=True)
                if cell not in [0, 1, 2, 3]:
                    ax.add_patch(tr1)
                    ax.add_patch(tr2)
                    ax.add_patch(tr3)
                    ax.add_patch(tr4)
                else:
                    ax.add_patch(tr1)
                    ax.add_patch(tr2)
                    ax.add_patch(tr3)
                    ax.add_line(line1)
                    ax.add_line(line2)
                # plt.show()
        plt.axis('scaled')
        plt.axis('off')
        # ax.set_xlim([-1, Ncells_x*2 + 1])
        # ax.set_ylim([-1, Ncells_y*2 + 1])
        ax.set_xlim([-(1./6.)*(Ncells_x * 2), (1+1./6.)*(Ncells_x * 2)])
        ax.set_ylim([-(1./6.)*(Ncells_y * 2), (1+1./6.)*Ncells_y * 2])
        print('mode {:d}'.format(m))
        print(u[m])
        plt.savefig(u'.\\results\\modescaling\\LineModes\\Pink_4x4_mode{:d}_Schematic.pdf'.format(m), transparent=True)
        print(np.dot(u[m].flatten(), np.reshape(u, (np.shape(u)[0], -1)).T))
        plt.show()
        plt.close()
    return 0

    # modes = get_modes(np.array(lattice), 1, 1, draw_lattice=1)


def main():
    # animate_unitcell(0, 1., 23, 2, 1)
    # draw_horizontal_connections()
    # draw_diagonal_connections()
    # draw_vertical_connections()
    # lattice = np.array([[3, 1, 3, 1], [3, 1, 3, 1], [3, 1, 3, 1], [3, 1, 3, 1]])
    # lattice = np.array([[0, 1, 2, 1, 0, 3, 0, 0, 3]])
    # lattice = np.reshape(lattice, (3, 3))
    # f, ax = plt.subplots()
    # r = np.array([[0, 0], [1, 0], [2, 0], [0, 1], [2, 1], [0, 2], [1, 2], [2, 2]])
    # draw_unitcell_coordinates(3, r, ax)
    # plt.show()
    # plt.close()
    # for k in range(2, 9):
    #     lattice = np.random.randint(0, 4, (k, k))
    #     f, ax = plt.subplots(figsize=(4, 2))
    #     plt.gca().set_aspect('equal', adjustable='box')
    #     draw_schematic(lattice, ax)
    #
    #     plt.savefig(r"results//modescaling//rand{:d}x{:d}_green".format(k, k)+'.png', transparent=True, bbox_inches='tight')
    #     plt.savefig(r"results//modescaling//rand{:d}x{:d}_green".format(k, k) + '.pdf', transparent=True, bbox_inches='tight')
    #     plt.savefig(r"results//modescaling//rand{:d}x{:d}_green".format(k, k) + '.svg', transparent=True, bbox_inches='tight')
    #     plt.show()
    #     plt.close()

    # sub_mat = np.array([0, 1, 2, 1, 1, 3, 0, 0, 3])
    # sub_mat = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2, 1, 1, 3, 1, 1, 3, 1, 1, 3, 0, 0, 3, 0, 0, 3, 0, 0, 3, 0, 1, 2, 0, 1, 2, 0, 1, 2, 1, 1, 3, 1, 1, 3, 1, 1, 3, 0, 0, 3, 0, 0, 3, 0, 0, 3, 0, 1, 2, 0, 1, 2, 0, 1, 2, 1, 1, 3, 1, 1, 3, 1, 1, 3, 0, 0, 3, 0, 0, 3, 0, 0, 3])
    # sub_mat = np.array([[0]])
    k = 4
    # data = np.loadtxt("results\modescaling\data_new_rrQR_i_n_M_{}x{}.txt".format(k, k), delimiter=',')
    # sub_mat = np.array([0, 1, 2, 1, 0, 3, 0, 0, 3])
    # sub_mat = np.array([0])
    # sub_mat = data[8322, 1:k*k+1]
    # sub_mat = np.array([4, 1, 2, 4, 0, 3, 4, 0, 3])
    # sub_mat = np.array([[0, 1, 2, 0, 0, 3, 0, 0, 3]])
    # lattice = np.reshape(sub_mat, (k, k))
    # sub_mat2 = np.array([[0, 1, 2, 0, 0, 3, 0, 0, 3]])
    # lat2 = np.reshape(sub_mat2, (k, k))
    sub_mat = np.array([[0, 0, 2, 3, 0, 0, 2, 3, 0, 0, 2, 3, 0, 0, 2, 3]])
    lattice = np.reshape(sub_mat, (k, k))
    sub_mat2 = np.array([[0, 0, 2, 3, 0, 0, 2, 3, 0, 0, 2, 3, 0, 0, 2, 3]])
    lat2 = np.reshape(sub_mat2, (k, k))
    draw_modes(lattice, 1.5, trivialmodes=True, draw_config=lat2)
    lattice = np.tile(lattice, (3, 3))
    f, ax = plt.subplots(figsize=(1.69291, 1.69291))
    draw_schematic(lattice, ax)
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    # plt.tight_layout()
    # plt.savefig('.\\results\\modescaling\\012003003_pink_tile3x3' + '.pdf', transparent=True, bbox_inches='tight')
    # plt.savefig('.\\results\\modescaling\\012003003_pink_tile3x3' + '.svg', transparent=True, bbox_inches='tight')
    # plt.savefig('.\\results\\modescaling\\012003003_pink_tile3x3' + '.png', transparent=True, bbox_inches='tight', dpi=400)
    plt.show()
    plt.close()
    f = plt.figure(1, figsize=(1.69291, 1.69291), frameon=False)
    ax = f.add_axes([0.2, 0.2, 0.6, 0.6])
    lattice = Pixel_rep(sub_mat, k, k)
    ax.imshow(lattice[0].T, origin='lower', cmap='Greys', vmin=0, vmax=1)
    for x in range(k):
        ax.axhline(2 * x + .5, lw=1, color='0.5')
    for y in range(k):
        ax.axvline(2 * y + .5, lw=1, color='0.5')
    ax.axis('off')
    # plt.tight_layout()
    plt.savefig('.\\results\\modescaling\\0023002300230023_pixrep' + '.pdf', transparent=True)
    plt.savefig('.\\results\\modescaling\\0023002300230023_pixrep' + '.svg', transparent=True)
    plt.savefig('.\\results\\modescaling\\0023002300230023_pixrep' + '.png', transparent=True,
                dpi=400)
    plt.show()
    plt.close()

    # plt.savefig(r"results//modescaling//unitcell_0_green.png", transparent=True, bbox_inches='tight', dpi=400)
    # plt.savefig(r"results//modescaling//unitcell_0_green.pdf", transparent=True, bbox_inches='tight')
    # plt.savefig(r"results//modescaling//unitcell_0_green.svg", transparent=True, bbox_inches='tight')
    # draw_pixel_rep(lattice, ax)
    # lattice = Martin_rep(sub_mat, k, k)

    # f, ax = plt.subplots(figsize=(4.7, 4.7), frameon=False)
    # ax.imshow(lattice[0].T, origin='lower', cmap='Greys', vmin=0, vmax=1)
    # for x in range(k):
    #     ax.axhline(2 * x + .5, lw=1, color='0.5')
    # for y in range(k):
    #     ax.axvline(2 * y + .5, lw=1, color='0.5')
    # # draw_pixel_rep(lattice, ax)
    # # # ax.set_xlim([5, 13])
    # # # ax.set_ylim([5, 13])
    # # ax.xaxis.set_visible(False)
    # # ax.yaxis.set_visible(False)
    # ax.axis('off')
    # plt.tight_layout()
    # plt.savefig('.\\results\\modescaling\\012013003_pixrep' + '.pdf', transparent=True, bbox_inches='tight')
    # plt.savefig('.\\results\\modescaling\\012013003_pixrep' + '.svg', transparent=True, bbox_inches='tight')
    # plt.savefig('.\\results\\modescaling\\012013003_pixrep' + '.png', transparent=True, bbox_inches='tight',
    #             dpi=400)
    # plt.show()
    # plt.close()
    #
    #
    # lattice = np.tile(lattice[0], (3, 3))
    # lattice = lattice[2*k-1: 4*k+1, 2*k-1: 4*k+1]
    # f, ax = plt.subplots(figsize=(4.7, 4.7), frameon=False)
    # ax.imshow(lattice.T, origin='lower', cmap='Greys', vmin=0, vmax=1)
    # for x in range(k+2):
    #     ax.axhline(2*x-.5, lw=1, color='0.5')
    # for y in range(k+2):
    #     ax.axvline(2*y-.5, lw=1, color='0.5')
    # # draw_pixel_rep(lattice, ax)
    # # # ax.set_xlim([5, 13])
    # # # ax.set_ylim([5, 13])
    # ax.xaxis.set_visible(False)
    # ax.yaxis.set_visible(False)
    # ax.axis('off')
    # plt.tight_layout()
    # plt.savefig('.\\results\\modescaling\\012013003_pixrep_padded' + '.pdf', transparent=True, bbox_inches='tight')
    # plt.savefig('.\\results\\modescaling\\012013003_pixrep_padded' + '.svg', transparent=True, bbox_inches='tight')
    # plt.savefig('.\\results\\modescaling\\012013003_pixrep_padded' + '.png', transparent=True, bbox_inches='tight',
    #             dpi=400)
    # plt.show()
    # plt.close()
    # plt.show()
    # plt.close()
    # fnamebasis = r"results//modescaling//LineModes//LineMode_outedge_outedge_diag"
    # modes = ['inside', 'inside', 'inside', 'inside']
    # tiles = (2, 2)
    # saveschem = 'results//modescaling//UC3_schematic_green'
    # savepix = 'results//modescaling//UC3_pixrep'
    # savevert = 'results//modescaling//UC3_vertrep'
    # config = lattice
    # # # config = np.array([[1, 2, 2], [0, 0, 0], [0, 0, 2]])
    # # # config = np.tile(config, (2, 2))
    # # # config = config[1:6, 1:6]
    # f, ax = plt.subplots(1, 1)
    # draw_schematic(config, ax)
    # plt.tight_layout()
    # plt.savefig(saveschem+'.pdf', transparent=True, bbox_inches='tight')
    # plt.savefig(saveschem+'.png', transparent=True, bbox_inches='tight')
    # plt.show()
    # plt.close()
    # f, ax = plt.subplots(1, 1)
    # draw_pixel_rep(config, ax)
    # # ax.set_xlim([5, 13])
    # # ax.set_ylim([5, 13])
    # plt.tight_layout()
    # plt.savefig(savepix + '.pdf', transparent=True, bbox_inches='tight')
    # plt.savefig(savepix + '.png', transparent=True, bbox_inches='tight')
    # plt.show()
    # plt.close()
    # f, ax = plt.subplots(1, 1)
    # draw_pixel_rep(config, ax, plaquette=False)
    # # ax.set_xlim([5, 13])
    # # ax.set_ylim([5, 13])
    # plt.tight_layout()
    # plt.savefig(savepix + '_NoPlaq.pdf', transparent=True, bbox_inches='tight')
    # plt.savefig(savepix + '_NoPlaq.png', transparent=True, bbox_inches='tight')
    # plt.show()
    # plt.close()


if __name__ == '__main__':
    main()
