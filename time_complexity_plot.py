import numpy as np
import matplotlib.pyplot as plt

def cm_to_inch(x):
    return x / 2.54

plt.style.use(r'C:\\Users\\ryanv\\PycharmProjects\\Matplotlib styles\\paper-onehalf.mplstyle')

complexity_matrix = [0.22739195823669434, 0.9893538951873779, 4.204845428466797, 14.523378849029541, 34.52356791496277, 75.95084404945374]

complexity_cnn = [1.4351816177368164, 0.08776521682739258, 0.08976030349731445, 0.08975958824157715,
                  0.08776521682739258, 0.09075736999511719]

f = plt.figure(1, figsize=(cm_to_inch(8.6), cm_to_inch(8.6)))
xlength = 3.32
ylength = 0.3
xoffset = 0.25 * (xlength / 8.6)
yoffset = 0.2 * (xlength / 8.6)
figfracx = (8.6 - 1.*xoffset * 8.6 - 0.3) / 8.6
figfracy = figfracx
    # figfracy = 0.7
ax = f.add_axes([xoffset, yoffset, figfracx, figfracy])

ax.plot(np.arange(3, 9), complexity_matrix, '.-', c='tab:red')
ax.plot(np.arange(3, 9), complexity_cnn, '.-', c='tab:blue')

ax.set_xlabel('$k$')
ax.set_ylabel('$t(s)$')

f.savefig('.\\results\\modescaling\\figures\\time_complexity_modescaling_vs_CNN_nf20_nh100.pdf', facecolor=f.get_facecolor())
f.savefig('.\\results\\modescaling\\figures\\time_complexity_modescaling_vs_CNN_nf20_nh100.png', facecolor=f.get_facecolor(), dpi=400)
f.savefig('.\\results\\modescaling\\figures\\time_complexity_modescaling_vs_CNN_nf20_nh100.svg', facecolor=f.get_facecolor())
plt.show()
plt.close()