import sys
from numpy.random import uniform, seed
from matplotlib.mlab import griddata
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cmx
import numpy as np
from scipy.spatial import Delaunay
from DemNets import *

norm  = colors.Normalize(vmin = 0, vmax = 1)
sccm = cmx.ScalarMappable(norm = norm, cmap = plt.cm.hot_r)

# point cloud data
#seed(0)
npts = 100
x = uniform(-2, 2, npts)
y = uniform(-2, 2, npts)
z = x*np.exp(-x**2 - y**2)
tri = Delaunay(np.transpose((x, y)))
net = FlowNetwork(z, tri.simplices.astype('uint32'))
slp = Slopes(x, y, z, net)
slp = slp.astype('float') / slp.max()

# grid data for visualisation
xi = np.linspace(-2.1, 2.1, 100)
yi = np.linspace(-2.1, 2.1, 200)
zi = griddata(x, y, z, xi, yi, interp = 'linear')
zimax = abs(zi).max()

# figure
plt.figure(1, (19.2, 12))
plt.contour(xi, yi, zi, 20, linewidths = 0.5, colors = 'k')
plt.contourf(xi, yi, zi, 10, cmap = plt.cm.rainbow,
             vmax=zimax, vmin=-zimax)
plt.colorbar()
#plt.scatter(x, y, marker = 'o', c = 'b', s = 5)

for i in xrange(npts):
    for k in xrange(net[i], net[i+1]):
        j = net[k]
        dx = (x[j] - x[i]) * 0.9
        dy = (y[j] - y[i]) * 0.9
        plt.arrow(x[i], y[i], dx, dy,
                  length_includes_head = True, shape = 'left',
                  color = sccm.to_rgba(slp[k]))

plt.xlim(-2, 2)
plt.ylim(-2, 2)
plt.title('Visuals (%d points)' % npts)
plt.tight_layout()
plt.savefig('%s.png' % sys.argv[0][:-3])
