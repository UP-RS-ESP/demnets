from numpy.random import uniform, seed
from matplotlib.mlab import griddata
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import Delaunay
from DemNets import Slopes

# point cloud data
seed(0)
npts = 1000
x = uniform(-2, 2, npts)
y = uniform(-2, 2, npts)
z = x*np.exp(-x**2 - y**2)

# grid data for visualisation
xi = np.linspace(-2.1, 2.1, 100)
yi = np.linspace(-2.1, 2.1, 200)
zi = griddata(x, y, z, xi, yi, interp = 'linear')
zimax = abs(zi).max()
CS = plt.contour(xi, yi, zi, 20, linewidths = 0.5, colors = 'k')
CS = plt.contourf(xi, yi, zi, 10, cmap = plt.cm.rainbow,
                  vmax=zimax, vmin=-zimax)
plt.colorbar()
#plt.scatter(x, y, marker = 'o', c = 'b', s = 5)

tri = Delaunay(np.transpose((x, y)))
net, slp = Slopes(x, y, z, tri.simplices.astype('uint32'))
for i in xrange(npts):
    for k in xrange(net[i], net[i+1]):
        dx = 0.9 * (x[net[k]] - x[i])
        dy = 0.9 * (y[net[k]] - y[i])
        plt.arrow(x[i], y[i], dx, dy, length_includes_head = True, shape = 'left')

plt.xlim(-2, 2)
plt.ylim(-2, 2)
plt.title('griddata test (%d points)' % npts)
plt.show()
