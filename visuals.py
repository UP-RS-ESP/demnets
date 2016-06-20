import sys
from numpy.random import uniform, seed
#from matplotlib.mlab import griddata
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cmx
import numpy as np
from scipy.spatial import Delaunay, Voronoi, voronoi_plot_2d
from DemNets import *

norm  = colors.Normalize(vmin = 0, vmax = 1)
sccm = cmx.ScalarMappable(norm = norm, cmap = plt.cm.hot_r)
cmap = plt.cm.rainbow

# point cloud data
#seed(0)
npts = 100
pts = uniform(-2, 2, (npts, 2))
x, y = pts[:,0], pts[:,1]
z = x*np.exp(-x**2 - y**2)
tri = Delaunay(pts)
net = FlowNetwork(z, tri.simplices.astype('uint32'))
slp = Slopes(x, y, z, net)
slp = slp.astype('float') / slp.max()

# grid data for visualisation
xi, yi = np.mgrid[-2:2:800j, -2:2:800j]
zi = griddata(pts, z, (xi, yi), method = 'nearest')
zimax = abs(zi).max()

# figure
plt.figure(1, (19.2, 12))
plt.imshow(zi.T, interpolation = 'nearest',
           cmap = cmap, origin = 'lower',
           aspect = 'auto', extent = [-2,2,-2,2])
cb = plt.colorbar()
cb.set_label('Elevation')

for i in xrange(npts):
    for k in xrange(net[i], net[i+1]):
        j = net[k]
        dx = (x[j] - x[i]) * 0.9
        dy = (y[j] - y[i]) * 0.9
        plt.arrow(x[i], y[i], dx, dy,
                  length_includes_head = True, shape = 'left',
                  color = sccm.to_rgba(slp[k]))

plt.title('DEM point cloud of %i points\n(arrows are colored according to steepness)' % npts)
plt.tight_layout()
plt.savefig('%s.png' % sys.argv[0][:-3])
