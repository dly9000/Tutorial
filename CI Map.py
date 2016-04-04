import numpy as np
import matplotlib.tri as tri
import scipy as sp
import matplotlib as mpl
import matplotlib.pyplot as plot
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import time
from decimal import Decimal
from scipy.spatial import Delaunay
from scipy.spatial import Voronoi

def _adjust_bounds(ax, points):
    ptp_bound = points.ptp(axis=0)
    ax.set_xlim(points[:,0].min() - 0.1*ptp_bound[0],
                points[:,0].max() + 0.1*ptp_bound[0])
    ax.set_ylim(points[:,1].min() - 0.1*ptp_bound[1],
                points[:,1].max() + 0.1*ptp_bound[1])

def voronoi_plot_2d(vor, ax=None):
    """

    """
    ax = fig.gca()
    if vor.points.shape[1] != 2:
        raise ValueError("Voronoi diagram is not 2-D")

    ax.plot(vor.points[:,0], vor.points[:,1], '.')
    #ax.plot(vor.vertices[:,0], vor.vertices[:,1], 'o')

    for simplex in vor.ridge_vertices:
        simplex = np.asarray(simplex)
        if np.all(simplex >= 0):
            ax.plot(vor.vertices[simplex,0], vor.vertices[simplex,1], 'k-')

    ptp_bound = vor.points.ptp(axis=0)

    center = vor.points.mean(axis=0)
    for pointidx, simplex in zip(vor.ridge_points, vor.ridge_vertices):
        simplex = np.asarray(simplex)
        if np.any(simplex < 0):
            i = simplex[simplex >= 0][0]  # finite end Voronoi vertex

            t = vor.points[pointidx[1]] - vor.points[pointidx[0]]  # tangent
            t /= np.linalg.norm(t)
            n = np.array([-t[1], t[0]])  # normal

            midpoint = vor.points[pointidx].mean(axis=0)
            direction = np.sign(np.dot(midpoint - center, n)) * n
            far_point = vor.vertices[i] + direction * ptp_bound.max()

            ax.plot([vor.vertices[i,0], far_point[0]],
                    [vor.vertices[i,1], far_point[1]], 'k--')

    _adjust_bounds(ax, vor.points)

    return ax.figure

CImap = np.loadtxt('STARTmap-working copy.txt',dtype='float', delimiter=',')

x=CImap[:,0]
X=x-np.min(x)

y=CImap[:,1]
Y=y-np.min(y)

CI=CImap[:,2]
CIlog=np.log10(CI)

triang = tri.Triangulation(x,y)
points=CImap[:,0:2]

fig = plot.figure(figsize=(11,7), dpi=100)
plot.hold(True)
vor = Voronoi(points)
voronoi_plot_2d(vor)
#plot.triplot(x,y)
plot.figure(num=1,figsize=(12,12),dpi=300)
plot.tricontourf(triang,CIlog,45, cmap=cm.coolwarm)
cb = plot.colorbar()
cb.set_label('$Log_{10} (CI)$')
plot.tricontour(triang,CIlog, 45, cmap=cm.coolwarm)
plot.gca().set_aspect('equal')
#plot.gca().axis('off')
plot.title('V&A Corrosion Index (CI) Contours')
plot.legend()
plot.xlabel('X')
plot.ylabel('Y')
plot.show()
