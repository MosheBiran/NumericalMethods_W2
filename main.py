
from opfunu.cec_basic.cec2014_nobias import *
from mealpy.swarm_based.PSO import BasePSO, PPSO, PSO_W, HPSO_TVA
from mealpy.math_based.HC import OriginalHC
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import random
from scipy.stats import multivariate_normal

# n = [5, 10, 20, 40]
# m = [0, 5, 10, 20]
# ul = [10, 20, 30, 40]
# r = [0.1, 0.3, 0.6, 0.9]
# w = [0.01, 0.03, 0.06, 0.09]
# d = [0.25, 0.5, 0.75, 1.0]
# sl = []


n=10
m=5
ul=20
r = 0.3
w = 0.03
d=0.5


fig = plt.figure(figsize=(13, 7))

ax = plt.axes(projection='3d')


def gaussian(x, mu, sig):
    return 1.0001*np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))
def multiGaussian(g1,g2):
    return g1*g2

t=1
gr=0.05
z=1
for i in range(6):
    x_values = np.linspace(-10, 10, 120)
    y_values = np.linspace(-10, 10, 120)
    X, Y = np.meshgrid(x_values, y_values)
    mu1=random.uniform(-5, 5)
    mu2=random.uniform(-5, 5)
    var1=random.uniform(0.56, 0.6)
    se1= var1 ** 0.5
    var2=random.uniform(0.56, 0.6)
    se2 = var2 ** 0.5
    g1=gaussian(X,mu1,se1)
    g2=gaussian(Y,mu2,se2)

    '''for mu, sig in [(-2, 0.5), (0, 0.15)]:
        plt.plot(x_values, gaussian(x_values, mu, sig))'''

    if(t==1):
        z = multiGaussian(g1, g2) * t
        t=r
    else:
        z = np.maximum(z,multiGaussian(g1, g2) * t)
        t=r*(1-gr)
        gr+=0.05

surf=ax.plot_surface(X, Y, z, rstride=1, cstride=1, cmap='coolwarm', edgecolor='none')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('PDF')
ax.set_title('Surface plot of Gaussian 2D KDE')
#fig.colorbar(surf, shrink=0.5, aspect=5) # add color bar indicating the PDF
ax.view_init(15, 45)
plt.show()




'''# Our 2-dimensional distribution will be over variables X and Y
N = 100
X = np.linspace(-10, 10, N)
Y = np.linspace(-10, 10, N)
X, Y = np.meshgrid(X, Y)

# Mean vector and covariance matrix
mu = np.array([-2., 3.])
Sigma = np.array([[ 1. , -0.5], [-0.5,  1.]])

# Pack X and Y into a single 3-dimensional array
pos = np.empty(X.shape + (2,))
pos[:, :, 0] = X
pos[:, :, 1] = Y

def multivariate_gaussian(pos, mu, Sigma):
    """Return the multivariate Gaussian distribution on array pos."""

    n = mu.shape[0]
    Sigma_det = np.linalg.det(Sigma)
    Sigma_inv = np.linalg.inv(Sigma)
    N = np.sqrt((2*np.pi)**n * Sigma_det)
    # This einsum call calculates (x-mu)T.Sigma-1.(x-mu) in a vectorized
    # way across all the input variables.
    fac = np.einsum('...k,kl,...l->...', pos-mu, Sigma_inv, pos-mu)

    return np.exp(-fac / 2) / N

# The distribution on the variables X, Y packed into pos.
Z = multivariate_gaussian(pos, mu, Sigma)

# plot using subplots
fig = plt.figure()
ax1 = fig.add_subplot(2,1,1,projection='3d')

ax1.plot_surface(X, Y, Z, rstride=3, cstride=3, linewidth=1, antialiased=True,
                cmap=cm.viridis)
ax1.view_init(55,-70)
ax1.set_xticks([])
ax1.set_yticks([])
ax1.set_zticks([])
ax1.set_xlabel(r'$x_1$')
ax1.set_ylabel(r'$x_2$')

ax2 = fig.add_subplot(2,1,2,projection='3d')
ax2.contourf(X, Y, Z, zdir='z', offset=0, cmap=cm.viridis)
ax2.view_init(90, 270)

ax2.grid(False)
ax2.set_xticks([])
ax2.set_yticks([])
ax2.set_zticks([])
ax2.set_xlabel(r'$x_1$')
ax2.set_ylabel(r'$x_2$')

plt.show()'''
