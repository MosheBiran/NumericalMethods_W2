from mealpy.swarm_based.GWO import BaseGWO
from opfunu.cec_basic.cec2014_nobias import *
from mealpy.swarm_based.PSO import BasePSO, PPSO, PSO_W, HPSO_TVA
from mealpy.math_based.HC import OriginalHC
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import random
from scipy.stats import multivariate_normal
n=10
m=5
ul=20
r = 0.3
w = 0.03
d=0.5

def gauss2d(mu, sigma, to_plot=False):

    std = [np.sqrt(sigma[0]), np.sqrt(sigma[1])]
    x_values = np.linspace(-10, 10, 120)
    y_values = np.linspace(-10, 10, 120)
    x, y = np.meshgrid(x_values, y_values)



    x_ = x.flatten()
    y_ = y.flatten()
    xy = np.vstack((x_, y_)).T
    normal_rv = multivariate_normal(mu, sigma)
    z = normal_rv.pdf(xy)
    z = z.reshape(120, 120, order='F')**(1 / len(mu))

    if to_plot:
        plt.contourf(x, y, z.T)
        plt.show()

    return z
def gaussmultiD(mu, sigma):

    std = np.sqrt(sigma)
    x_values = np.linspace(-10, 10, 120)
    y_values = np.linspace(-10, 10, 120)
    x, y = np.meshgrid(x_values, y_values)



    x_ = x.flatten()
    y_ = y.flatten()
    xy = np.vstack((x_, y_)).T

    normal_rv = multivariate_normal(mu, sigma)


    return normal_rv

def plot3d():
    fig = plt.figure(figsize=(13, 7))

    ax = plt.axes(projection='3d')



    random.seed(4)
    t=1
    gr=0.05
    z=1
    for i in range(6):
        x_values = np.linspace(-10, 10, 120)
        y_values = np.linspace(-10, 10, 120)
        X, Y = np.meshgrid(x_values, y_values)
        mu=[]
        var=[]
        for j in range(2):
            mu.append(random.uniform(-5, 5))
            var.append(random.uniform(0.06, 0.6))



        '''for mu, sig in [(-2, 0.5), (0, 0.15)]:
            plt.plot(x_values, gaussian(x_values, mu, sig))'''

        if(t==1):
            z = gauss2d(mu, var) * t
            t=r
        else:
            z = np.maximum(z,gauss2d(mu, var) * t)
            t=r*(1-gr)
            gr+=0.05
    #z=(z - np.min(z))/np.ptp(z)
    surf=ax.plot_surface(X, Y, z, rstride=1, cstride=1, cmap='coolwarm', edgecolor='none')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('PDF')
    ax.set_title('Surface plot of Gaussian 2D KDE')
    #fig.colorbar(surf, shrink=0.5, aspect=5) # add color bar indicating the PDF
    ax.view_init(15, 45)
    plt.show()

def hc(dem=2):
    random.seed(4)
    t = 1
    gr = 0.05
    z = 1
    gussiandemlist=[]
    for i in range(6):
        mu=[]
        var=[]
        for j in range(dem):
            mu.append(random.uniform(-5, 5))
            var.append(random.uniform(0.06, 0.6))

        if(t==1):
            z = gaussmultiD(mu, var)
            gussiandemlist.append([z,t])
            t=r
        else:
            z =  gaussmultiD(mu, var)
            gussiandemlist.append([z, t])
            t=r*(1-gr)
            gr+=0.05
    def targetfunc(solution):
        templist=[]
        for f in gussiandemlist:
            templist.append(f[0].pdf(solution)**(1/dem)*f[1])
        return -max(templist)

    obj_func = targetfunc  # This objective function come from "opfunu" library. You can design your own objective function like above
    verbose = True  # Print out the training results
    epoch = 1000  # Number of iterations / generations / epochs
    pop_size = 50  # Populations size (Number of individuals / Number of solutions)

    # A - Different way to provide lower bound and upper bound. Here are some examples:

    ## 1. When you have different lower bound and upper bound for each variables
    lb1 = [-10]*dem
    ub1 = [10]*dem

    md1 = OriginalHC(obj_func, lb1, ub1, verbose, epoch, pop_size)
    best_pos1, best_fit1, list_loss1 = md1.train()
    print(best_pos1)
def gwo(dem=2):
    random.seed(4)
    t = 1
    gr = 0.05
    z = 1
    gussiandemlist=[]
    for i in range(6):
        mu=[]
        var=[]
        for j in range(dem):
            mu.append(random.uniform(-5, 5))
            var.append(random.uniform(0.06, 0.6))

        if(t==1):
            z = gaussmultiD(mu, var)
            gussiandemlist.append([z,t])
            t=r
        else:
            z =  gaussmultiD(mu, var)
            gussiandemlist.append([z, t])
            t=r*(1-gr)
            gr+=0.05
    def targetfunc(solution):
        templist=[]
        for f in gussiandemlist:
            templist.append(f[0].pdf(solution)**(1/dem)*f[1])
        return -max(templist)

    obj_func = targetfunc  # This objective function come from "opfunu" library. You can design your own objective function like above
    verbose = True  # Print out the training results
    epoch = 100  # Number of iterations / generations / epochs
    pop_size = 50  # Populations size (Number of individuals / Number of solutions)

    # A - Different way to provide lower bound and upper bound. Here are some examples:

    ## 1. When you have different lower bound and upper bound for each variables
    lb1 = [-10]*dem
    ub1 = [10]*dem

    md1 = BaseGWO(obj_func, lb1, ub1, verbose, epoch, pop_size)
    best_pos1, best_fit1, list_loss1 = md1.train()
    print(best_pos1)

gwo(5)