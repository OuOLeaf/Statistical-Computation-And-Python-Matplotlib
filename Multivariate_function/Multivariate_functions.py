#%%
from numpy.lib.arraysetops import setxor1d
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt
from scipy.stats import norm
#%%
k = np.empty([6, 5])
k[0, :] = [1, 1, 1, 1, 1]
k
#%% Mixtured Normal
# mu1 = 2 mu2 = 5 sigma1 = sigma2 = 2 p1 = 0.4

m1 = 0
m2 = 4
s1 = 2
s2 = 1
p1 = 0.6

n = 1000
n1 = np.random.binomial(n, p1)
n2 = n - n1

#%% 兩個矩陣合併 sampling
sample = np.r_[norm.rvs(size = n1, loc = m1, scale = s1), norm.rvs(size = n2, loc = m2, scale = s2)]
#%% estimate
f = lambda p: -1 * np.sum(np.log(p[0] * norm.pdf(sample, loc = p[1], scale = p[2]) + (1 - p[0]) * norm.pdf(sample, loc = p[3], scale = p[4])))
# L = lambda x : -np.sum(np.log(x[0] * beta.pdf(sample, x[1], x[2]) + (1 - x[0]) * beta.pdf(sample, x[3], x[4])))
bnd = [(0, 1), (0, np.inf), (0.01, np.inf), (0, np.inf), (0.01, np.inf)]
x0 = [0.1, 2, 2, 1.5, 1.5]
opts = dict(disp = True, maxiter = 1e4)
res = opt.minimize(f, bounds = bnd, tol = 1e-8, x0 = x0)
print(res)

%store 
#%%
plt.style.use('dark_background')
x = np.linspace(-10, 20, 10000)
pdf = p1 * norm.pdf(x, m1, s1) + (1 - p1) * norm.pdf(x, m2, s2)
pdf1 = norm.pdf(x, m1, s1)
pdf2 = norm.pdf(x, m2, s2)
estpdf = res.x[0] * norm.pdf(x, res.x[1], res.x[2]) + (1 - res.x[0]) * norm.pdf(x, res.x[3], res.x[4])
plt.hist(sample, density = True, edgecolor = 'orange', alpha = 0.9, color = 'mediumseagreen')
plt.plot(x, pdf, lw = 3, label = 'True Mixture', color = 'hotpink')
plt.plot(x, estpdf, lw = 3, label = 'Estimate Mixture', color = 'cyan')
plt.plot(x, pdf1, x, pdf2, alpha = 0.7, color = 'pink', linestyle = '--')
plt.legend()

#%%
plt.style.use('dark_background')

def mixnormal(params, seed, x0):
    p1, m1, s1, m2, s2 = params[0], params[1], params[2], params[3], params[4]
    sample_size = [50, 100, 300, 500, 1000, 10000]
    k = np.empty([6, 5])
    
    
    fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3, figsize = [15, 10])
    for i, n in enumerate(sample_size):
        n1 = np.random.binomial(n, p1)
        n2 = n - n1

        sample = np.r_[norm.rvs(size = n1, loc = m1, scale = s1, random_state = seed), norm.rvs(size = n2, loc = m2, scale = s2, random_state = seed)]
        f = lambda p: -1 * np.sum(np.log(p[0] * norm.pdf(sample, loc = p[1], scale = p[2]) + (1 - p[0]) * norm.pdf(sample, loc = p[3], scale = p[4])))
        # L = lambda x : -np.sum(np.log(x[0] * beta.pdf(sample, x[1], x[2]) + (1 - x[0]) * beta.pdf(sample, x[3], x[4])))
        bnd = [(0, 1), (-np.inf, np.inf), (0.01, np.inf), (-np.inf, np.inf), (0.01, np.inf)]
        x0 = [0.1, 2, 2, 1.5, 1.5]
        opts = dict(disp = True, maxiter = 1e4)
        res = opt.minimize(f, bounds = bnd, tol = 1e-8, x0 = x0, options = opts)
        k[0, :] = [res.x[0], res.x[1], res.x[2], res.x[3], res.x[4]]
        
        x = np.linspace(-10, 20, 10000)
        pdf = p1 * norm.pdf(x, m1, s1) + (1 - p1) * norm.pdf(x, m2, s2)
        pdf1 = norm.pdf(x, m1, s1)
        pdf2 = norm.pdf(x, m2, s2)
        estpdf = res.x[0] * norm.pdf(x, res.x[1], res.x[2]) + (1 - res.x[0]) * norm.pdf(x, res.x[3], res.x[4])
        ax = {1:ax1, 2:ax2, 3:ax3, 4:ax4, 5:ax5, 6:ax6}.get(i + 1, 'None')
        ax.hist(sample, density = True, edgecolor = 'orange', alpha = 0.5, color = 'lime')
        ax.plot(x, pdf, lw = 3, label = 'True Mixture', color = 'hotpink', alpha=0.9)
        ax.plot(x, estpdf, lw = 3, label = 'Estimate Mixture', color = 'cyan', alpha=0.9, linestyle = '--')
        ax.plot(x, pdf1, x, pdf2, alpha = 0.7, color = 'pink', linestyle = '-.')
        ax.set_title("n = {}".format(n))
        ax.legend()
#%%
mixnormal([0.3, 2, 1, 0, 1], x0 = [0.5, 1, 1, 1, 1], seed = 2)
#%% alpha beta
UV = pd.read_csv('./UV.txt', sep = '\t', names = ['u', 'v'], skiprows = 1)
UV.head()
# %%
UV = UV.sort_values(by = ['u'])
plt.plot(UV.u, UV.v, alpha = 0.7, color = 'purple')
plt.xlabel('u'), plt.ylabel('v')
plt.show()
# %%
## 起始點不能設定為 (0, 0)
F = lambda a, b: 1 - np.exp(-a * UV.u ** b)
f = lambda a, b: a * b * (UV.v ** (b - 1)) * np.exp(- a * (UV.v ** b))
lnMLE = lambda ab: np.log(sum(f(ab[0], ab[1]) + 1/F(ab[0], ab[1])))
opts = dict(disp = True, maxiter=1e4)
bnd = [(0.0001, np.inf), (0.0001, np.inf)]
cons = [{'type': 'ineq', 'fun': lambda x:  x[0]}, {'type': 'ineq', 'fun': lambda x:  x[1]}]
res = opt.minimize(lnMLE, x0=[2, 2], bounds = bnd, options = opts, tol = 1e-8)
pd.DataFrame(res.x, index = ['alpha', 'beta'], columns = ['lnMLE']).T

#%%
## 等高線圖
# To draw a contour plot
lnMLE = lambda a, b: np.log(sum(f(a, b) + 1/F(a, b)))
x = np.linspace(0.5, 3, 100)
y = np.linspace(0.5, 2, 100)
X, Y = np.meshgrid(x, y) # mesh grid matric
vlnMLE = np.vectorize(lnMLE)
Z = vlnMLE(X, Y)
Z
#%%
levels = np.arange(0, 8, 0.03) # levels of contour lines
contours = plt.contour(X, Y, Z, levels=levels) # check dir(contours)
# add function value on each line    
plt.clabel(contours, inline = 0, fontsize = 10) # inline =1 or 0 
cbar = plt.colorbar(contours)
plt.xlabel('X'), plt.ylabel('Y')
cbar.ax.set_ylabel('Z = f(X,Y)') # set colorbar label
# cbar.add_lines(contours) # add contour line levels to the colorbar 
plt.title('Contour Plot')
plt.grid(True)
plt.show()
# %%
import scipy.optimize as opt
import numpy as np
 
f = lambda x: (x[0] - 2)**4 + (x[0] - 2)**2*x[1]**2 + (x[1]+1)**2
opts = dict(disp = True, maxiter=1e4)
# 1
cons = [{'type': 'ineq', 'fun': lambda x:  x[0]}, {'type': 'ineq', 'fun': lambda x:  x[1]}]
bnds = [(0, np.inf), (0, np.inf)]
 
res = opt.minimize(f, x0=[0, 0], 
    bounds = bnds,
    # constraints = cons,
    options = opts,
    tol = 1e-8)
# print(res)
print('x1 = {:.4f}, x2 = {:.4f}'.format(res.x[0], res.x[1]))
print('function value is {:.4f}'.format(res.fun))
# %%
