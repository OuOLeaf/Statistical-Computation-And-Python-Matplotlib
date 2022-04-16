#%%
from matplotlib import transforms
import numpy as np
import matplotlib.pyplot as plt 
import scipy.optimize as opt
from scipy.misc import derivative
import scipy.integrate as integrate
import pandas as pd
#%%
plt.style.use('dark_background')
# %% 1
# x 要大於 -1 
# x很大時 值很大
# colors = ["lawngreen", "deepskyblue", "yellow", "pink", "orange", "red"]
plt.style.use('dark_background')

f = lambda x: np.sqrt((x ** 2 + 1) / (x + 1))
x = np.linspace(-1, 10, 1000)
minif = opt.minimize_scalar(f)
x2 = np.linspace(0, 2, 1000)
fd1 = derivative(f, x2, dx = 0.001, n = 1)

fig, ax = plt.subplots(1, 2, figsize= (10, 4))

ax[0].grid(True, alpha = 0.3)
ax[0].plot(np.linspace(0, 2, 1000), fd1, lw = 2)
ax[0].axhline(y = 0, color = 'yellow', lw = 1)
ax[0].plot(0.41, 0, marker = 'o')
ax[0].text(0.41+0.2, 0+0.03 , '(0.41, 0)')
ax[0].text(0.41+0.1, -0.06 , 'Extremum emerges at x = 0.41')

ax[0].set_title('1st Derivative')
ax[0].axis([-.1, 2.1, -0.55, 0.4])
ax[1].plot(x, f(x), lw = 2)
ax[1].grid(True, alpha = 0.3)
ax[1].plot(minif.x, f(minif.x), marker = 'o')
ax[1].text(minif.x - 0.8, f(minif.x) + 1, '({:.2f}, {:.2f})'.format(minif.x, f(minif.x)))
ax[1].text(0.5, 0.7, r'$\min_x\sqrt{\dfrac{x^2 + 1}{x + 1}}$', fontsize = 15, transform = ax[1].transAxes)
ax[1].set_title('Function')
plt.show()

#%%
f = lambda x: ((x + 1) ** 5) * np.sin(x - 3)
x = np.linspace(2, 3, 1000)
fd1 = derivative(f, x, dx = 0.001, n = 1)
fig, ax = plt.subplots(2, 2, figsize= (10, 8))
ax[0, 0].plot(x, fd1, lw = 2)
# ax[0, 0].text(-3.3, 100, 'Extremum emerges at x from 2 to 3')

x2 = np.linspace(-3, 0, 1000)
fd2 = derivative(f, x2, dx = 0.001, n = 1)
ax[0, 1].plot(x2, fd2, lw = 2)
ax[0, 1].text(-2.7, -3, 'Extremum emerges at x from -3 to -2.5')

x3 = np.linspace(-1.5, 0, 1000)
fd3 = derivative(f, x3, dx = 0.001, n = 1)
ax[1, 0].plot(x3, fd3, lw = 2)
ax[1, 0].text(-1.5, -0.5, 'Extremum emerges at x from -0.5 to 0')

x4 = np.linspace(-1.25, -0.75, 1000)
fd4 = derivative(f, x4, dx = 0.001, n = 1)
ax[1, 1].plot(x4, fd4, lw = 2)
ax[1, 1].text(-1.2, 0.0155, 'Extremum emerges at x from -1.1 to -0.9')

plt.suptitle('1st Derivative', fontsize = 30)

#%%
f = lambda x: ((x + 1) ** 5) * np.sin(x - 3)
x = np.linspace(-4, 3, 1000)
fig, ax = plt.subplots(2, 2, figsize= (10, 8))
ax[0, 0].plot(x, f(x), lw = 2)
ax[0, 0].text(-3.3, 30, 'Minimum emerges at x from 2 to 3')


x2 = np.linspace(-3, 0, 1000)
fd2 = derivative(f, x2, dx = 0.001, n = 1)
ax[0, 1].plot(x2, f(x2), lw = 2)
ax[0, 1].text(-2.35, -6, 'Minimum emerges at x from -3 to -2.5')

x3 = np.linspace(-1.5, 0, 1000)
fd3 = derivative(f, x3, dx = 0.001, n = 1)
ax[1, 0].plot(x3, f(x3), lw = 2)
ax[1, 0].text(-1.4, -0.05, 'Maximum emerges at x from -0.5 to 0')


x4 = np.linspace(-1.1, -0.9, 1000)
fd4 = derivative(f, x4, dx = 0.001, n = 1)
ax[1, 1].plot(x4, f(x4), lw = 2)
ax[1, 1].text(-1.07, -0.000005, 'Inflection point seems to emerge at \nx from -1.1 to -0.9')

plt.suptitle('function', fontsize = 30)

#%%

mini_neg_f = opt.minimize_scalar(f, bounds=(2, 3), method='bounded')
mini_pos_f = opt.minimize_scalar(f, bounds=(-3, -2.5), method='bounded')
max_pos_f = opt.minimize_scalar(lambda x:-1 * f(x), bounds=(-0.5, 0), method='bounded')
print(mini_neg_f.x, mini_pos_f.x, max_pos_f.x)

x = np.linspace(-4, 3, 1000)
plt.plot(x, f(x), alpha= 0.7)
plt.grid(True, alpha = 0.3)
plt.scatter([mini_neg_f.x, mini_pos_f.x], [f(mini_neg_f.x), f(mini_pos_f.x)], marker = 'o', color = 'yellow', label = 'minimum')
plt.scatter(max_pos_f.x, f(max_pos_f.x), marker = 'o', color = 'pink', label = 'maximum')
plt.text(0.12, 0.3, r'$\min_{-4 \leq x \leq 3}\;(x + 1)^5\;sin(x - 3)$', fontsize = 15, transform = plt.gca().transAxes)
plt.text(mini_neg_f.x - 2, f(mini_neg_f.x), '({:.2f}, {:.2f})'.format(mini_neg_f.x, f(mini_neg_f.x)))
plt.text(mini_pos_f.x - 0.5 , f(mini_pos_f.x) - 25, '({:.2f}, {:.2f})'.format(mini_pos_f.x, f(mini_pos_f.x)))
plt.text(max_pos_f.x - 0.5, f(max_pos_f.x) + 25, '({:.2f}, {:.2f})'.format(max_pos_f.x, f(max_pos_f.x)))
plt.legend()
#%%

df = lambda t: np.sqrt(1 + (t ** 2)) 
x = np.linspace(0, 8, 1000)
lx = lambda x: integrate.quad(df, 0, x)[0] - 10
vlx = np.vectorize(lx)
r = opt.root_scalar(lx, bracket = [0, 5])
plt.grid(True, alpha= 0.4)
plt.plot(x, vlx(x), lw = 2)
plt.plot(np.linspace(0, r.root, 1000), [0] * 1000, color = 'orange', linestyle = "--")
plt.plot([r.root] * 1000, np.linspace(-10, 0, 1000), color = 'orange', linestyle = "--")
plt.plot(r.root, 0, marker = 'o', alpha = 0.4)
plt.text(3.3, 2, '({:.2f}, {})'.format(r.root, 0))
plt.text(1, 15, r"$L(x) = \int^{x}_a \sqrt{1 + (f'(t))^2} dt$", fontsize = 15)


# %%


lambda3 = np.random.exponential(scale = 1/3, size = 1000)
lambda4 = np.random.exponential(scale = 1/4, size = 1000)
lambda5 = np.random.exponential(scale = 1/5, size = 1000)
for i in range(3):
    f = lambda l:(1000 * np.log(l) - l * (np.sum({0:lambda3, 1:lambda4, 2:lambda5}.get(i, "no data"))))
    fl = np.vectorize(f)
    x = np.linspace(0.5, 8, 10000)
    y = fl(x)
    plt.grid(True, alpha = 0.3)
    plt.plot(x, y, lw = 2, label = r'$\lambda = {}$'.format(i+1))

    plt.axvline(x = 2 + i )
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
props = dict(boxstyle='round', facecolor='black', alpha=0.7)
plt.text(0.2, 0.15, r"$\;ln \prod_{i=1}^{N}\;-\lambda e^{-\lambda x_i}$", fontsize = 15, bbox=props, transform = plt.gca().transAxes)

# %%
def switch(k):
    f1 = lambda l: -1 * (n[j] * np.log(l) - l * (np.sum(allsamples)))
    f2 = lambda l: -1 * ((l ** n[j]) * np.exp(- l * np.sum(allsamples)))
    f3 = lambda s: 1 / np.mean(s)
    return({'log_MLE': opt.minimize_scalar(f1, bounds = [0.1, 10]).x, 'MLE': opt.minimize_scalar(f1, bounds = [0.1, 10]).x, 'real_estimate':f3(allsamples)}.get(k, 'no exists'))

n = np.array([10, 20, 30, 50, 100, 300, 500])
res = [0]  * len(n)
var = [0]  * len(n)
dfmatrix = np.empty(shape = (len(n), 2))
ways = ["log_MLE", 'real_estimate', "MLE"]
dfmatrix = np.empty(shape = (len(n), 2))
for k in range(2):
    for j in range(len(n)):
        simulation = [0] * 10000
        for i in range(10000):
            allsamples = np.random.exponential(scale = 1/2, size = n[j])
            simulation[i] = switch(ways[k])
        res[j] = np.mean(simulation)
        if k == 0:
            var[j] = np.std(simulation)
    dfmatrix[:, k] = res
#%%
expMLE = pd.DataFrame(dfmatrix, columns = ways[:2])
expMLE['n']= n
expMLE.set_index('n', inplace = True)
expMLE.T.head()
plt.plot(n.astype(str), expMLE['log_MLE'], label = 'log_MLE', marker = 's')
plt.plot(n.astype(str), expMLE['real_estimate'], label = 'real_estimate', marker = 's')
plt.text(0.4, 0.5, r"$\max_{\lambda}\;ln \prod_{i=1}^{N}\;-\lambda e^{-\lambda x_i}$", fontsize = 15, transform = plt.gca().transAxes)
plt.legend()