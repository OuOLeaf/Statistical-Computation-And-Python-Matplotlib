#%% 

import numpy as np
from scipy.stats import norm, chi2, beta, t, f, poisson
import matplotlib.pyplot as plt
from matplotlib import cm
#%% Normal

# pdf

# setting
fig, ax = plt.subplots(2, 1, figsize=(10,8))

# different mu
n = 1000
x = np.linspace(-4, 8, n).reshape(n, 1)
mu = np.arange(5)
y = norm.pdf(x, loc = mu)
mu_labels = ["$\mu = " + str(u) + "$" for u in mu]
ax[0].plot(x, y, lw = 3, alpha = 0.7)
ax[0].legend(mu_labels)
ax[0].set_title("Normal different mu")

# different sigma
x = np.linspace(-8, 8, n).reshape(n, 1)
sigma = np.arange(5) + 1
y = norm.pdf(x, scale = sigma)
sigma_labels = np.array(["$\sigma = " + str(u) + "$" for u in sigma])
ax[1].plot(x, y, lw = 3, alpha = 0.7)
ax[1].legend(sigma_labels)
ax[1].set_title("Normal different sigma")
#%%
# setting
fig, ax = plt.subplots(2, 1, figsize=(10,8))

# cdf
z = np.linspace(-4, 8, n).reshape(n, 1)
F = norm.cdf(z , loc = mu)
ax[0].plot(z, F, lw = 3, alpha = 0.7)
ax[0].legend(mu_labels)
z = np.linspace(-8, 8, n).reshape(n, 1)
F = norm.cdf(z , scale = mu)
ax[1].plot(z, F, lw = 3, alpha = 0.7)
ax[1].legend(sigma_labels)

#%% Chi-square
# different df
n = 1000
x = np.linspace(0, 50, n).reshape(n, 1)
df = np.arange(5, 31, 5) 
y = chi2.pdf(x, df = df)
df_labels = np.array(["df = " + str(u)  for u in df])
cmap = cm.get_cmap("Blues")
index = np.linspace(0.3, 0.8, np.size(df))
color1 = cmap(index)
for i in range(np.size(df)):
    plt.plot(x, y[:,i], lw = 3, color = color1[i])
    plt.legend(df_labels)
    plt.title("Chi-Square different df")



#%% T
# different df
n = 1000
x = np.linspace(-8, 8, n).reshape(n, 1)
df = np.arange(0.1, 31, 0.6) 
y = t.pdf(x, df = df)
cmap = cm.get_cmap("Purples")
index = np.linspace(0.3, 0.5, np.size(df))
colors = cmap(index)
df_labels = np.array(["df =  " + str(u)  for u in df])
for i in range(np.size(df)):
    plt.plot(x, y[:,i], lw = 2, color = colors[i])
plt.plot(x, norm.pdf(x), lw = 3, color = "red", alpha = 0.6, label = r"$N(0, 1)$")    
plt.xlim(-6, 6)
plt.legend()
plt.title(r"$T$ different df")


#%% beta 
n = 1000
def betadraw(a, b, axis = [], title = "", labelshow = False):
    x = np.linspace(0, 1, n).reshape(n, 1) 
    y = beta.pdf(x, a, b)
    a_b_labels = ["a = " + str(al) + r", b = " + str(be) for al, be in zip(a, b)]
    cmap = cm.get_cmap("Greens")
    index = np.linspace(0.3, 1, np.size(b))
    colors = cmap(index)
    for i in range(np.size(a)):
        plt.plot(x, y[:, i], color = colors[i])
    if axis:
        plt.axis(axis)
    if labelshow:
        plt.legend(a_b_labels, loc = "upper right")
    plt.title(title)
    plt.show()
# %%
# bell shape

a = np.arange(3, 11)
b = np.arange(3, 11) 
betadraw(a, b, title = "a and b are simultaneously increasing")
# u shape
a = np.linspace(0.5, 1, 10, endpoint = True)
b = np.linspace(1, 0.5, 10, endpoint = True)
betadraw(a, b, [-0.05, 1.05, 0.2, 5], title = "a + b = 1")

# straight lines
a = np.array([1, 1, 2])
b = np.array([2, 1, 1])
betadraw(a, b, labelshow = True)

#%%
n = 1000
bodya = np.linspace(8, 2, 100, endpoint = True)
bodyb = np.linspace(2, 8, 100, endpoint = True)
x = np.linspace(0, 1, n).reshape(n, 1) 
y = beta.pdf(x, bodya, bodyb)
a_b_labels = ["a = " + str(al) + r", b = " + str(be) for al, be in zip(a, b)]
cmap = cm.get_cmap("PuOr")
index = np.linspace(0, 1, np.size(bodyb))
colors = cmap(index)
for i in range(np.size(bodyb)):
        plt.plot(x, y[:, i], color = colors[i])


#%% F

n = 1000
def fdraw(df1, df2, axis = [], showlabel = True, title = ""):
    x = np.linspace(0, 3, n).reshape(n, 1) 
    # nominator denominator
    y = f.pdf(x, dfn = df1, dfd = df2)
    cmap = cm.get_cmap("Purples")
    index = np.linspace(0.3, 1, np.size(df1))
    colors = cmap(index)
    dfs_labels = ["$df_n$ = " + str(d1) + ", $df_d$ = " + str(d2) for d1, d2 in zip(df1, df2)]
    plt.figure(figsize = [8, 5])
    for i in range(np.size(df1)):
        plt.plot(x, y[:, i], color = colors[i])
    if axis:
        plt.axis(axis)
    if showlabel:
        plt.legend(dfs_labels, loc = "upper right", framealpha = 0.7)
    plt.title( title, fontsize = 20)
    plt.show()
dfn = np.arange(5, 101, 5)
dfd = np.arange(5, 101, 5) 
fdraw(dfn, dfd, showlabel = False, title = "The two same df from 5 to 100")
#%%

df1 = np.arange(5, 10, 2) 
df2 = np.arange(15, 20, 2)
prob = np.linspace(0, 1, n, endpoint = True).reshape(1000, 1)
cmap = cm.get_cmap("Greens")
index = np.linspace(0.3, 0.8, np.size(df1))
color1 = cmap(index)
cmap = cm.get_cmap("Purples")
index = np.linspace(0.3, 0.8, np.size(df1))
color2 = cmap(index)
for i in range(np.size(df1)):
    y1 = f.ppf(prob, df1[i], df2[i])
    y2 = f.ppf(1 - prob, df2[i], df1[i])
    plt.plot(y1, prob, color = color1[i], lw = 3)
    plt.plot(y2, prob, color = color2[i], lw = 3)
plt.xlim((0, 7))
plt.ylabel("F statistics")
plt.xlabel("Cumulative")
plt.legend([r"$F_{a, v1, v2}$", r"$F_{-a, v2, v1}$"], loc = "upper right", prop={'size': 13})
plt.plot()
plt.text(3, 0.5, r"$F_{-a, v1, v2} = \dfrac{1}{F_{a, v2, v1}}$", fontsize = 15)
plt.title("Reciprocal Property of F-distribution")
plt.show()

#%%
# discrete poisson
plt.style.use('dark_background')
plt.grid(alpha = 0.4)
nx = 20
x = np.arange(0, nx)
lam = np.array([4, 6, 8, 10]).reshape(4, 1)
n = np.size(lam)
ypmf = poisson.pmf(x , lam)
ycdf = poisson.cdf(x, lam)
colors = ["pink", "yellow", "orange", "red"]
shape = ["^", "s", "o", "*"]
for i in range(n):
    y = ypmf[i]
    plt.vlines(x, 0, y, color = colors[i], alpha = 0.5, label = r"$\lambda$ = {}".format(int(lam[i])))
    plt.plot(x, y, "o", color = colors[i], alpha = 0.5, marker = shape[i])
ax = plt.gca()
ax.set_xticks(np.arange(0, 21, 1))
plt.legend()
plt.show()
#%%
nx = 20
x = np.arange(0, nx)
lam = np.array([4, 6, 8, 10]).reshape(4, 1)
n = np.size(lam)
ypmf = poisson.pmf(x , lam)
ycdf = poisson.cdf(x, lam)
for j in range(n):
    y = ycdf[j]
    plt.step(x, y, alpha = 0.7, lw = 2, label = r"$\lambda$ = {}".format(int(lam[j])) )
plt.legend(bbox_to_anchor = (0.9, 0.55))
ax = plt.gca()
ax.set_xticks(np.arange(0, 21, 1))
plt.show()

# %%
# Way1 4 subplots
fig, ax = plt.subplots(2, 2, figsize = (10, 8))
k = 0
for row in range(2):
    for col in range(2):
     y = ypmf[k]
     ax[row, col].plot(x, y, "o", color = colors[k], alpha = 0.7)
     ax[row, col].vlines(x, 0, y, color = colors[k], alpha = 0.7)
     ax[row, col].set_title(r"$\lambda = {}$".format(int(lam[k])))
     k = k + 1
plt.show()
#%%
# Way2 three dimension
plt.figure(figsize = (10, 8))
plt.rcParams['grid.color'] = (0.5, 0.5, 0.5, 0.4)
ax = plt.axes(projection='3d')
ax.view_init(35, 53)
for i in range(4):
    y = list(range(20))
    z = list(ypmf[i])
    x = [int(lam[i])+.315] * 20
    ax.scatter3D(x, y, z, c = colors[i], label = r"$\lambda$ = {}".format(int(lam[i])))
    for j in range(nx):
        zline = [0, z[j]]
        xline = x[:2]
        yline = [y[j]] * 2
        ax.plot3D(xline, yline, zline, c = colors[i], lw = 3)
plt.legend()
plt.title("Poisson with different rate", fontsize = 20)
ax.w_xaxis.pane.fill = False
ax.w_yaxis.pane.fill = False
ax.w_zaxis.pane.fill = False
ax.set_yticks(np.arange(0, 20, 2))
ax.set_xlim(3.5, 10.5)
plt.show()
plt.style.use("default")
#%%
