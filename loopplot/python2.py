#%% 
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
import pandas as pd
from scipy.stats import chi2
#%%
fig, ax = plt.subplots(figsize = (10, 10))
n = 6
markerslist = list(Line2D.markers.keys())
mymarker = ["s", "*", "d", "D", "p", "o"]
rad = np.linspace(10, 20, n)
colors = plt.cm.jet(np.linspace(0,1,n))
for i in range(n):
    ax.add_patch(plt.Circle((0, 0), rad[i],\
         fill = None, color = colors[i], lw = 3))
    v = [rad[i], 0]
    angle30 = np.pi/6
    rot30 = np.array([[np.cos(angle30), -np.sin(angle30)], [np.sin(angle30), np.cos(angle30)]])
    for j in range(12):
        plt.plot(v[0], v[1], marker = mymarker[i], ms = 13, markerfacecolor="None", 
            markeredgecolor=colors[i], markeredgewidth=2)
        v = v@rot30
# ax.set_xlim(-21, 21)
# ax.set_ylim(-21, 21)
ax.grid(True)
ax.axis([-21, 21, -21, 21])
plt.show()

#%%
fig, ax = plt.subplots(figsize = (6, 6))
ax.set_aspect("equal")
ax.axis([-6, 6, -6, 6])
n = 8
angle = (2 * np.pi)/n
rot = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
linebox = np.array([[0, 0], [3, 0], [3, -1], [5, -1], [5, 1], [3, 1], [3, 0]])
for i in range(n):
    plt.plot(linebox[:, 0], linebox[:, 1])
    linebox = linebox@rot 

# %%
def drawmany(n):
    fig, ax = plt.subplots(figsize = (6, 6))
    ax.set_aspect("equal")
    ax.axis([-6, 6, -6, 6])
    angle = (2 * np.pi)/n
    rot = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    linebox = np.array([[0, 0], [3, 0], [3, -1], [5, -1], [5, 1], [3, 1], [3, 0]])
    for i in range(n):
        plt.plot(linebox[:, 0], linebox[:, 1])
        linebox = linebox@rot 
# %%
drawmany(68)

# %%

# 課本是以右邊面積累積面積，函式是左邊累積面積 
F = np.array([0.995, 0.99, 0.975, 0.95, 0.9, 0.1, 0.05, 0.025, 0.01, 0.005]) # cumalative to F
df = 1
pd.set_option('display.float_format', lambda x: '%.3f' % x)
chiarray = np.array([0])
for i in range(29):
    x = chi2.ppf(1 - F, i + 1)  # inverse of CDF
    chiarray = np.append(chiarray, x)
chiarray = chiarray[1:].reshape(29, 10)

chitable = pd.DataFrame(chiarray, columns = F)
#%%
chitable["degree of freedom"] = np.linspace(1, 29, 29, endpoint = True)
chitable.set_index("degree of freedom", inplace = True)

# %%
chitable.to_excel("D:\statistics_python\chisquare_table.xlsx",
        float_format = '%.3f', sheet_name = "卡方分配-累積分布函數與自由度之機率")

# %%
print(chitable)

#%%

z = np.random.normal(0, 1, 100000)
z2 = z**2
chi = np.random.chisquare(1, 100000)
z2hist, z2n, _ = plt.hist(z2, bins = 300, alpha = 0, density = True)
chihist, chin,_ = plt.hist(chi, bins = 300, alpha = 0, density = True)
chin = (chin[1:] + chin[:-1])/2
z2n = (z2n[1:] + z2n[:-1])/2
plt.plot(np.insert(z2n, 0, 0), np.insert(z2hist, 0, 0), color = "green", alpha = 0.6, label = r"$Z^2$")
plt.plot(np.insert(chin, 0, 0), np.insert(chihist, 0, 0), color = "red", alpha = 0.3, label = r"$\chi^2$")
plt.axis([-0.2, 5, 0, np.max([z2hist, chihist])+0.3])
plt.legend()
plt.xlabel("")
plt.ylabel("")
plt.show()
# %%
print(len(chihist), len(chin))
# %%
# setting
x = np.linspace(0, 8, 1000)
dfs = 5

# draw
for k in range(dfs):
    df = k + 1
    chi = chi2.pdf(x, df = df)
    plt.plot(x, chi, lw = 2, label = "k=" + str(df), color = colors[k])

# adjust figure and show
plt.title("機率密度函數", fontproperties="SimSun",fontsize=15)
plt.legend(loc = "center right", bbox_to_anchor = (0.9, 0.6), frameon=False)
plt.axis([0, 8, 0, 1])
plt.show()
#%%
# setting
y = np.linspace(0, 1, 1000, endpoint = True)
colors = ["black", "blue", "green", "red", "purple"]

# draw
for k in range(dfs):
    df = k + 1
    chi = chi2.ppf(y, df = df)
    plt.plot(chi,  y, lw = 2, label = "k=" + str(df), color = colors[k])

# adjust figure and show
plt.title("累積分布函數", fontproperties="SimSun",fontsize=15)
plt.legend(loc = "center right", bbox_to_anchor = (0.9, 0.3), frameon=False)
plt.axis([0, 8, 0, 1])
plt.show()

# %%
n = 6
x = np.linspace(0, n, 1000)
amplitude = 0.5
y = np.cos(x * 2 * np.pi) + 1 

plt.plot(x, y)
plt.show()
# %%
theta = np.linspace(0, 2 * np.pi, 1000)
plt.polar(theta, y, alpha = 0)
plt.plot(0, 0, marker = "o", ms = 12, color = "red")
axes = plt.gca()
axes.set_yticklabels([])
axes.set_xticklabels([])
axes.grid(False)
axes.fill(theta,y,'orange')
axes.spines['polar'].set_visible(False)
plt.show()
# %%
