#%%
from scipy.stats import skew, kurtosis, norm, chi2, t, kstest, anderson, jarque_bera
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import seaborn as sns
from statsmodels.distributions.empirical_distribution import ECDF

# %% G3
def G3(n):
    m = 10000
    g2 = lambda x: np.sqrt(len(x)/ 24) * kurtosis(x)
    g1 = lambda x: np.sqrt(len(x)/ 6) * skew(x)
    g3 = lambda x: g2(x) ** 2 + g1(x) ** 2
    g3s = [0] * m
    for i in range(m):
        x_normal = np.random.normal(size = n, loc = 0, scale = 1)
        g3s[i] = g3(x_normal)
    return g3s
#%%
def obnormal(data):
    # 直方圖
    # histogram
    # connect the centre of bar
    histy, histdata, _= plt.hist(data, density = True)
    histx = (histdata[1:] + histdata[:-1])/2
    plt.plot(histx, histy, color = "red")
    # normal line
    x = np.linspace(-4, 4, 1000)
    y = norm.pdf(x, loc = 0, scale = 1)
    plt.plot(x, y, color = "purple")
    plt.show()

    # qqplot
    data = np.array(data)
    fig = sm.qqplot(data, line='45')

    # Grab the lines with blue dots
    dots = fig.findobj(lambda x: hasattr(x, 'get_color') and x.get_color() == 'b')

    _ = [d.set_alpha(0.3) for d in dots]

    plt.show()

    # 檢定
    mu, sigma = np.mean(data), np.std(data)
    y = np.random.normal(size = 1000,loc =  mu, scale = sigma)
    s, pvalue = kstest(data,'norm')
    pvalue = ">= 0.05" if pvalue > 0.05 else "< 0.05"
    necdf = ECDF(y)
    dataecdf = ECDF(data)
    plt.plot(necdf.x, necdf.y, alpha = 0.5, lw = 3)
    plt.plot(dataecdf.x, dataecdf.y, alpha = 0.5, lw = 3)
    t = "statistics = {}, p-value {}".format(round(s, 5), pvalue)
    plt.suptitle("Kolmogorov test")
    plt.title(t)
    plt.xlabel("x")
    plt.ylabel("cdf")
    plt.show()
# %%
g310 = G3(10)
g3100 = G3(100)
#%%
fig, ax = plt.subplots(1, 2)
sm.qqplot(np.array(g310), line='45', ax = ax[0])
sm.qqplot(np.array(g310), line='45', ax = ax[1])
#%% okay
histy, histdata, _= plt.hist(g3s, density = True, bins = 60)
histx = (histdata[1:] + histdata[:-1])/2
plt.plot(np.insert(histx, 0, 0), np.insert(histy, 0, 0), alpha= 0)
plt.plot(np.linspace(0, 20, 1000), chi2.pdf(np.linspace(0, 20, 1000), df = 2))

# %% 
s, pvalue = kstest(g3s,'chi2', args = (2, ))
pvalue = ">= 0.05" if pvalue > 0.05 else "< 0.05"    
fig = sm.qqplot(np.array(g3s), chi2, distargs=(2, ), line = '45')
t = "statistics = {}, p-value {}".format(round(s, 5), pvalue)
plt.suptitle("Kolmogorov test")
plt.title(t)
plt.show()
#%%

#%%
fig, ax = plt.subplots(1, 2, figsize = (10, 4))
n = 10000
y = np.random.chisquare(df = 2, size = n)
ax[0].hist(g3s, alpha = 0.3, lw = 3, label = "$\chi^2_{(2)}$")
ax[0].hist(y, alpha = 0.3, lw = 3, label = "G3")
ax[0].set_ylabel("Frequency")
ax[0].legend()
ax[0].set_title("Histogram")
necdf = ECDF(y)
dataecdf = ECDF(g3s)
ax[1].plot(necdf.x, necdf.y, alpha = 0.5, lw = 3, label = "$\chi^2_{(2)}$")
ax[1].plot(dataecdf.x, dataecdf.y, alpha = 0.5, lw = 3, label = "G3")
ax[1].set_ylabel("cdf")
ax[1].legend(loc = "upper right")
ax[1].set_title("ECDF")


# %%
# 應該為越少越好
# H0: 此分布為常態
# Ha: 此分布不為常態
def JBtest(x):
    g2 = lambda x: np.sqrt(len(x)/ 24) * (kurtosis(x, fisher = False) - 3)
    g1 = lambda x: np.sqrt(len(x)/ 6) * skew(x)
    g3 = lambda x: g2(x) ** 2 + g1(x) ** 2
    stat = g3(x)
    p_value = 1 - chi2.cdf(stat, df = 2)
    return stat, p_value

#%%
def randomdist(n, dist, arg):
    if dist == "normal":
        return np.random.normal(size = n, loc = arg[0], scale = arg[1])
    if dist == "t":
        return np.random.standard_t(size = n, df = arg[0])
    if dist == "uniform":
        return np.random.uniform(size = n, low=arg[0], high= arg[1])
    if dist == "chisquare":
        return np.random.chisquare(size = n, df = arg[0])
#%%
def calpower(JBdata):
    crivalue = chi2.ppf(0.95, df = 2)
    rej = np.sum(JBdata > crivalue)
    return(rej/len(JBdata))
#%%
powerarr = np.empty(42)
i = 0
for j,k in zip(["normal", "t", "t", "t", "uniform", "chisquare"], [[0, 1], [3], [10], [30], [0, 1], [8]]):
    print(j)
    for n in [10, 20, 30, 50, 100, 300, 500]:
        distdata = randomdist(n * 50000, j,k).reshape(n, 50000)
        stat, _ = JBtest(distdata)
        powerarr[i] = calpower(stat)
        i = i + 1 
        print(calpower(stat))
#%%
powerdf = pd.DataFrame(powerarr.reshape(6, 7).T)
powerdf.columns = ["normal(0,1)", "t(3)", "t(10)", "t(30)", "uniform(0,1)", "chisquare(8)"]
powerdf["n"] = [10, 20, 30, 50, 100, 300, 500]
print(powerdf.set_index('n'))

# %%
def anderson_p(dist, args):
    n = [10, 20, 30, 50, 100, 300, 500]
    m = 500
    power = np.empty(len(n))
    for i in range(len(n)):
        count = 0
        for j in range(m):
            ns = randomdist(n[i], dist,  args)
            stat, cv, _= anderson(ns)
            if stat > cv[2]:
                count = count + 1 
        power[i] = count/m
    return(power)
def komolgorov_p(dist, args):
    n = [10, 20, 30, 50, 100, 300, 500]
    m = 500
    power = np.empty(len(n))
    for i in range(len(n)):
        count = 0
        for j in range(m):
            ns = randomdist(n[i], dist,  args)
            _, pv= kstest(ns, 'norm')
            if pv < 0.05:
                count = count + 1 
        power[i] = count/m
    return(power)
def normaltest(func, dist, args):
    n = [10, 20, 30, 50, 100, 300, 500]
    m = 500
    power = np.empty(len(n))
    for i in range(len(n)):
        count = 0
        if func == 'anderson':
            for j in range(m):
                ns = randomdist(n[i], dist,  args)
                stat, cv, _= anderson(ns)
                if stat > cv[2]:
                    count = count + 1 
        if func == 'kolmogorov':
            for j in range(m):
                ns = randomdist(n[i], dist,  args)
                _, pv= kstest(ns, 'norm')
                if pv < 0.05:
                    count = count + 1  
        if func == 'jarque_beta':
            for j in range(m):
                ns = randomdist(n[i], dist,  args)
                _, pv= jarque_bera(ns)
                if pv < 0.05:
                    count = count + 1   
            
        power[i] = count/m
    return(power)
# %%
plt.plot(powerdf.n.astype(str), normaltest('anderson', 'normal', [0, 1]), label = "JB", lw = 3)
plt.plot(powerdf.n.astype(str), normaltest('kolmogorov', 'normal', [0, 1]), label = "Kolmogorov", lw = 3)
plt.plot(powerdf.n.astype(str), normaltest('jarque_beta', 'normal', [0, 1]), label = "Anderson", lw = 3)
plt.legend(loc = "upper right")
plt.title("Normal (0, 1)")
plt.show()
# %%
plt.plot(powerdf.n.astype(str), powerdf['normal(0,1)'], label = "JB", lw = 3)
plt.plot(powerdf.n.astype(str), komolgorov_p("normal", [0,1]), label = "Kolmogorov", lw = 3)
plt.plot(powerdf.n.astype(str), anderson_p("normal", [0,1]), label = "Anderson", lw = 3)
plt.legend(loc = "upper right")
plt.title("Normal (0, 1)")
plt.show()

# %%
plt.plot(powerdf.n.astype(str), powerdf['t(3)'], label = "JB", lw = 3)
plt.plot(powerdf.n.astype(str), komolgorov_p("t", [3]), label = "Kolmogorov", lw = 3)
plt.plot(powerdf.n.astype(str), anderson_p("t", [3]), label = "Anderson", lw = 3)
plt.legend(loc = "upper right")
plt.title("T(3)")
plt.show()
#%%
powerdf