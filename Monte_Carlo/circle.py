#%%
n = 100
x = np.random.uniform(-1, 1, n)
y = np.random.uniform(-1, 1, n)
dis = np.sqrt(x**2 + y**2)
incircle = [1  if u <= 1 else 0 for u in dis]
expi = (4 * sum(incircle))/n 
print(expi)
#%%
theta = np.linspace(0, 2 * np.pi, n)
cx = np.sin(theta)
cy = np.cos(theta)

#%%
plt.figure(figsize = (4, 4))
plt.scatter(x, y, color = ["orange" if k == 1 else "blue" for k in incircle], alpha = 0.3)
plt.plot(cx, cy, lw = 2, color = "purple")
plt.axis([-1, 1, -1, 1])

#%%
def simultpi(n, repeat):
    pilist = np.empty(repeat)
    for i in range(repeat):
        x = np.random.uniform(-1, 1, n)
        y = np.random.uniform(-1, 1, n)
        dis = np.sqrt(x**2 + y**2)
        incircle = [1  if u <= 1 else 0 for u in dis]
        expi = (4 * sum(incircle))/n
        pilist[i] = expi
    return(pilist)
    
# %%
pd.DataFrame([np.mean(simultpi(1000, 100)), np.var(simultpi(1000, 100))])

