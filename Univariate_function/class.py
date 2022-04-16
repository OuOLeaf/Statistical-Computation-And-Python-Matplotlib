#%%
import numpy as np
import matplotlib.pyplot as plt 
import scipy.optimize as opt
import numpy.polynomial.polynomial as poly
from scipy.stats import beta
#%%
coef = [2, -3, 1] # from low to high
r = poly.polyroots(coef)
print('The roots are {}'.format(r))
# %% 取實數根出來
coef = [8, -2, 16, -8, 1]
r = poly.polyroots(coef)
r_x =np.real(r[np.isreal(r)])
#%%
x = np.linspace(-1, 5, 100)
y = poly.polyval(x, coef)

plt.style.use('ggplot')
plt.plot(x, y, color = 'g', linewidth = 2)
plt.scatter(r_x, [0, 0], s = 20, c = 'r')
plt.grid(True)
plt.title('Root finding of a polynomial function')
plt.show()
# %%
x = np.linspace(0, 1, 100000)
f1 = beta.pdf(x, a = 2, b = 6)
f2 = beta.pdf(x, a = 4, b = 2)

# %%
plt.plot(x, f1, x, f2)

#%%
f = lambda x: beta.pdf(x, a = 2, b = 6) - beta.pdf(x, a = 4, b = 2)
#%%
sol = opt.root_scalar(f, bracket=[0.1, 0.9], method='brentq')
# %%
print(sol.root)
#%%

# %%
