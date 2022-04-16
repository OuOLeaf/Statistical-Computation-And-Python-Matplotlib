#%%
plt.style.use("default")
#%%
import numpy as np
import matplotlib.pyplot as plt
 opts = dict(disp = 1, xtol = 1e-6, ftol = 1e-6, maxfun = 1e4, maxiter = 1e4, full_output = True)
xtol - x值最小容忍可使演算法停止 重要的參數
ftol - 函數值最小容忍可使演算法停止 重要的參數
maxfun - 函式計算最多次數
maxiter - 函式可重複的次數 但因有些iteration中會使用兩三次參數 因此 maxiter > maxfun

OptVal = opt.fmin(func = f, x0=[0, 0], **opts)
x0 - 起始值設定
 
f = lambda x : x[0] * np.exp(-x[0]**2 - x[1]**2)
 
x = np.linspace(-2, 2, 100)
y = np.linspace(-2, 3, 100)
X, Y = np.meshgrid(x, y) 
# 底下(xy平面)的網格大小
Z = f([X, Y])
 
fig = plt.figure(figsize=(9, 6))
ax = plt.axes(projection = '3d')
ax.plot_wireframe(X, Y, Z, color ='blue',
    alpha=0.3, rstride = 1, cstride = 1)
# cstride rstride 跳格畫圖

ax.set_xlabel('X'), ax.set_ylabel('Y')
ax.set_zlabel('f(X,Y)')
ax.view_init(10, -60)  #(elev=-165, azim=60)
plt.title('Wireframe (Mesh) Plot')
plt.show()

#%%
# for surface 3D plot
fig = plt.figure(figsize=(9, 6))
ax = plt.axes(projection = '3d')
surf = ax.plot_surface(X, Y, Z, color = 'r', \
    rstride=4, cstride=4, alpha =0.6, cmap='ocean') # cmap = plt.cm.bone
 # cmap = plt.cm.bone
fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10) # aspect = length/width ratio

ax.view_init(10, -60)  #(elev=-165, azim=60)
ax.set_xlabel('X'), ax.set_ylabel('Y')
plt.title('Surface Plot')
plt.show()
# %%
# To draw a contour plot
levels = np.arange(-0.4, 0.4, 0.01) # levels of contour lines
contours = plt.contour(X, Y, Z, levels=levels) # check dir(contours)
# add function value on each line    
plt.clabel(contours, inline = 0, fontsize = 10) # inline =1 or 0 
# inline 是否嵌在線內
cbar = plt.colorbar(contours)
plt.xlabel('X'), plt.ylabel('Y')
cbar.ax.set_ylabel('Z = f(X,Y)') # set colorbar label
# cbar.add_lines(contours) # add contour line levels to the colorbar 
plt.title('Contour Plot')
plt.grid(True)
plt.show()
#%%
# 顏色表達高度
# draw a contour plot with contourf
C1 = plt.contourf(X, Y, Z, 30, \
    cmap = plt.cm.bone)
C2 = plt.contour(C1, levels = C1.levels, \
    colors = 'r') # check dir(contours)
plt.colorbar(C1)
plt.xlabel('X'), plt.ylabel('Y')
plt.title('contourf + contour')  
plt.show()
# %%
ax = plt.axes(projection='3d')
ax.plot_surface(X, Y, Z, rstride=1, cstride=1,
                cmap='viridis', antialiased=False)
# antialiased more smooth
ax.set_title('surface')
# %%
f = lambda x : (x[0] - 2) ** 4 + (x[0] - 2)**2 * x[1]**2 + (x[1] + 1) ** 2
x = np.linspace(-2, 5, 100)
y = np.linspace(-5, 5, 100)
X, Y = np.meshgrid(x, y) 
# 底下(xy平面)的網格大小
Z = f([X, Y])
fig = plt.figure(figsize=(9, 6))
ax = plt.axes(projection = '3d')
ax.plot_surface(X, Y, Z, rstride=1, cstride=1,
                cmap='viridis', antialiased=True)

#%%
contours = plt.contour(X, Y, Z, 300) # check dir(contours)
plt.clabel(contours, inline = 0, fontsize = 10) # inline =1 or 0 
# 300 條等高線
#%%
contours.labels

# %%
import matplotlib.pyplot as plt  
from mpl_toolkits.mplot3d.axes3d import Axes3D, get_test_data  
from matplotlib import cm   
import numpy as np   
import random  
X_k_list = range(1, 100, 10)  
Y_p_list = [ float(x)/100.0 for x in range(1, 100, 10) ]   
# set up a figure twice as wide as it is tall  
fig = plt.figure(figsize=plt.figaspect(0.5))  
# set up the axes for the first plot  
ax = fig.add_subplot(1, 1, 1, projection='3d')  
# plot a 3D surface like in the example mplot3d/surface3d_demo  
X, Y = np.meshgrid(X_k_list, Y_p_list)  
def critical_function(b, c):  
    num = random.uniform(0, 1) * 10.0  
    return num + (b * c)   
  
Z_accuracy = X.copy()  
Z_accuracy = Z_accuracy.astype(np.float32)  
for i in range(len(X_k_list)):  
    for j in range(len(Y_p_list)):  
        Z_accuracy[j][i] = critical_function(Y_p_list[j], X_k_list[i])  
  
surf = ax.plot_surface(X, Y, Z_accuracy,   
    rstride=1, cstride=1, cmap=cm.coolwarm,  
    linewidth=0, antialiased=False)  
  
ax.set_xlabel('X')  
ax.set_ylabel('Y')  
ax.set_zlabel('Z')  
fig.colorbar(surf, shrink=0.5, aspect=10)  
plt.show()
# %%
# %%
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()
ax = Axes3D(fig)
X = np.arange(-4, 4, 0.25)
Y = np.arange(-4, 4, 0.25)
X, Y = np.meshgrid(X, Y)
R = np.sqrt(X ** 2 + Y ** 2)
Z = np.sin(R)

ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=plt.cm.hot)
ax.contourf(X, Y, Z, zdir='z', offset=-2, cmap=plt.cm.hot)
# zdir 繪製二維影象時的z軸方向
ax.set_zlim(-2, 2)

plt.show()
# %%
