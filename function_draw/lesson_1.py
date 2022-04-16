# %%
import numpy as np
X = np.array([[1, 2, 3], [4, 5, 6]])
A = np.array([[1, 1, 1], [2, 4, 6], [5, 7, 6]])
b = np.array([[1], [2], [3]])


# %% 1
a = X[0, 1]
print(a)
# %% 2
P = X * X
print(P)
# %% 3
Q = X/X
print(Q)
# %% 4
R = np.matmul(X, X.T)
RFAST = X @ X.T
print(R, RFAST)
# %% 5
ts = X.sum()
print(ts)
# %% 6
rs = X.sum(axis = 1)
cs = X.sum(axis = 0)
print(rs, cs)
# %% 7
Ainv = np.linalg.inv(A)
x = np.matmul(Ainv, b)
print(Ainv, A, x)
# %% 8
I = np.round(np.matmul(A, np.linalg.inv(A)))
print(I.astype(int))

# %% 9
A2 = A ** 2
print(A2)
# %% 10
Asqrt = np.sqrt(A)
print(Asqrt)

# %% 11
B = np.concatenate((A, b.T), axis = 0)
print(B)
# %% 12
C = np.concatenate((A, b), axis = 1)
print(C)

