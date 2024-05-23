import scipy.optimize as optimize

fun = lambda x: (x[0] - 1)**2 + (x[1] - 2.5)**2
res = optimize.minimize(fun, (2, 0), method='TNC', tol=1e-10)
print(res.x)