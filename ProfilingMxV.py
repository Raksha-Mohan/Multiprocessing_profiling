import cProfile
import numpy as np
import time
cProfile.run('''
import numpy as np
import time
n = 100
A = np.random.rand(n,n)
x = np.random.rand(n)
b = np.zeros(n)
bb = np.zeros(n)

start_loop = time.perf_counter()
for i in range(n):
    for j in range(n):
        b[i] += A[i, j] * x[j]
end_loop = time.perf_counter()
tl = end_loop - start_loop
print('loop time:', tl)

# Partial Vectorization
start_pvt = time.perf_counter()
for i in range(n):
    bb[i] = np.dot(A[i, :], x)
end_pvt = time.perf_counter()
pvt = end_pvt- start_pvt
print('Partial Vectorization time:', pvt)

#Vectorization
start_vt = time.perf_counter()
bbb = np.dot(A, x)
end_vt = time.perf_counter()
vt = end_vt- start_vt
print('Vectorization time:', vt)

# Comparing the results
print("Difference in the results (b and bb):", np.linalg.norm(b - bb))
print("Difference in the results (bb and bbb):", np.linalg.norm(bb - bbb))

#Speedup
print('Speedup', tl/pvt)
print('Speedup1', tl/vt)
print('Speedup2', pvt/vt)
''')