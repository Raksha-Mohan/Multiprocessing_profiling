import cProfile
import numpy as np
import time
cProfile.run('''
n = 10000
a = np.random.rand(n)
b = np.random.rand(n)
c = 0

# For-loop
start_loop = time.perf_counter()
for i in range(n):
    c += a[i] * b[i]
end_loop = time.perf_counter()
dt = end_loop- start_loop
print('loop time:', dt)

# Vectorization
start_vt = time.perf_counter()
cc = np.dot(a,b)
end_vt = time.perf_counter()
vt = end_vt-start_vt
print('Vectorization time:', vt)

# Comparing the results
print("Difference in the results:", np.linalg.norm(c - cc))

# Measure the speed-up
Sp = dt/vt

print ('Speedup:', Sp)
''')