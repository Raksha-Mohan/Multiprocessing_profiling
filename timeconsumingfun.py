from multiprocessing  import Pool
import multiprocessing
import time

# Define a more CPU-bound time-consuming function
def timeconsumingfun(x):
    time.sleep(5)
    return x
workers= 4
if __name__ == "__main__":
    # Serial computation
    n = int(input("n: "))  # Number of iterations
    start_time = time.perf_counter()
    for i in range(n):
        timeconsumingfun(i)
    ts = time.perf_counter() - start_time
    print("ts:", ts)

    # Parallel computation
    start_time = time.perf_counter()
    pool = multiprocessing.Pool(workers)  # Creating a pool of worker processes
    results = pool.map(timeconsumingfun, range(n))
    tp = time.perf_counter() - start_time
    print("tp:", tp)

    # Calculate speedup
    speedup = ts / tp
    print("Speedup:", speedup)
    efficiency = speedup / workers*100
    print('Efficiency:', efficiency)
