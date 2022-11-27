import numpy as np
from cec17_functions import cec17_test_func

for i in range(1, 31):
    solution = np.random.uniform(-100, 100, 10)
    f = [0]
    cec17_test_func(solution, f, 10, 1, i)
    print(f[0])