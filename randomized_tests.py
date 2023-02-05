import os
import random
from XORSAT_TN_counting import main

# Define values for randomized tests
nb_tests_per_N = 1
path = "Data/3regularGraphs/"
directories = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
numbers = [int(d.replace("N", "")) for d in directories if d.startswith("N")]
numbers.sort(reverse = True) # There are 53 different tensor networks sizes, ranging from 16 tensors to 320 tensors

# Start the tests
if __name__ == "__name__":
    for N in numbers:
        print(f"\nTests for N = {N}: ")
        for i in range(nb_tests_per_N):
            sample = random.randint(0, 99)
            print(f"  * sample{sample}")
            main(N, sample)
