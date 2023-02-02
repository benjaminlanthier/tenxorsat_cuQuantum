import os
import random
from cuQuantum_testing import main

# Define values for randomized tests
nb_tests = 106
path = "Data/3regularGraphs/"
directories = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
numbers = [int(d.replace("N", "")) for d in directories if d.startswith("N")]
numbers = sorted(numbers)

# Start the tests
if __name__ == "__name__":
    for i in range(nb_tests):
        print(f"- Test {i}:")
        N = random.choice(numbers)
        sample = random.randint(0, 99)
        print(f"  * N = {N} with sample{sample}")
        main(N, sample)
