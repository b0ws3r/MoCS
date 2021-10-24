import numpy as np
import random
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import cm

from IPython import display
from time import sleep

def run_random_walker(world, position, history, steps, show_past=True, speed_up=True):
    N1 = world.shape[0]
    N2 = world.shape[1]
    (i, j) = position
    for k in range(steps):  # for every step
        adj_to_stuck = False
        # for dx,dy in zip([-1, 0, 1, 0], [0, 1, 0, -1]): # von neuman
        for dx in [-1, 0, 1]:  # Moore
            for dy in [-1, 0, 1]:
                if (world[(i + dx) % N1, (j + dy) % N2] == 2):
                    adj_to_stuck = True;
                    break

        if (adj_to_stuck):
            world[i, j] = 2  # stuck
        else:
            # leave last cell
            if (show_past):
                world[i, j] = 0.67
            else:
                world[i, j] = 0
            (i, j) = position

            right_prob = .25
            left_prob = .25
            down_prob = .25

            # speed up using "gravity" bias
            if (speed_up):
                right_prob = (N2 - j) / (N2 * 2)
                left_prob = j / (N2 * 2)
                down_prob = i / (N1 * 2)

            die = random.uniform(0, 1)
            if die < right_prob:  # right step
                position = (i, (j + 1) % N2)
            elif die < right_prob + down_prob:  # bottom step
                position = ((i - 1) % N1, j)
            elif die < right_prob + down_prob + left_prob:  # left step
                position = (i, (j - 1) % N2)
            else:  # top step
                position = ((i + 1) % N1, j)

            if (world[position] == 1 or world[position] == 2):
                world[i, j] = 1
                continue
            world[position] = 1  # continue

        history.append(position)

    return world, position, history

# Parameters
N1, N2 = 512, 512 #height, width
steps, dt = 50000, 50000 #steps to walk, plot at each dt
world = np.zeros((N1,N2)) #initial conditions of zero
world[N1//2,N2//2] = 1 #initial conditions
position = (N1//2,N2//2)
history = []

# Plot everything slowly, one generation at a time
fig, ax = plt.subplots(figsize=(10, 10))
ax.set_ylim(0,N1)
ax.set_xlim(0,N2)
plt.axis('off')
ax.set_aspect('equal')
for step in range(steps//dt):
    (world,position,history) = run_random_walker(world,position,history,dt) #run model
    plt.imshow((world), cmap=plt.get_cmap(cm.bone), origin = 'lower')
    display.display(plt.gcf())
    display.clear_output(wait=True)
    sleep(0.025) #plot slower, useful if not using colab.

# Add label
ax.text(2, N1+5, f'Unbiased random walk of length {dt*(steps//dt)}', color='Orange', fontsize=20)
plt.show()


# Site percolation
def run_percolation(world, probability):
    N1 = world.shape[0]
    N2 = world.shape[1]
    for i in range(N1):  # for cell in every row
        for j in range(N2):  # and every column
            die = random.uniform(0, 1)
            if die < probability:
                world[(i, j)] = 1
            else:
                world[(i, j)] = 0

    return (world)


# Parameters
N1, N2 = 100, 100  # height, width
N1, N2 = 128, 128  # height, width for BCD
# occupation probability
probability = 0.11  # @param {type:"slider", min:0, max:1, step:0.01}
# initial conditions of zero
world = np.zeros((N1, N2), dtype=np.uint8)

world = run_percolation(world, probability)  # Set up a bunch of random walkers.
world[N1 // 2, N2 // 2] = 2  # the middle one is fixed

# colors
colors = ["royalblue", "tan", "yellow"]
cmap = mpl.colors.LinearSegmentedColormap.from_list("", colors=colors)
plt.figure(figsize=(10, 10))
num_steps = 10000
had_one = True
for k in range(num_steps):
    if (not had_one): continue
    # for all cells set to 1 except the middle 1, run random walker (do not keep history)
    for i in range(N1):  # for cell in every row
        for j in range(N2):  # and every column
            if world[i, j] == 1:
                (world, _, _) = run_random_walker(world=world, position=[i, j], history=[], steps=1, show_past=False,
                                                  speed_up=True)

    if (k % 50 == 0 and had_one):
        plt.grid(False)
        plt.imshow((world), cmap=cmap, vmin=0, vmax=2, origin='lower')
        display.display(plt.gcf())
        display.clear_output(wait=True)
        had_one = 1 in world

# trivial length (1), to 1/N1 (size of matrix)


# Obviously we need 1 box of size 1 to cover our structure
#    show this by dividing our matrix into 1 boxes
#    check each box (for our first case, there's 1...) to see if it contains a stuck guy
#    N_epsilon = number of matrices containing a stuck guy, epsilon = 1
def split(array, nrows, ncols):
    """Split a matrix into sub-matrices."""

    r, h = array.shape
    return (array.reshape(h//nrows, nrows, -1, ncols)
                 .swapaxes(1, 2)
                 .reshape(-1, nrows, ncols))


epsilons = [f for f in range(2, N1) if N1 % f == 0]
n_epsilons = []
for epsilon in epsilons:
    N1 * epsilon, N1 * epsilon
    # num_submatrices = N1* epsilon # if n1 = 512, 256
    submatrices = split(world, int(N1/epsilon), int(N1/epsilon))
    n_epsilon = 0
    for matrix in submatrices:
        if 2 in matrix:
            n_epsilon += 1
    n_epsilons.append(n_epsilon)

for idx, n in enumerate(n_epsilons):
    bcd = np.log(n_epsilon) / np.log(epsilons[idx])
    print(f"size {1 / epsilons[idx]:5f}\tboxes {n:3d}\tbcd: {bcd:.3f}")

# Plot epsilons vs n_epsilons on log-log scale
epsilons = [1 / e for e in epsilons]
plt.style.use("seaborn")
plt.xlabel("Box size")
plt.ylabel("N Boxes")

plt.title("Box Counting")
# fairly certain that the slope of the line on the log/log scale is the box-counting dimension, since D ~= log(n_epsilon)/log(1/epsilon)
slope, intercept = np.polyfit(np.log(epsilons), np.log(n_epsilons), 1)
print(f"\ndimensions={slope}")

plt.annotate(f"slope={slope:.3f}", xy=(epsilons[3] + .2, n_epsilons[3] - 1))
plt.loglog(epsilons, n_epsilons, lw=3)
plt.loglog([epsilons[0], epsilons[-1]], [n_epsilons[0], n_epsilons[-1]], '--', color='g', lw=1)
ax = plt.gca()
ax.set_xticks(epsilons)
ax.set_yticks(n_epsilons)
ax.get_xaxis().set_major_formatter(mpl.ticker.ScalarFormatter())
ax.get_yaxis().set_major_formatter(mpl.ticker.ScalarFormatter())

# d = np.log(n_epsilon)/np.log(epsilons)
# print(d)