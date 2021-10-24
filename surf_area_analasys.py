import math
import os
import random

import numpy
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt 
import random as r

import pandas
from IPython import display
import pandas as pd
import imageio
# Initialize plot
# @title Algae Blooms {run: "auto"}
# plt.style.use('seaborn-notebook')
from scipy import ndimage

plt.rcParams["figure.figsize"] = (10, 10)


def reset_plot():
    plt.cla()
    plt.grid(False)
    plt.axis('off')


# Parameters

# parameters
n = 100  # size of space n
p = 0.01  # Probability of initial cells on

# chance that clean becomes polluted per polluted neighbor
prob_clean_to_polluted = 0.01  # @param{type: "slider", min:0, max:1, step:0.001}
#  chance of "diffusion"
prob_polluted_to_clean = 0.002  # @param{type: "slider", min:0, max:1, step:0.001}
#  chance of bloom growth per neighboring pollution
prob_polluted_to_algae = .006  # @param{type: "slider", min:0, max:1, step:0.001}
#  chance of bloom growth per neighboring bloom
prob_clean_to_algae = .00005  # @param{type: "slider", min:0, max:1, step:0.001}
#  chance of algae bloom disappearing
prob_algae_to_clean = 0.0004  # @param{type: "slider", min:0, max:1, step:0.001}
# potential 'helper' methods / variables
states = dict({'clean': 0, 'polluted': 1, 'algae': 2})

# colors
colors = ["royalblue", "tan", "mediumseagreen"]
cmap = mpl.colors.LinearSegmentedColormap.from_list("", colors=colors)


def initialize(n, p):
    config = np.zeros([n, n])
    for x in range(n):  # for every row
        for y in range(n):  # for every column
            config[x, y] = 1 if r.random() < p else 0  # percolation
    return (config)


def initialize_northwest_pointsource_of_size_s(n, p, s):
    config = np.zeros([n, n])
    for x in range(n):  # for every row
        for y in range(n):  # for every column
            config[x, y] = 1 if r.random() < p else 0  # percolation
    # for top left corner:
    for x in range(s):  # for every row
        for y in range(s):  # for every column
            config[x, y] = 1 if r.random() < 0.75 else 0
    return config


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


def initialize_percolation_cluster(n, perc_prob):
    config = np.zeros([n, n])
    config = run_percolation(config, perc_prob)
    # filter largest cluster
    structure = [[0, 1, 0], [1, 1, 1], [0, 1, 0]]  # define connection
    label_world, nb_labels = ndimage.label(config, structure)  # label clusters
    sizes = ndimage.sum(config, label_world, range(nb_labels + 1))
    mask = sizes >= sizes.max()
    binary_img = mask[label_world] # binary img will give us our pollution cluster
    binary_img_int = np.zeros([n, n])
    for i in range(n):
        for j in range(n):
            if binary_img[i, j]:
                binary_img_int[i,j] = 1
            else: # polluted with probability p
                die = random.uniform(0, 1)
                if die < p:
                    binary_img_int[(i, j)] = 1
                else:
                    binary_img_int[(i, j)] = 0
    return binary_img_int, max(sizes)


def observe(config):
    reset_plot()
    plt.imshow(config, vmin=0, vmax=2, cmap=cmap)


def update(config, n):
    nextconfig = np.zeros([n, n])  # all-zero array w/ n rows and n columns
    for x in range(n):  # for every row
        for y in range(n):  # for every column
            counts = [0, 0, 0]  # counters for active cells
            for dx in [-1, 0, 1]:  # loop over 1 dimension of Moore neighborhood
                for dy in [-1, 0, 1]:  # loop over the other dimension
                    cell_state = int(config[(x + dx) % n, (y + dy) % n])
                    counts[cell_state] += 1

            [clean_count, polluted_count, algae_count] = counts

            # Current state
            current_state = config[x, y]

            # Clean
            if (current_state == states['clean']):
                pollution_prob = prob_clean_to_polluted * polluted_count
                algae_prob = prob_clean_to_algae * algae_count
                if r.uniform(0, 1) < pollution_prob:
                    nextconfig[x, y] = states['polluted']  # clean to polluted
                elif r.uniform(0, 1) < algae_prob:
                    nextconfig[x, y] = states['algae']  # clean to algae
                else:
                    nextconfig[x, y] = config[x, y]  # stay the same
                # OR if majority of neighbors are algae:
                #   (like if pollution is surrounded by algae, it should probably just be consumed by the surrounding algae? )
                # OR if all neighbors are pollution, should it become algae?

            # Polluted
            elif (current_state == states['polluted']):
                algae_prob = prob_polluted_to_algae * polluted_count
                if (r.uniform(0, 1) < prob_polluted_to_clean):
                    nextconfig[x, y] = states['clean']  # polluted to clean
                elif (r.uniform(0, 1) < algae_prob):
                    nextconfig[x, y] = states['algae']  # polluted to algae
                else:
                    nextconfig[x, y] = config[x, y]  # stay the same

            # Algae
            elif (current_state == states['algae']):
                algae_prob = prob_polluted_to_algae * algae_count
                if (r.uniform(0, 1) < prob_algae_to_clean):
                    nextconfig[x, y] = states['clean']  # algae to clean
                else:
                    nextconfig[x, y] = config[x, y]  # stay the same

    return (nextconfig)


def get_final_state_ratios():
    global sizes
    clean = 0
    polluted = 0
    algae = 0
    for x in range(n):
        for y in range(n):
            if world[x, y] == 0:
                clean += 1
            elif world[x, y] == 1:
                polluted += 1
            else:
                algae += 1
    return [clean / (n * n * 1.00), polluted / (n * n * 1.00), algae / (n * n * 1.00)]


def create_pie_chart(sizes):
    fig1, ax1 = plt.subplots()
    labels = ["Clean", "Polluted", "Algae"]
    explode = (0, 0, 0.1)
    ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
            shadow=True, startangle=90, colors=colors)
    plt.savefig(f"Plots/pie_chart_threshold")


def create_gif():
    global step
    files = os.listdir("Plots/gifMaker/")
    with imageio.get_writer('threshold.gif', mode='I') as writer:
        step = 0
        for filename in files:
            if step % 2 == 0:
                image = imageio.imread("Plots/gifMaker/" + filename)
                writer.append_data(image)
            step += 1
    # Remove files
    for filename in set(files):
        os.remove("Plots/gifMaker/" + filename)


epsilons = [1,2,3,4]
length = epsilons.co
# world = initialize(n, p)
# world = initialize_northwest_pointsource_of_size_s(n, p, round(math.sqrt(n*n*.20)))
perc_probs = np.linspace(0, .53, 53) # only go a little past perc threshold because of what we know about percolation
algae_sizes = []
cluster_sizes = []
dfItems = []

for perc_prob in perc_probs:
    world, cluster_size = initialize_percolation_cluster(n, perc_prob)
    fractional_cluster_size = cluster_size/(n*n)
    print(fractional_cluster_size)
    for step in range(500):
        world = update(world, n)
        observe(world)
        # plt.gcf()
        # plt.title(f"p1: {prob_clean_to_polluted}, p2: {prob_polluted_to_clean}, p3: {prob_polluted_to_algae}, p4: {prob_clean_to_algae}, p5: {prob_algae_to_clean}")
        plt.savefig(f"Plots/gifMaker/threshold{step:4d}")
        # display.display(plt.gcf())
        # display.clear_output(wait=True)

    sizes = get_final_state_ratios()
    algae_size = sizes[2]
    dfItems.append([fractional_cluster_size, algae_size])
    # create_pie_chart(sizes)
    # create_gif()

# now plot cluster size vs end state algae size
df = pandas.DataFrame(dfItems, columns=["cluster size", "algae size"])
df.plot(kind="scatter", x="cluster size", y="algae size")
plt.show()




