import matplotlib.pyplot as plt
import sys
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import json

def parse_file(file_str):
    buffer = []
    experiment_buffer = []
    all_positions = []
    for line in file_str.splitlines():
        if line == '':
            continue
        if line[0] == "#":
            if experiment_buffer != []:
                buffer.append(experiment_buffer)
                experiment_buffer = []
            if buffer != []:
                all_positions.append(buffer)
                buffer = []
            continue

        if line.split(" ")[0] == "Experiment": 
            if experiment_buffer != []:
                buffer.append(experiment_buffer)
                experiment_buffer = []
            continue
        # extract positions
        # <Body Name>: x y z -> [x, y, z]
        positions = line.split(": ")[1].split(" ")
        experiment_buffer.append([float(x) for x in positions])

    return all_positions

def plot_positions(positions_at_each_timestep):
    number_of_bodies = len(positions_at_each_timestep[0])
    trajectories = [[] for _ in range(number_of_bodies)]
    for snapshot in positions_at_each_timestep:
        for i, body in enumerate(snapshot):
            trajectories[i].append((body[0], body[1], body[2]))

    # also write all points to a json file
    positions_array = [] 
    for trajectory in trajectories:
        x, y, z = zip(*trajectory)  
        positions_array.append(
            {
                "x": x,
                "y": y,
                "z": z
            }
        )
        plt.plot(x, y, linewidth=0.5)

    # write to file
    with open("positions.json", "w") as f:
        json.dump(positions_array, f, indent=4)
    
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.title('Trajectories of Bodies Over Time')
    plt.show()


def plot_positions_3d(positions_at_each_timestep):
    number_of_bodies = len(positions_at_each_timestep[0])
    trajectories = [[] for _ in range(number_of_bodies)]
    for snapshot in positions_at_each_timestep:
        for i, body in enumerate(snapshot):
            trajectories[i].append((body[0], body[1], body[2]))
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    for trajectory in trajectories:
        x, y, z = zip(*trajectory)
        ax.plot(x, y, z, linewidth=0.5)
    
    all_coords = np.array([body for snapshot in positions_at_each_timestep for body in snapshot])
    max_range = np.ptp(all_coords, axis=0).max() / 2.0
    mid_x, mid_y, mid_z = all_coords.mean(axis=0)
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Y Coordinate')
    ax.set_zlabel('Z Coordinate')
    ax.set_title('Trajectories of Bodies Over Time')
    
    plt.show()


parsed = np.array(parse_file(open(sys.argv[1]).read()))
for i in range(parsed.shape[1]):
    indexed_array = parsed[:, i, :, :]
    plot_positions_3d(indexed_array)