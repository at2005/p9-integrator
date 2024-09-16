import matplotlib.pyplot as plt
import numpy as np

def parse_file(file_str):
    buffer = []
    all_positions = []
    for line in file_str.splitlines():
        if line == '' or line[0] == "#" or line[0] == "\n":
            all_positions.append(buffer)
            buffer = []
            continue
        # extract positions
        # <Body Name>: x y z -> [x, y, z]
        positions = line.split(": ")[1].split(" ")
        buffer.append([float(x) for x in positions])

    return all_positions

def plot(positions_at_each_timestep):
    for snapshot in positions_at_each_timestep:
        for body in snapshot:
            plt.plot(body[0], body[1], body[2])
    plt.show()