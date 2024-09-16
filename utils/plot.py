import matplotlib.pyplot as plt
import sys

def parse_file(file_str):
    buffer = []
    all_positions = []
    for line in file_str.splitlines():
        if line == '' or line[0] == "#" or line[0] == "\n":
            if buffer != []:
                all_positions.append(buffer)
                buffer = []
            continue
        # extract positions
        # <Body Name>: x y z -> [x, y, z]
        positions = line.split(": ")[1].split(" ")
        buffer.append([float(x) for x in positions])

    return all_positions

def plot_positions(positions_at_each_timestep):
    number_of_bodies = len(positions_at_each_timestep[0])
    trajectories = [[] for _ in range(number_of_bodies)]
    for snapshot in positions_at_each_timestep:
        for i, body in enumerate(snapshot):
            trajectories[i].append((body[0], body[1]))
    
    for trajectory in trajectories:
        x, y = zip(*trajectory)  
        plt.plot(x, y)  
    
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.title('Trajectories of Bodies Over Time')
    plt.show()


plot_positions(parse_file(open(sys.argv[1]).read()))