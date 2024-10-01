import numpy as np
import json
import random

def generate_range(mean, delta_upper, delta_lower):
    return [mean - delta_lower, mean + delta_upper]

def generate_sweep_array(bounds, search_space, schedule="linear"):
    """
    Generate an array of values between lower and upper with linear spacing
    """
    lower, upper = bounds
    # use linear schedule for now
    if schedule == "linear":
        spacing = (upper - lower) / search_space
        lst = [lower + i*spacing for i in range(search_space)]
        random.shuffle(lst)
        return lst
    return []

def combinatorial_sweep(N, element_sweep_array):
    """
    Generates N arrays of orbital elements, each with a different combination of values
    Inputs: 
    N, the number of arrays to generate
    element_sweep_array, a matrix of values to sweep over: [[lower, upper] for each element]
    """
    element_arrays = [generate_sweep_array(element_sweep_array[i], N, "linear") for i in range(len(element_sweep_array))]
    return [[element_arrays[i][j] for i in range(len(element_arrays))] for j in range(N)]
    

def planetesimal_setup_config():
    """
    Generates a JSON config file for planetary embryos
    """
    a_spacing = (1.2 - 0.5) / 30
    e_spacing = 0.01 / 30
    # 0.2 earth masses
    m_upper = 0.2 * (1/ 332946)
    # 0.6 lunar masses
    m_lower = 0.6 * (1 / 27068510)
    m_spacing = (m_upper - m_lower) / 30

    config_dict = {
        "bodies": []
    }

    for i in range(30):
        config_dict["bodies"].append({
            "name": f"Small-Body-{i}",
            "mass": m_lower + m_spacing * i,
            "semi_major_axis": 0.5 + a_spacing * i,
            "eccentricity": i*e_spacing,
            "inclination": 0,
            "longitude_of_ascending_node": np.random.uniform(0, 2*np.pi),
            "argument_of_perihelion": np.random.uniform(0, 2*np.pi),
            "mean_anomaly": np.random.uniform(0, 2*np.pi)
        })

    config_dict["bodies"].sort(key=lambda x: x["mass"], reverse=True)

    # write to file
    with open("../examples/planetary_embryos.json", "w") as f:
        json.dump(config_dict, f, indent=4) 


def p9_setup_config():
    """
    Generates a JSON config file for TNOs as described in
    https://arxiv.org/pdf/2108.09868
    """
    num_tnos = 640 * 16 - 5
    a_upper = 500
    a_lower = 150
    a_spacing = (a_upper - a_lower) / num_tnos
    i_spacing = np.radians(25) / num_tnos

    q_lower = 30
    q_upper = 50
    e_lower = - (q_lower / a_lower) + 1
    e_upper = - (q_upper / a_upper) + 1
    e_spacing = (e_upper - e_lower) / num_tnos

    config_dict = {
        "sweeps": [],
        "bodies": []
    }

    p9_mass = generate_range(6.9 * 3.00338e-06, 2.6 * 3.00338e-06, 1.6 * 3.00338e-06)
    p9_a = generate_range(460.7, 178.8, 103.3)
    p9_i = generate_range(np.radians(15.6), np.radians(5.2), np.radians(5.4))
    p9_e = generate_range(0.3, 0.1, 0.1)
    p9_longitude_of_perihelion = generate_range(np.radians(246.7), np.radians(15.1), np.radians(13.4))
    p9_longitude_ascending_node = generate_range(np.radians(96.9), np.radians(17.3), np.radians(15.5))
    
    p9_sweep = combinatorial_sweep(8, 
        [p9_mass, p9_a, p9_i, p9_e, p9_longitude_of_perihelion, p9_longitude_ascending_node]
    )
    
    for i in range(len(p9_sweep)):
        # convert to argument of perihelion
        p9_sweep[i][-2] -= p9_sweep[i][-1]
        config_dict["sweeps"].append({
            "name": f"Planet Nine",
            "mass": p9_sweep[i][0],
            "semi_major_axis": p9_sweep[i][1],
            "inclination": p9_sweep[i][2],
            "eccentricity": p9_sweep[i][3],
            "argument_of_perihelion": p9_sweep[i][4],
            "longitude_of_ascending_node": p9_sweep[i][5],
            "mean_anomaly": 0
        })

    # adding Planet Nine under hypothetical elements
    config_dict["bodies"].append({
        "name": "Planet Nine",
        # five earth masses
        "mass": 5 * 3.00338e-06,
        "semi_major_axis": 500,
        "eccentricity":  0.25,
        "inclination": np.radians(20),
        "longitude_of_ascending_node": np.radians(86),
        "argument_of_perihelion": np.radians(138),
        "mean_anomaly": 2*np.pi + np.radians(-176.2)
    })

    # adding 4 gas giants
    config_dict["bodies"].append({
        "name": "Neptune",
        "mass": 5.14855965e-5,
        "semi_major_axis": 30.06896348,
        "eccentricity" :0.00858587,
        "inclination" : np.radians(1.76917),
        "longitude_of_ascending_node" : np.radians(131.72169),
        "argument_of_perihelion" : 2*np.pi + np.radians(-86.75034),
        "mean_anomaly" : 0.00
    })


    config_dict["bodies"].append({
        "name": "Uranus",
        "mass": 4.36430044e-5,
        "semi_major_axis": 19.19126393,
        "eccentricity" : 0.04716771,
        "inclination" : np.radians(0.76986),
        "longitude_of_ascending_node" : np.radians(74.22988),
        "argument_of_perihelion" : 2* np.pi + np.radians(-96.73436),
        "mean_anomaly" : 0.00
    })

    config_dict["bodies"].append({
        "name" : "Saturn",
        "mass": 0.000285716656,
        "semi_major_axis": 9.53707032,
        "eccentricity": 0.05415060,
        "inclination": np.radians(2.48446),
        "longitude_of_ascending_node": np.radians(113.71504),
        "argument_of_perihelion": 2*np.pi + np.radians(-21.2831),
        "mean_anomaly": 0.00
    })

    config_dict["bodies"].append({
        "name" : "Jupiter",
        "mass": 0.000954588,
        "semi_major_axis": 5.20336301,
        "eccentricity": 0.04839266,
        "inclination": np.radians(1.30530),
        "longitude_of_ascending_node": np.radians(100.55615),
        "argument_of_perihelion": 2* np.pi + np.radians(-85.8023),
        "mean_anomaly": 0.00
    })

    for i in range(num_tnos):
        config_dict["bodies"].append({
        "name": f"TNO-{i}",
        "mass": 0,
        "semi_major_axis": 150 + a_spacing * i,
        "eccentricity": e_lower + i*e_spacing,
        "inclination": i_spacing * i,
        "longitude_of_ascending_node": np.random.uniform(0, 2*np.pi),
        "argument_of_perihelion": np.random.uniform(0, 2*np.pi),
        "mean_anomaly": np.random.uniform(0, 2*np.pi)
        })
    
    # ensure that p9 is always the first body, assumption we make for kernel lol
    config_dict["bodies"] = [config_dict["bodies"][0]] + sorted(config_dict["bodies"][1:], key=lambda x: x["mass"], reverse=True)

    with open("../examples/p9.json", "w") as f:
        json.dump(config_dict, f, indent=4) 


p9_setup_config()