import numpy as np
import json

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


    # write to file
    with open("../examples/planetary_embryos.json", "w") as f:
        json.dump(config_dict, f, indent=4) 


def p9_setup_config():
    """
    Generates a JSON config file for TNOs as described in
    https://arxiv.org/pdf/2108.09868
    """
    num_bodies = 1024
    a_upper = 500
    a_lower = 150
    a_spacing = (a_upper - a_lower) / num_bodies
    i_spacing = np.radians(25) / num_bodies

    q_lower = 30
    q_upper = 50
    e_lower = - (q_lower / a_lower) + 1
    e_upper = - (q_upper / a_upper) + 1
    e_spacing = (e_upper - e_lower) / num_bodies

    config_dict = {
        "bodies": []
    }

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

    for i in range(num_bodies):
        config_dict["bodies"].append({
        "name": f"TNO-{i}",
        "mass": 0,
        "semi_major_axis": 150 + a_spacing * i,
        "eccentricity": i*e_spacing,
        "inclination": i_spacing * i,
        "longitude_of_ascending_node": np.random.uniform(0, 2*np.pi),
        "argument_of_perihelion": np.random.uniform(0, 2*np.pi),
        "mean_anomaly": np.random.uniform(0, 2*np.pi)
        })

    with open("../examples/p9.json", "w") as f:
        json.dump(config_dict, f, indent=4) 


p9_setup_config()