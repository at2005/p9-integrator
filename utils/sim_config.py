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


def tno_setup_config():
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

    with open("../examples/tno_system.json", "w") as f:
        json.dump(config_dict, f, indent=4) 