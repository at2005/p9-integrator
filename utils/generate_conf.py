import numpy as np
import json

def generate_conf():
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
    with open("planetary_embryos.json", "w") as f:
        json.dump(config_dict, f, indent=4) 

generate_conf()