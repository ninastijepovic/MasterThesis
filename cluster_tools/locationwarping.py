import numpy as np

warp_strength = 10 ** (-1.5)


def warp_station_locations(pqr_by_name):
    return {name: {'warped_location': warp_station_location(pqr_by_name[name])} for name in pqr_by_name}


def warp_station_location(pqr):
    p, q, r = pqr
    angle = np.arctan2(p, q)
    old_amplitude = np.sqrt(p ** 2 + q ** 2 + r ** 2)
    amplitude = np.log(warp_strength * old_amplitude + 1)
    return amplitude * np.sin(angle), -amplitude * np.cos(angle)
