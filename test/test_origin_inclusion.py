import numpy as np
from operations.fitness import _extract_routes

def test_origin_inclusion():
    
    individual = np.array([5, 6, 4, 11, 3, 1, 2, 7, 10, 8, 9], dtype=int)

    breaks = [3, 8, 11]

    origin = 0

    routes = _extract_routes(individual, breaks, origin=origin)

    print("--"*50)
    print("Previous Routes: \n", routes)

    print("Routes with origin: \n")
    for rt_idx, rt in enumerate(routes):
        rt_cmp = "OK" if rt[0] == origin and rt[-1] == origin else "NOT OK"
        print(f"Route {rt_idx}: ", rt_cmp)
    print("--"*50)

if __name__ == "__main__":
    test_origin_inclusion()
