import numpy as np
from random import random
 
 
def get_constant_ts(args: dict, length: int) -> np.array:
    if "value" not in args:
        raise ValueError(f"Missing value in {args}")
    return np.array([args["value"]] * length)
 
 
def get_uniform_ts(args: dict, length: int) -> np.array:
    if "base" not in args:
        raise ValueError(f"Missing base in {args}")
    if "low" not in args:
        raise ValueError(f"Missing low in {args}")
    if "high" not in args:
        raise ValueError(f"Missing high in {args}")
    if args["low"] > args["high"]:
        raise ValueError(f"Low {args['low']} is greater than high {args['high']}")
 
    if isinstance(args["base"], list):
        return np.array(
            [
                round(
                    _randomize(
                        x=args["base"][min(i, len(args["base"]) - 1)],
                        low=args["low"],
                        high=args["high"],
                    )
                )
                for i in range(length)
            ]
        )
    else:
        return np.array(
            [
                round(_randomize(x=args["base"], low=args["low"], high=args["high"]))
                for _ in range(length)
            ]
        )
 
 
def get_gaussian_ts(args: dict, length: int) -> np.array:
    if "base" not in args:
        raise ValueError(f"Missing base in {args}")
    if "mean" not in args:
        raise ValueError(f"Missing mean in {args}")
    if "std" not in args:
        raise ValueError(f"Missing std in {args}")
    return np.array(
        [
            args["base"] + np.random.normal(args["mean"], args["std"])
            for _ in range(length)
        ]
    )
 
 
def get_sinusoidal_ts(args: dict, length: int) -> np.array:
    if "base" not in args:
        raise ValueError(f"Missing base in {args}")
    if "amplitude" not in args:
        raise ValueError(f"Missing amplitude in {args}")
    if "period" not in args:
        raise ValueError(f"Missing period in {args}")
    if "std" not in args:
        raise ValueError(f"Missing std in {args}")
    return np.array(
        [
            args["base"]
            + args["amplitude"] * np.sin(2 * np.pi * t / args["period"])
            + np.random.normal(0, args["std"])
            for t in range(length)
        ]
    )
 
 
def _randomize(x, low, high):
    return x * (random() * (high - low) + low)
 
 
TIME_SERIES = {
    "constant": get_constant_ts,
    "uniform": get_uniform_ts,
    "gaussian": get_gaussian_ts,
    "sinusoidal": get_sinusoidal_ts,
}
 
 
def get_time_series(mode: str, args: dict, length: int) -> np.ndarray:
    if mode not in TIME_SERIES:
        raise ValueError(f"Time series {mode} not in {TIME_SERIES.keys()}")
    return TIME_SERIES[mode](args, length)