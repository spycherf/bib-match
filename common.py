import csv
import functools
import os
import time


def timer(func):
    @functools.wraps(func)
    def wrapper_timer(*args, **kwargs):
        start = time.perf_counter()
        value = func(*args, **kwargs)
        end = time.perf_counter()
        elapsed_time = end - start
        print(f"Elapsed time: {elapsed_time:0.4f} seconds")

        return value

    return wrapper_timer
