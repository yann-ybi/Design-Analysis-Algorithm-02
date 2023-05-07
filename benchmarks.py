import time
from random import randint
from typing import List
from typing import Set

import matplotlib.pyplot as plt

from convex_hull import Point
from convex_hull import base_case_hull
from convex_hull import compute_hull


def generate_points(
        num_points: int,
        min_x: int = 0,
        max_x: int = 1_000,
        min_y: int = 0,
        max_y: int = 1_000,
) -> List[Point]:
    """ Creates a list of random and unique points for benchmarking the convex_hull algorithm.

    :param num_points: number of unique points to generate.
    :param min_x: minimum x-coordinate for points
    :param max_x: maximum x-coordinate for points
    :param min_y: minimum y-coordinate for points
    :param max_y: maximum y-coordinate for points
    """
    points: Set[Point] = set()
    while len(points) < num_points:
        points.add((randint(min_x, max_x), randint(min_y, max_y)))
    return list(points)


def run_benchmarks():
    # TODO: Generate points randomly, run your convex hull function,
    #  and record the time it takes on inputs of different sizes.
    # TODO: Plot a graph of runtime vs input size. What can you infer from the shape?

    sizes: List[int] = list(range(0, 1_000_000, 500))
    dnc_hull_times: List[float] = list()
    naive_hull_times: List[float] = list()
    for n in sizes:
        print(f'n: {n},', end=' ')

        points = generate_points(n)
        start_time = time.time()
        # TODO: call compute_hull here
        compute_hull(points)
        time_taken = time.time() - start_time  # time taken (in seconds) for divide-and-conquer

        print(f'dnc_time_taken: {time_taken:.3f},', end=' ')
        dnc_hull_times.append(time_taken)

        start_time = time.time()
        # TODO: call base_case_hull here
        # base_case_hull(points)
        time_taken = time.time() - start_time  # time taken (in seconds) for naive

        print(f'naive_time_taken: {time_taken:.3f}')
        naive_hull_times.append(time_taken)

    plt.scatter(sizes, dnc_hull_times, c='blue')
    plt.plot(sizes, dnc_hull_times, c='blue', label='add label here')
    plt.scatter(sizes, naive_hull_times, c='red')
    plt.plot(sizes, naive_hull_times, c='red', label='add label here')
    plt.legend()
    plt.xlabel('add x-label here')
    plt.ylabel('add y-label here')
    plt.title('Add title here')
    plt.savefig('benchmark_plot.png')
    return


if __name__ == '__main__':
    run_benchmarks()
