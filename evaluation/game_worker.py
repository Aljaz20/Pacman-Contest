"""Course launcher with independent map-generation and game RNG seeds."""
import argparse
import random
import sys
import time
from contest import capture


def execute(argv):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--setRandomSeed')
    seed, _ = parser.parse_known_args(argv)
    started = time.monotonic()
    options = capture.read_command(argv)
    # read_command builds RANDOM layouts, whose generator reseeds global random.
    # Reapply the game seed after all setup so constructor imports and map generation cannot override it.
    if seed.setRandomSeed is not None:
        random.seed(seed.setRandomSeed)
    games = capture.run_games(**options)
    total = round(time.monotonic() - started, 3)
    if games:
        capture.save_score(games=games, total_time=total, **options)
    print('Total game time:', total)


if __name__ == '__main__':
    execute(sys.argv[1:])
