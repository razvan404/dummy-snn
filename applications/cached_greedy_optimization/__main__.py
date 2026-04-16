"""Entry point: python -m applications.cached_greedy_optimization"""

import logging

from applications.cached_greedy_optimization.optimize import main

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
main()
