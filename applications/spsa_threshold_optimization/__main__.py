"""Entry point: python -m applications.spsa_threshold_optimization"""

import logging

from applications.spsa_threshold_optimization.optimize import main

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
main()
