"""Logger of the LADS OPC UA client library."""

from __future__ import annotations

import logging
import sys

# initialize logger (one logger for the whole package, so that all modules share its handler)
level = logging.WARNING
_logger = logging.getLogger(__package__)
_logger.setLevel(level)
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(level)
formatter = logging.Formatter(
    fmt='[%(asctime)s] [%(levelname)s] %(name)s:%(lineno)d in %(funcName)s() → %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S')
console_handler.setFormatter(formatter)
if not _logger.handlers:
    _logger.addHandler(console_handler)
