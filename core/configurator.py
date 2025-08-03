# noinspection PyPackageRequirements
import configparser
import os

try:
    from importlib.metadata import version
except ImportError:
    # Fallback for Python < 3.8
    from pkg_resources import get_distribution
    def version(package_name):
        return get_distribution(package_name).version

from .shared import RobustDict


try:
    __version__ = version('hypertools')
except Exception:
    # Fallback if package not installed
    __version__ = '0.8.0-dev' 