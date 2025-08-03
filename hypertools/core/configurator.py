# noinspection PyPackageRequirements
import datawrangler as dw
import os
from configparser import ConfigParser

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


def get_default_options(fname=None):
    """
    Parse a config.ini file

    Parameters
    ----------
    :param fname: absolute-path filename for the config.ini file (default: hypertools/hypertools/core/config.ini)
    Returns
    -------
    :return: A dictionary whose keys are function names and whose values are dictionaries of default arguments and
    keyword arguments
    """
    if fname is None:
        fname = os.path.join(os.path.dirname(__file__), 'config.ini')
        
    return RobustDict(dw.core.update_dict(dw.core.get_default_options(),
                      dw.core.get_default_options(fname)),
                      __default_value__={})
