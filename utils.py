import logging
import os
import time
import datetime as dt
import pandas as pd
import numpy as np
import re
import resource
from functools import wraps, partial
from configobj import ConfigObj

logger = logging.getLogger("utils.py")

def logged(fn = None, logger = None):
    if fn is None:
        return partial(logged, logger=logger)
    if logger is None:
        logger = fn.__globals__.get('logger', logging.getLogger(fn.__name__))

    @wraps(fn)
    def wrapped(*args, **kwargs):
        start_time = dt.datetime.now()
        resource_usage = resource.getrusage(resource.RUSAGE_SELF)
        logger.info("starting %s, memory: %s", fn.__name__, "{:,}".format(resource_usage.ru_maxrss))
        retval = fn(*args, **kwargs)
        end_time = dt.datetime.now()
        resource_usage = resource.getrusage(resource.RUSAGE_SELF)
        logger.info("finished %s, runtime: %s, memory: %s", fn.__name__,
                    "{:,}".format((end_time - start_time).total_seconds()),
                    "{:,}".format(resource_usage.ru_maxrss))
        return retval
    return wrapped

def skip_if_exception(exception, message, retval=None):
    def decorator(fn):
        @wraps(fn)
        def wrapped(*args, **kwargs):
            try:
                fn(*args, **kwargs)
            except exception:
                logger.info(message)
                return retval
        return wrapped
    return decorator

def is_excel_file(name):
    """
    check if file extension is '.xlsx'
    >>> is_excel_file('path/to/file.xlsx')
    True
    >>> is_excel_file('path.to.file.xlsx')
    True
    >>> is_excel_file('path/to/file.csv')
    False
    >>> is_excel_file('path/to/file.xls')
    False
    """
    return os.path.splitext(name)[-1] == '.xlsx'

GLOBAL_CONFIG = None
# GLOBAL_CONFIG = {'database': {'uri': 'sqlite:///test/db',
#                               'text_factory_str': True,
#                               'engine': {'pool_recycle': 7200, 'echo': False},
#                               'pragmas': {'foreign_keys': 'ON'}
#                               },
#                  'parameters': {'overrides': {'reset_lp_flag' = True},
#                                 'defaults': {'fixed_cost_sos_flag': True, 'lp_file_extension' = 'mps'}
#                                 }
#                   }

def get_config():
    global GLOBAL_CONFIG
    if GLOBAL_CONFIG is None:
        config_file_name = 'config.ini'
        if not os.path.exists(config_file_name):
            config_file_name = "default_config.ini"
        logger.info("reading configuration from %s", config_file_name)
        GLOBAL_CONFIG = ConfigObj(config_file_name, stringify=True, unrepr=True)
    config = GLOBAL_CONFIG
    if 'input_directory' not in config or not os.path.isdir(config['input_directory']):
        config['input_directory'] = "."
    return config

def get_config_value(keys, default, config = None):
    value = config or get_config()
    for key in keys:
        try:
            value = value[key]
        except KeyError:
            return default
        except:
            logger.warn("invalid config file value looking for value: %s using default %s", keys, default)
            return default
    return value

class lazy_property(object):
    """
    used for lazy evaluation of an object attribute.
    property should represent non-mutable data, as it replaces itself.
    """
    def __init__(self, fget):
        self.fget = fget
        self.func_name = fget.__name__

    def __get__(self, obj, cls):
        if obj is None:
            return None
        value = self.fget(obj)
        setattr(obj, self.func_name, value)
        return value

def uncamel_case(name):
    """
    >>> uncamel_case(12)
    '12'
    >>> uncamel_case('CamelCamelCase')
    'camel_camel_case'
    >>> uncamel_case('Camel Camel Case')
    'camel_camel_case'
    >>> uncamel_case('Camel Camel 2case')
    'camel_camel_2case'
    >>> uncamel_case("Dave  Freight")
    'dave_freight'
    >>> uncamel_case("DaveFreight")
    'dave_freight'
    >>> uncamel_case("Dave freight")
    'dave_freight'
    """
    s1 = re.sub('\s+', r'_', str(name))
    s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', s1)
    s1 = re.sub('_+', '_', s1)
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()

def as_repr(*values, **kwargs):
    """
    >>> as_repr(1, 2, sep=".")
    '1.2'
    >>> as_repr(1, 2)
    '1|2'
    """
    if 'sep' in kwargs:
        sep = kwargs['sep']
    else:
        sep ='|'
    return sep.join(str(v) for v in values)

def pluralize(name):
    if name.endswith('y'):
        return name[:-1] + 'ies'
    return name + 's'

def get_parameter_value(name, value):
    inferred_type = get_inferred_type(name)
    if inferred_type == bool:
        return get_truthiness(value)
    return inferred_type(value)

def get_inferred_type(name):
    if name[-5:] == "_flag":
        return bool
    elif name [-7:] == "_weight":
        return float
    elif name[:3] == "nb_":
        return int
    elif name[-4:] in ['_tol', '_eps']:
        return float
    elif name.endswith("factor"):
        return float
    elif name.endswith("size"):
        return int
    return str

def get_truthiness(x):
    if isinstance(x, basestring):
        if x.lower() == 'true':
            return True
        elif x.lower() == 'false':
            return False
    try:
        return bool(int(x))
    except ValueError:
        return bool(float(x))
