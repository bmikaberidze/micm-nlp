import datetime
import importlib
import inspect
import json
import os
import pickle
import sys
import time
from collections.abc import ItemsView, KeysView, ValuesView
from types import SimpleNamespace
from uuid import UUID

import numpy as np
import yaml
from rich import print as rprint
from tqdm import tqdm


# Class resolution ---------------------------------------------------------------------------------------------------------------------
def resolve_cls(cls_name, modules, yaml_path=None):
    """Resolve a bare class name by importing it from one of `modules` (str or
    list, tried in order). Used to load HF / custom classes named in YAML
    without maintaining a registry. Raises ValueError on missing/unknown name.
    """
    label = yaml_path or 'cls'
    if not cls_name:
        raise ValueError(f'{label} is required.')
    mods = [modules] if isinstance(modules, str) else list(modules)
    tried = []
    for mod_name in mods:
        try:
            mod = importlib.import_module(mod_name)
        except ImportError:
            continue
        cls = getattr(mod, cls_name, None)
        if cls is not None:
            return cls
        tried.append(mod_name)
    raise ValueError(f'Unknown {label}={cls_name!r}. Not found in modules: {tried}.')


# Module info / debug ------------------------------------------------------------------------------------------------------------------
def info(file=__file__, name=__name__, package=__package__):
    print(
        'Module info:',
        json_dumps(
            {
                'file      ': file,
                'module    ': name,
                'package   ': package,
            }
        ),
    )


def print_traceback(show_locals=False, width=120, extra_lines=1):
    from micm_nlp.setup import init_rich

    init_rich({'show_locals': show_locals, 'width': width, 'extra_lines': extra_lines})
    raise Exception('Ephemeral Exception')


# Timing -------------------------------------------------------------------------------------------------------------------------------
def tik(tok, key, callback, params=()):
    start = time.perf_counter()
    res = callback(*params)
    end = time.perf_counter()
    tok[key] = format_seconds(end - start)
    return res


def format_seconds(n):
    return str(datetime.timedelta(seconds=n))


def get_time_id():
    return datetime.datetime.now().strftime('%Y%m%d_%H%M%S')


# JSON serialization -------------------------------------------------------------------------------------------------------------------
def json_dumps(object, **kwargs):
    return json.dumps(object, sort_keys=False, indent=4, ensure_ascii=False, **kwargs)


def json_dumps_numpy(object, **kwargs):
    return json_dumps(object, cls=NumpyEncoder, **kwargs)


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def safe_json_default(o):
    if isinstance(o, np.generic):
        return o.item()
    elif isinstance(o, np.ndarray):
        return o.tolist()
    elif hasattr(o, '__dict__'):
        return o.__dict__
    else:
        return str(o)  # fallback for enums, etc.


# Simple Namespace ---------------------------------------------------------------------------------------------------------------------
def json_dumps_simple_nsp(simple_nsp):  # dump
    return json_dumps(simple_nsp, default=safe_json_default)


def json_load_simple_nsp(json_string):  # load
    return json.loads(json_string, object_hook=lambda d: SimpleNamespace(**d))


def copy_simple_nsp(simple_nsp):  # dump and load
    return json_load_simple_nsp(json_dumps_simple_nsp(simple_nsp))


def dict_to_simple_nsp(dictionary=None):  # dump and load
    if dictionary is None:
        dictionary = {}
    return json_load_simple_nsp(json_dumps(scientific_notation_to_float(dictionary)))


def simple_nsp_to_dict(simple_nsp):  # dump and load
    return json.loads(json_dumps_simple_nsp(simple_nsp))


def simple_nsps_to_params(*simple_nsps):
    params = {}
    # Merging each SimpleNamespace into the params dictionary
    for simple_nsp in simple_nsps:
        params.update(vars(simple_nsp))
    return params


def update_simple_nsp(simple_nsps, updates):
    for key, value in dict(updates).items():
        setattr(simple_nsps, key, value)


# File I/O -----------------------------------------------------------------------------------------------------------------------------


# Read
def read_jsons_in_folder(folder_path):
    """loads all .json files inside given folder and returns their list"""
    files = [folder_path + '/' + name for name in os.listdir(folder_path) if name[-5:] == '.json']
    jss = []
    for file in files:
        f = open(file)
        jss.append(json.load(f))
        f.close()
    return jss


def json_file_to_dict(json_file_path):
    with open(json_file_path) as json_file:
        return json.load(json_file)


def json_file_to_simple_nsp(json_file_path):
    return dict_to_simple_nsp(json_file_to_dict(json_file_path))


def yaml_file_to_dict(yaml_file_path):
    with open(yaml_file_path) as yaml_file:
        return yaml.safe_load(yaml_file)


def yaml_file_to_simple_nsp(yaml_file_path):
    return dict_to_simple_nsp(yaml_file_to_dict(yaml_file_path))


# Write
def dict_to_json_file(dict, json_file_path):
    with open(json_file_path, 'w') as json_file:
        json_file.write(json_dumps(dict))


def simple_nsp_to_json_file(object, json_file_path):
    with open(json_file_path, 'w') as json_file:
        json_file.write(json_dumps_simple_nsp(object))


def dict_to_yaml_file(dict, yaml_file_path):
    with open(yaml_file_path, 'w') as yaml_file:
        yaml.dump(dict, yaml_file, default_flow_style=False)


def simple_nsp_to_yaml_file(object, yaml_file_path):
    dict_to_yaml_file(simple_nsp_to_dict(object), yaml_file_path)


# Pickle
def pickle_save(object, path):
    with open(path, 'wb') as f:
        pickle.dump(object, f)


def pickle_load(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


# File info
def file_len(path, print_lines=False):
    with open(path) as f:
        l = sum(1 for _ in tqdm(f))
    if print_lines:
        print(f'{l} lines in {path}')
    return l


def sizeof_file(file):
    return format_size(os.path.getsize(file)) if os.path.exists(file) else 0


# Data transforms ----------------------------------------------------------------------------------------------------------------------
def scientific_notation_to_float(item):
    if isinstance(item, dict):
        return {k: scientific_notation_to_float(v) for k, v in item.items()}
    elif isinstance(item, list):
        return [scientific_notation_to_float(v) for v in item]
    elif isinstance(item, str):
        try:
            return float(item) if 'e' in item else item
        except ValueError:
            return item
    else:
        return item


def format_size(num, suffix='B'):
    for unit in ['', 'Ki', 'Mi', 'Gi', 'Ti', 'Pi', 'Ei', 'Zi']:
        if abs(num) < 1024.0:
            return f'{num:3.1f}{unit}{suffix}'
        num /= 1024.0
    return '{:.1f}{}{}'.format(num, 'Yi', suffix)


def sizeof_object(object):
    return format_size(sys.getsizeof(object))


def to_utf8_if_binary(text):
    if isinstance(text, list) and isinstance(text[0], bytes):
        text = [t.decode('utf-8') for t in text]
    elif isinstance(text, bytes):
        text = text.decode('utf-8')
    return text


def get_placeholder(name):
    return '<PLACEHOLDER:' + name + '>'


# Collection helpers -------------------------------------------------------------------------------------------------------------------
def is_sublist(smaller_list, larger_list):
    """
    Check if smaller_list is a sublist of larger_list.

    Args:
    - smaller_list (list or np.array): The list to be checked as a sublist.
    - larger_list (list or np.array): The list in which to search for the sublist.

    Returns:
    - bool: True if smaller_list is a sublist of larger_list, False otherwise.
    """

    # Convert to numpy arrays if they are not already
    if not isinstance(smaller_list, np.ndarray):
        smaller_list = np.array(smaller_list)
    if not isinstance(larger_list, np.ndarray):
        larger_list = np.array(larger_list)

    len_smaller = len(smaller_list)
    len_larger = len(larger_list)

    # Edge case: if smaller list is empty, it's considered a sublist
    if len_smaller == 0:
        return True

    # Edge case: if smaller list is larger than the larger list, it can't be a sublist
    if len_smaller > len_larger:
        return False

    # Sliding window approach with numpy
    for i in range(len_larger - len_smaller + 1):
        if np.array_equal(larger_list[i : i + len_smaller], smaller_list):
            return True

    return False


def try_set_add(set, element):
    """
    Attempts to add an element to the set.
    Returns True if the element was added (it did not exist in the set).
    Returns False if the element was not added (it already existed in the set).
    """
    l = len(set)
    set.add(element)
    return True if l != len(set) else False


def dict_diff(d1, d2, print_diff=True):
    """
    Compare two dictionaries and show the differences between their keys and values.

    Args:
        d1: First dictionary
        d2: Second dictionary
        print_diff: If True, print the differences (default: True)

    Returns:
        Dictionary containing the differences
    """
    if not isinstance(d1, dict) or not isinstance(d2, dict):
        if print_diff:
            p('One or both inputs are not dictionaries')
        return {'error': 'One or both inputs are not dictionaries'}

    diff = {}

    # Find keys in d1 but not in d2
    only_in_d1 = set(d1.keys()) - set(d2.keys())
    if only_in_d1:
        diff['only_in_first'] = {k: d1[k] for k in only_in_d1}

    # Find keys in d2 but not in d1
    only_in_d2 = set(d2.keys()) - set(d1.keys())
    if only_in_d2:
        diff['only_in_second'] = {k: d2[k] for k in only_in_d2}

    # Find keys with different values
    common_keys = set(d1.keys()) & set(d2.keys())
    diff_values = {}

    for k in common_keys:
        # If values are dictionaries, recursively compare them
        if isinstance(d1[k], dict) and isinstance(d2[k], dict):
            sub_diff = dict_diff(d1[k], d2[k], print_diff=False)
            if sub_diff and not (len(sub_diff) == 1 and 'error' in sub_diff):
                diff_values[k] = sub_diff
        # Otherwise directly compare values
        elif d1[k] != d2[k]:
            diff_values[k] = {'first': d1[k], 'second': d2[k]}

    if diff_values:
        diff['different_values'] = diff_values

    # Print the differences if requested
    if print_diff:
        if not diff:
            p('No differences found between the dictionaries')
        else:
            p(diff)

    return diff


# Introspection ------------------------------------------------------------------------------------------------------------------------
def filter_kwargs_by_method_signature(method, kwargs):
    """Filter kwargs to only include valid parameters for the given method."""
    signature = inspect.signature(method)
    valid_params = set(signature.parameters)
    return {k: v for k, v in kwargs.items() if k in valid_params}


def is_valid_uuid(uuid_to_test, version=4):
    try:
        uuid_obj = UUID(uuid_to_test, version=version)
    except ValueError:
        return False
    return str(uuid_obj) == uuid_to_test


def monkey_patch_globally(name: str, new_obj, verbose=False):
    count = 0
    for module in list(sys.modules.values()):
        if module and hasattr(module, '__dict__') and name in module.__dict__:
            module.__dict__[name] = new_obj
            count += 1
            if verbose:
                print(f"Patched '{name}' in {module.__name__}")
    return count


# Script args --------------------------------------------------------------------------------------------------------------------------
def get_script_param(len, number, default=None, p=False):
    param = default if len <= number else sys.argv[number]
    param = type(default)(param) if param is not None and default is not None else param
    print(param) if p else None
    return param


def parse_script_args(ap=None):
    """
    Extention Example:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--arg_2', type=str, help='Named Flag Argument')
    args, config_name = parse_script_args(ap)
    """
    import argparse

    config_name_only = False if ap else True
    ap = ap or argparse.ArgumentParser()
    ap.add_argument('--config', required=True, help='Configuration File Name')
    args = ap.parse_args()
    return args.config if config_name_only else (args, args.config)


def parse_config_name():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument('config_name', help='Configuration File Name')
    args = ap.parse_args()
    return args.config_name


# Debug / Print ------------------------------------------------------------------------------------------------------------------------
def print_list(list):
    for element in list:
        print(element)


def print_stack():
    print('Call stack:')
    for frame in inspect.stack():
        print(f'  {frame.filename}:{frame.lineno} - {frame.function}')


def p(*objects, end='\n', sep=' '):
    """
    Pretty prints multiple objects in a readable format using appropriate dump functions.

    Args:
        *objects: One or more objects of any type to be printed
        end: String to append after the last value (default: newline)
        sep: Separator between objects (default: space)
    """
    result = []

    for obj in objects:
        if isinstance(obj, SimpleNamespace):
            result.append(json_dumps_simple_nsp(obj))
        elif isinstance(obj, np.ndarray):
            result.append(json_dumps_numpy(obj))
        elif isinstance(obj, dict):
            result.append(json_dumps(obj))
        elif isinstance(obj, (KeysView, ValuesView, ItemsView, set, frozenset)):
            result.append(json_dumps(list(obj)))
        elif isinstance(obj, list) and len(obj) > 0 and isinstance(obj[0], dict):
            result.append(json_dumps(obj))
        elif isinstance(obj, (int, float, str, bool, type(None))):
            result.append(str(obj))
        else:
            try:
                # Try to convert to dict if object has __dict__ attribute
                if hasattr(obj, '__dict__'):
                    result.append(json_dumps(obj.__dict__))
                else:
                    result.append(str(obj))
            except Exception:
                result.append(str(obj))

    rprint(sep.join(result), end=end)
