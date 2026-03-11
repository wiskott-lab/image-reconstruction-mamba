import uuid
import config
import ast


def get_parent_file(path):
    return str(path.parent).rsplit('/', 1)[-1]


def get_module_str_from_model(model):
    split = model.__module__.split('.')
    if split[0] == 'timm':
        return split[-1]
    else:
        return split[1]


def get_module_str_from_module(module):
    return module.__name__.split('.')[1]


def generate_tmp_path():
    return config.TMP_DIR / str(uuid.uuid4())


def ensure_list(obj):
    if isinstance(obj, (list, tuple)):
        return obj
    elif obj is None:
        return []
    else:
        return [obj]


def parse_cfg(d):
    if isinstance(d, dict):
        return {k: parse_cfg(v) for k, v in d.items()}
    else:
        return parse_value(d)


def parse_value(value):
    """Safely parse a string representing a tuple of numbers (ints or floats)."""
    if isinstance(value, str):
        if value == 'None':
            return None
        else:
            try:
                parsed = ast.literal_eval(value)
                if isinstance(parsed, tuple) and all(isinstance(x, (int, float)) for x in parsed):
                    return parsed
            except (ValueError, SyntaxError):
                pass  # Return the original value if parsing fails
    return value


def tuples_to_strings(obj):
    if isinstance(obj, dict):
        return {k: tuples_to_strings(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [tuples_to_strings(v) for v in obj]
    elif isinstance(obj, set):
        return {tuples_to_strings(v) for v in obj}
    elif isinstance(obj, tuple) or obj is None:
        return str(obj)  # convert tuple → string
    else:
        return obj