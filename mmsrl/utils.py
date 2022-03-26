
from typing import Any, Callable, Dict, List, Union
import importlib
import logging
import sys
import types

logger = logging.getLogger(__name__)


LABELS = ["hero", "villain", "victim", "other"]

class dotdict(dict):
    """ Dictionary which can be access through var.key instead of var["key"]. """
    def __getattr__(self, name: str):
        if name not in self:
            raise AttributeError(f"Config key {name} not found")
        return dotdict(self[name]) if type(self[name]) is dict else self[name]
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def eval_arg(config: Dict[str, Any], arg: str) -> None:
    """
    Evaluate arg in the context config, and update it.

    The argument is expected to be of the form:
        (parent.)*key(=value)?
    If no value is provided, the key is assumed to be a boolean and True is assigned to it.
    When passing a string argument through the shell, it must be enclosed in quote (like all python string), which usually need to be escaped.
    """
    key: str
    value: Any
    if '=' in arg:
        key, value = arg.split('=', maxsplit=1)
        try:
            value = eval(value, config)
        except NameError:
            pass
        config.pop("__builtins__", None)
    else:
        key, value = arg, True
    path: List[str] = key.split('.')
    for component in path[:-1]:
        config = config[component]
    key = path[-1]
    if isinstance(config, list):
        key = int(key)
    config[key] = value


def import_arg(config: Dict[str, Any], arg: str) -> None:
    """
    Load file arg, and update config with its content.

    The file is loaded in an independent context, all the variable defined in the file (even through import) are added to config, with the exception of builtins and whole modules.
    """
    if arg.endswith(".py"):
        arg = arg[:-3].replace('/', '.')
    module: types.ModuleType = importlib.import_module(arg)
    for key, value in vars(module).items():
        if key not in module.__builtins__ and not key.startswith("__") and not isinstance(value, types.ModuleType):  # pytype: disable=attribute-error
            config[key] = value


def parse_args() -> dotdict:
    """
    Parse command line arguments and return config dictionary.

    Two kind of argument are supported:
        - When the argument starts with -- it is evaluated by the eval_arg function
        - Otherwise the argument is assumed to be a file which is loaded by the import_arg function
    """
    config: Dict[str, Any] = {}
    for arg in sys.argv[1:]:
        if arg.startswith("--"):
            eval_arg(config, arg[2:])
        else:
            import_arg(config, arg)
    return dotdict(config)


def display_dict(output: Callable[[str], None], input: Dict[str, Any], depth: int = 0) -> None:
    """ Display nested dictionaries in input using the provided output function. """
    for key, value in input.items():
        indent = '\t'*depth
        output(f"{indent}{key}:")
        if isinstance(value, dict):
            output('\n')
            display_dict(output, value, depth+1)
        else:
            output(f" {value}\n")


def print_dict(input: Dict[str, Any]) -> None:
    """ Print dictionary to standard output. """
    display_dict(lambda x: print(x, end=""), input)


def log_dict(logger: logging.Logger, input: Dict[str, Any]) -> None:
    """ Log dictionary to the provided logger. """
    class log:
        buf: str = ""

        def __call__(self, x: str) -> None:
            self.buf += x
            if self.buf.endswith('\n'):
                logger.info(self.buf[:-1])
                self.buf = ""
    display_dict(log(), input)


def flatten_dict(input: Dict[str, Any]) -> Dict[str, Union[bool, int, float, str]]:
    """
    Replace nested dict by dot-separated keys, and cast keys to simple types.

    repr() is used to cast non-base-type to str.
    """
    def impl(result: Dict[str, Union[bool, int, float, str]], input: Dict[str, Any], prefix: str):
        for key, value in input.items():
            if isinstance(value, dict):
                impl(result, value, f"{key}.")
            else:
                result[f"{prefix}{key}"] = value if type(value) in [bool, int, float, str] else repr(value)

    result: Dict[str, Union[bool, int, float, str]] = {}
    impl(result, input, "")
    return result
