from functools import wraps
import inspect
from torch.nn import Module as TorchModule
from tensorflow import Module as TensorflowModule
from tensorflow.keras import Model as KerasModel
from tensorflow.keras.layers import Layer as KerasLayer
import json
import requests
from dotenv import load_dotenv, dotenv_values
from loguru import logger
from maave._config import config


def analyze_forward(forward_func, name, offline=False, debug=False):
    @wraps(forward_func)
    def wrap_forward(*args, **kwargs):
        arguments = inspect.getfullargspec(forward_func)
        print(arguments)
        return forward_func(*args, **kwargs)

    return wrap_forward


def analyze_constructor(constructor_func, name, offline=False, verbosity="", debug=False):
    assert verbosity in ["debug", "info", "warning", ""]
    arguments = inspect.getfullargspec(constructor_func)

    if debug:
        logger.info(arguments)

    @wraps(constructor_func)
    def wrap_constructor(*args, **kwargs):
        if not offline:
            #  POST the arguments to the names model
            #  check if arguments exist for this model
            #  - if they do, add another entry
            #  - if they don't, new model version
            pass
        else:
            pass
        return constructor_func(*args, **kwargs)

    return wrap_constructor


def analyze(component, name, offline=False, debug=True):
    if not offline:
        try:
            model_info = requests.get(f"{config['SERVER_ADDRESS']}/model?name={name}", timeout=2)  # TODO: Add token
        except requests.exceptions.Timeout:
            logger.warning("Request timed out, using offline mode.")
            offline = True

    if inspect.isclass(component):
        if issubclass(component, TorchModule):
            if debug:
                logger.info("Analyzing TorchModule with name: {name}", name=name)
            setattr(component, "forward", analyze_forward(vars(component).get("forward"), name, offline, debug=debug))
            setattr(component, "__init__", analyze_constructor(vars(component).get("__init__"), name, offline, debug=debug))
            return component
        elif issubclass(component, TensorflowModule):
            if debug:
                logger.info("Analyzing TensorflowModule with name: {name}", name=name)
        elif issubclass(component, KerasModel):
            if debug:
                logger.info("Analyzing KerasModel with name: {name}", name=name)
            """
            I can access the layers with .layers method
            """

        elif issubclass(component, KerasLayer):
            if debug:
                logger.info("Analyzing KerasLayer with name: {name}", name=name)
    elif inspect.isfunction(component):
        if debug:
            logger.info("Analyzing functional component with name: {name}", name=name)
    else:
        raise TypeError("Expected class or function object")
