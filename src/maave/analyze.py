from functools import wraps
import inspect
from torch.nn import Module as TorchModule
from tensorflow import Module as TensorflowModule
from tensorflow.keras import Model as KerasModel
from tensorflow.keras.layers import Layer as KerasLayer


def analyze_forward(forward_func):
    @wraps(forward_func)
    def wrap_forward(*args, **kwargs):
        arguments = inspect.getfullargspec(forward_func)
        print(arguments)
        return forward_func(*args, **kwargs)
    return wrap_forward


def analyze_constructor(constructor_func):
    @wraps(constructor_func)
    def wrap_constructor(*args, **kwargs):
        arguments = inspect.getfullargspec(constructor_func)
        print(arguments)
        return constructor_func(*args, **kwargs)
    return wrap_constructor


def analyze(component):

    if inspect.isclass(component):
        if issubclass(component, TorchModule):
            setattr(component, "forward", analyze_forward(vars(component).get("forward")))
            setattr(component, "__init__", analyze_constructor(vars(component).get("__init__")))
            return component
        elif issubclass(component, TensorflowModule):
            print("this is Tensorflow Module")
        elif issubclass(component, KerasModel):
            print("this is Keras Model")
            """
            I can access the layers with .layers method
            """
        elif issubclass(component, KerasLayer):
            print("this is Keras Layer")
    elif inspect.isfunction(component):
        print("this is a function")
    else:
        raise TypeError("Expected class or function object")
