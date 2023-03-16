from typing import List, Tuple
import itertools
from src import datasets

def model_generator(dataset_name: List[str], loss_function: str, learning_rate: float, dl_kwargs: dict, epochs: int,
                    optimizer: Tuple[str, dict], realworld_model: List[str] = None,
                    activation_fcts: List[Tuple[Tuple[str, dict]]] = None,
                    post_activation_fcts: List[Tuple[Tuple[str, dict]]] = None,
                    post_conv_fcts: List[Tuple[Tuple[str, dict]]] = None,
                    final_activation_fcts: List[Tuple[Tuple[str, dict]]] = None, linear_blocks: List[Tuple] = None,
                    conv_blocks: List[Tuple[Tuple]] = None, learning_rate_scheduler: Tuple[str, dict] = None,
                    n_ensembled: List[int]=None) -> List[
    dict]:
    """
    This function creates a list of model_configs which can be passed to BaseModule.
    The inputs are lists of different instances for every part of the model.
    The function returns the Cartesian product of these instances in the required format.

    Arguments:
        activation_fcts, post_activation_fcts, post_conv_fcts, final_activation_fcts:
            A list of the following form, where one fct in the list of fcts can include more than
            one layer.
            [(("layer_name", dict(kwargs)), ("layer_name", dict(kwargs)), ...), ... ]
        linear_blocks: list, where each list entry defines the number of output features of one set of linear layers.
            For example, [(1000, 100, 10), (1024, 128, 16), ...]
        conv_blocks: list, where each entry defines one conv_block, which consists of several convolutional layers.
            For example, [((32, 1, 1, "same"), (64, 1, 1, 0)), ((),(),()), ((),())],
            where (32, 1, 1, "same") means out_channels = 32, kernel_size = 1, stride = 1, padding = "same"

    Returns:
        list of dict: list of model_config. See docstring of BaseModule for details.
    """
    # 1) Get the input arguments into the correct format to be used in itertools.product,
    #    and collect them in lists_for_iterator
    input_args = [arg for arg in locals().copy().items() if arg[1] is not None]
    # Examples for key_arg_tuple: ('datasets', ['MNIST']) or ('activation_fcts', [('ReLU', {}), ('LogSigmoid', {})])
    lists_for_iterator = [format_model_generator_args(*key_arg_tuple) for key_arg_tuple in input_args]

    # 2) Create all combinations of the inputs using itertools.product
    iterator_of_configs = itertools.product(*lists_for_iterator)

    # 3) Turn the iterator of tuples into a list of dicts.
    list_of_configs = []
    for config_tuple in iterator_of_configs:
        model_config = {}
        for entry in config_tuple:
            model_config.update(entry)
        model_config["number_of_classes"] = datasets.get_data_module(model_config["dataset_name"]).number_of_classes
        list_of_configs.append(model_config)

    return list_of_configs


def format_model_generator_args(key: str, arg: List):
    """
    This function converts the input arguments of model_generator to the format
    used in itertools.product

    Arguments:
        arg: The input argument to model_generator()
        key: The name of the input argument

    Returns:
        A list that is passed to itertools.product
    """
    output_list = []

    if key in ["activation_fcts", "post_activation_fcts", "post_conv_fcts", "final_activation_fcts"]:
        # Examples for fct: (('ReLU', {}),) or (('MaxPool2d', {'kernel_size': 2}), ('AvgPool2d', {'kernel_size': 2, 'stride': 2}))
        for fct in arg:
            output_list.append({f"{key}".rpartition("s")[0]: [{"name": layer[0], "kwargs": layer[1]} for layer in fct]})

    elif key == "linear_blocks":
        # Example for linear_block: (1000, 'num_classes')
        for linear_block in arg:
            linear_layers = []
            for number_out_features in linear_block:
                linear_layers.append({"out_features": number_out_features})
            output_list.append({"linear_layers": linear_layers})

    elif key == "conv_blocks":
        # Example for conv_block: ((32, 1, 1, 'same'), (64, 1, 1, 0))
        for conv_block in arg:
            conv_layers = []
            for conv_layer in conv_block:  # conv_layer = Tuple containing int and str elements
                conv_layers.append(
                    {"out_channels": conv_layer[0], "kernel_size": conv_layer[1], "stride": conv_layer[2],
                     "padding": conv_layer[3]})
            output_list.append({"conv_layers": conv_layers})

    else:
        for entry in arg:
            # Example for entry: "MNIST" or {"batch_size": 2048, "num_workers": 12}
            output_list = [{key: x} for x in arg]

    return output_list
