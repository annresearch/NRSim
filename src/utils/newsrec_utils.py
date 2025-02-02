# Copyright (c) Recommenders contributors.
# Licensed under the MIT License.

import random
import re
import yaml

class HParams:
    """Class for holding hyperparameters for DeepRec algorithms."""

    def __init__(self, hparams_dict):
        """Create an HParams object from a dictionary of hyperparameter values.

        Args:
            hparams_dict (dict): Dictionary with the models hyperparameters.
        """
        for val in hparams_dict.values():
            if not (
                isinstance(val, int)
                or isinstance(val, float)
                or isinstance(val, str)
                or isinstance(val, list)
            ):
                raise ValueError(
                    "Hyperparameter value {} should be integer, float, string or list.".format(
                        val
                    )
                )
        self._values = hparams_dict
        for hparam in hparams_dict:
            setattr(self, hparam, hparams_dict[hparam])

    def __repr__(self):
        return "HParams object with values {}".format(self._values.__repr__())

    def values(self):
        """Return the hyperparameter values as a dictionary.

        Returns:
            dict: Dictionary with the hyperparameter values.
        """
        return self._values

def flat_config(config):
    """Flat config loaded from a yaml file to a flat dict.

    Args:
        config (dict): Configuration loaded from a yaml file.

    Returns:
        dict: Configuration dictionary.
    """
    f_config = {}
    category = config.keys()
    for cate in category:
        for key, val in config[cate].items():
            f_config[key] = val
    return f_config

def load_yaml(filename):
    """Load a yaml file.

    Args:
        filename (str): Filename.

    Returns:
        dict: Dictionary.
    """
    try:
        with open(filename, "r") as f:
            config = yaml.load(f, yaml.SafeLoader)
        return config
    except FileNotFoundError:  # for file not found
        raise
    except Exception:  # for other exceptions
        raise IOError("load {0} error!".format(filename))

def check_type(config):
    """Check that the config parameters are the correct type

    Args:
        config (dict): Configuration dictionary.

    Raises:
        TypeError: If the parameters are not the correct type.
    """

    int_parameters = [
        "word_size",
        "his_size",
        "title_size",
        "body_size",
        "npratio",
        "word_emb_dim",
        "attention_hidden_dim",
        "epochs",
        "batch_size",
        "show_step",
        "save_epoch",
        "head_num",
        "head_dim",
        "user_num",
        "filter_num",
        "window_size",
        "gru_unit",
        "user_emb_dim",
        "vert_emb_dim",
        "subvert_emb_dim",
    ]
    for param in int_parameters:
        if param in config and not isinstance(config[param], int):
            raise TypeError("Parameters {0} must be int".format(param))

    float_parameters = ["learning_rate", "dropout"]
    for param in float_parameters:
        if param in config and not isinstance(config[param], float):
            raise TypeError("Parameters {0} must be float".format(param))

    str_parameters = [
        "wordEmb_file",
        "wordDict_file",
        "userDict_file",
        "vertDict_file",
        "subvertDict_file",
        "method",
        "loss",
        "optimizer",
        "cnn_activation",
        "dense_activation" "type",
    ]
    for param in str_parameters:
        if param in config and not isinstance(config[param], str):
            raise TypeError("Parameters {0} must be str".format(param))

    list_parameters = ["layer_sizes", "activation"]
    for param in list_parameters:
        if param in config and not isinstance(config[param], list):
            raise TypeError("Parameters {0} must be list".format(param))

    bool_parameters = ["support_quick_scoring"]
    for param in bool_parameters:
        if param in config and not isinstance(config[param], bool):
            raise TypeError("Parameters {0} must be bool".format(param))


def check_nn_config(f_config):
    """Check neural networks configuration.

    Args:
        f_config (dict): Neural network configuration.

    Raises:
        ValueError: If the parameters are not correct.
    """

    if f_config["model_type"] in ["nrms", "NRMS"]:
        required_parameters = [
            "title_size",
            "his_size",
            "wordEmb_file",
            "wordDict_file",
            "userDict_file",
            "npratio",
            "data_format",
            "word_emb_dim",
            # nrms
            "head_num",
            "head_dim",
            # attention
            "attention_hidden_dim",
            "loss",
            "data_format",
            "dropout",
        ]

    elif f_config["model_type"] in ["naml", "NAML"]:
        required_parameters = [
            "title_size",
            "body_size",
            "his_size",
            "wordEmb_file",
            # "subvertDict_file",
            # "vertDict_file",
            "wordDict_file",
            "userDict_file",
            "npratio",
            "data_format",
            "word_emb_dim",
            "ctg_emb_dim",
            # "subvert_emb_dim",
            # naml
            "filter_num",
            "cnn_activation",
            "window_size",
            "dense_activation",
            # attention
            "attention_hidden_dim",
            "loss",
            "data_format",
            "dropout",
        ]
    elif f_config["model_type"] in ["lstur", "LSTUR"]:
        required_parameters = [
            "title_size",
            "his_size",
            "wordEmb_file",
            "wordDict_file",
            "userDict_file",
            "npratio",
            "data_format",
            "word_emb_dim",
            # lstur
            "gru_unit",
            "type",
            "filter_num",
            "cnn_activation",
            "window_size",
            # attention
            "attention_hidden_dim",
            "loss",
            "data_format",
            "dropout",
        ]
    elif f_config["model_type"] in ["npa", "NPA"]:
        required_parameters = [
            "title_size",
            "his_size",
            "wordEmb_file",
            "wordDict_file",
            "userDict_file",
            "npratio",
            "data_format",
            "word_emb_dim",
            # naml
            "user_emb_dim",
            "filter_num",
            "cnn_activation",
            "window_size",
            # attention
            "attention_hidden_dim",
            "loss",
            "data_format",
            "dropout",
        ]
    else:
        required_parameters = []

    # check required parameters
    for param in required_parameters:
        if param not in f_config:
            raise ValueError("Parameters {0} must be set".format(param))

    if f_config["model_type"] in ["nrms", "NRMS", "lstur", "LSTUR"]:
        if f_config["data_format"] != "news":
            raise ValueError(
                "For nrms and naml models, data format must be 'news', but your set is {0}".format(
                    f_config["data_format"]
                )
            )
    elif f_config["model_type"] in ["naml", "NAML"]:
        if f_config["data_format"] != "naml":
            raise ValueError(
                "For nrms and naml models, data format must be 'naml', but your set is {0}".format(
                    f_config["data_format"]
                )
            )

    check_type(f_config)


def create_hparams(flags):
    """Create the models hyperparameters.

    Args:
        flags (dict): Dictionary with the models requirements.

    Returns:
        HParams: Hyperparameter object.
    """
    init_dict = {
        # data
        "support_quick_scoring": False,
        # models
        "dropout": 0.0,
        "attention_hidden_dim": 200,
        # nrms
        "head_num": 4,
        "head_dim": 100,
        # naml
        "filter_num": 200,
        "window_size": 3,
        "ctg_emb_dim": 100,
        # "subvert_emb_dim": 100,
        # lstur
        "gru_unit": 400,
        "type": "ini",
        # naml
        "user_emb_dim": 50,
        # train
        "learning_rate": 0.001,
        "optimizer": "adam",
        "epochs": 10,
        "batch_size": 1,
        # show info
        "show_step": 1,
    }
    init_dict.update(flags)
    return HParams(init_dict)


def prepare_hparams(yaml_file=None, **kwargs):
    """Prepare the models hyperparameters and check that all have the correct value.

    Args:
        yaml_file (str): YAML file as configuration.

    Returns:
        HParams: Hyperparameter object.
    """
    if yaml_file is not None:
        config = load_yaml(yaml_file)
        config = flat_config(config)
    else:
        config = {}

    config.update(kwargs)

    check_nn_config(config)
    return create_hparams(config)


def word_tokenize(sent):
    """Split sentence into word list using regex.
    Args:
        sent (str): Input sentence

    Return:
        list: word list
    """
    pat = re.compile(r"[\w]+|[.,!?;|]")
    if isinstance(sent, str):
        return pat.findall(sent.lower())
    else:
        return []



