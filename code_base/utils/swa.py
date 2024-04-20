import os
import re
from collections import OrderedDict
from pprint import pprint
from typing import List, Optional

import torch


def delete_prefix_from_chkp(chkp_dict: OrderedDict, prefix: str):
    new_dict = OrderedDict()
    for k in chkp_dict.keys():
        if k.startswith(prefix):
            new_dict[k[len(prefix) :]] = chkp_dict[k]
        else:
            new_dict[k] = chkp_dict[k]

    return new_dict


def avarage_weights(
    nn_weights: List[OrderedDict],
    delete_prefix: Optional[str] = None,
    take_best: Optional[int] = None,
):
    if take_best is not None:
        print("solo model")
        avaraged_dict = OrderedDict()
        for k in nn_weights[take_best].keys():
            if delete_prefix is not None:
                new_k = k[len(delete_prefix) :]
            else:
                new_k = k

            avaraged_dict[new_k] = nn_weights[take_best][k]
    else:
        n_nns = len(nn_weights)
        if n_nns < 2:
            raise RuntimeError("Please provide more then 2 checkpoints")

        avaraged_dict = OrderedDict()
        for k in nn_weights[0].keys():
            if delete_prefix is not None:
                new_k = k[len(delete_prefix) :]
            else:
                new_k = k

            avaraged_dict[new_k] = sum(nn_weights[i][k] for i in range(n_nns)) / float(n_nns)

    return avaraged_dict


def create_chkp(
    model_chkp_root,
    model_chkp_basename=None,
    model_chkp_regex=None,
    delete_prefix=None,
    swa_sort_rule=None,
    n_swa_to_take=3,
    prune_checkpoint_func=None,
):
    if model_chkp_basename is None:
        basenames = os.listdir(model_chkp_root)
        checkpoints = []
        for el in basenames:
            matches = re.findall(model_chkp_regex, el)
            if not matches:
                continue
            parsed_dict = {key: value for key, value in matches}
            parsed_dict["name"] = el
            checkpoints.append(parsed_dict)
        print("SWA checkpoints")
        pprint(checkpoints)
        checkpoints = sorted(checkpoints, key=swa_sort_rule)
        checkpoints = checkpoints[:n_swa_to_take]
        print("SWA sorted checkpoints")
        pprint(checkpoints)
        if len(checkpoints) > 1:
            checkpoints = [
                torch.load(os.path.join(model_chkp_root, el["name"]), map_location="cpu")["state_dict"]
                for el in checkpoints
            ]
            t_chkp = avarage_weights(nn_weights=checkpoints, delete_prefix=delete_prefix)
        else:
            chkp_path = os.path.join(model_chkp_root, checkpoints[0]["name"])
            print("vanilla model")
            print("Loading", chkp_path)
            t_chkp = torch.load(chkp_path, map_location="cpu")["state_dict"]
            if delete_prefix is not None:
                t_chkp = delete_prefix_from_chkp(t_chkp, delete_prefix)
    else:
        chkp_path = os.path.join(model_chkp_root, model_chkp_basename)
        print("vanilla model")
        print("Loading", chkp_path)
        t_chkp = torch.load(chkp_path, map_location="cpu")["state_dict"]
        if delete_prefix is not None:
            t_chkp = delete_prefix_from_chkp(t_chkp, delete_prefix)

    if prune_checkpoint_func is not None:
        t_chkp = prune_checkpoint_func(t_chkp)
    # print("Missing keys: ", set(t_model.state_dict().keys()) - set(t_chkp))
    # print("Extra keys: ",  set(t_chkp) - set(t_model.state_dict().keys()))
    return t_chkp
