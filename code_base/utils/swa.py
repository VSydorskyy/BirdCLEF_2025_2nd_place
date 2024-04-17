from collections import OrderedDict
from typing import List, Optional


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
