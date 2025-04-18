import argparse
from collections import OrderedDict

import torch


def prune_checkpoint(input_path: str, save_path: str):
    # Load the checkpoint
    chkp = torch.load(input_path, map_location="cpu")
    chkp = chkp["state_dict"]

    # Extract only the backbone weights
    distilled_chkp = OrderedDict()
    for key in chkp:
        if key.startswith("model.backbone."):
            distilled_chkp[key.replace("model.backbone.", "")] = chkp[key]

    # Save the pruned checkpoint
    torch.save(distilled_chkp, save_path)
    print(f"Pruned backbone checkpoint saved to {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prune checkpoint to save only backbone weights.")
    parser.add_argument("input_path", type=str, help="Path to the original checkpoint file.")
    parser.add_argument("save_path", type=str, help="Path to save the pruned backbone checkpoint.")
    args = parser.parse_args()

    prune_checkpoint(args.input_path, args.save_path)
