import os
import shutil

logdirs_root = "logdirs"
output_dir = "all_configs"

# Create the output folder if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

for exp_name in os.listdir(logdirs_root):
    exp_path = os.path.join(logdirs_root, exp_name)
    code_dir = os.path.join(exp_path, "code")
    if not os.path.isdir(code_dir):
        continue

    # Look for a .py file that starts with "__"
    config_file = None
    for fname in os.listdir(code_dir):
        if fname.endswith(".py") and fname.startswith("__"):
            config_file = os.path.join(code_dir, fname)
            break

    if config_file:
        dest_path = os.path.join(output_dir, f"{exp_name}.py")
        shutil.copyfile(config_file, dest_path)
        print(f"Copied: {config_file} → {dest_path}")
    else:
        print(f"No __*.py config found in {code_dir}")
