import os
import shutil

def copy_runtime(target):
    dest_dir = f"./target/{target}"
    installation_path = os.environ["RYZEN_AI_INSTALLATION_PATH"]
    
    # Copy onnxruntime bin files
    src_path = os.path.join(installation_path, "onnxruntime", "bin")
    print("Copy:", src_path, "->", dest_dir)
    shutil.copytree(src_path, dest_dir, dirs_exist_ok=True)
    
    # Copy vaip_config.json
    src_file = os.path.join(installation_path, "voe-4.0-win_amd64", "vaip_config.json")
    print("Copy:", src_file, "->", dest_dir)
    shutil.copy2(src_file, dest_dir)

if __name__ == "__main__":
    copy_runtime("debug")
    copy_runtime("release")