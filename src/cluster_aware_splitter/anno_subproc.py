import subprocess
import os
import json
from glob import glob
import shutil
from tqdm import tqdm
import logging
import uuid

logging.basicConfig(level=logging.DEBUG, 
                    format="%(asctime)s - %(levelname)s - %(message)s",
                    filename="annonymization.logs",
                    filemode="a"
                    )
logger = logging.getLogger(__name__)


def get_cmd(source_target, target_path, model=None,
            **kwargs
            ):
    if not model:
        cmd = ["python3", "anonymize.py", "-s",f"{source_target}", "-t", f"{target_path}"]
    elif model:
        cmd = ["python3", "anonymize.py", "-s",f"{source_target}", "-t", f"{target_path}", "-m", f"{model}"]
    return cmd

def annonymize_in_batches(source_dir, target_dir, 
                          batch_size=10, model=None,
                          **kwargs
                          ):
    save_file = f"annonymize_{str(uuid.uuid4())}.json"
    logger.info(f"Started annonymize_in_batches")
    img_paths = sorted(glob(f"{source_dir}/*"))
    cmd = get_cmd(source_target=source_dir, target_path=target_dir,
                  model=model
                  )
    logger.info(f"command for running annonymize: {cmd}")
    for i in tqdm(range(0, len(img_paths), batch_size),
                  desc=f"Annonymizing images per batches of {batch_size} ...",
                  total=len(img_paths)
                  ):
        logger.info(f"Batching images for index {i} to {i + batch_size}")
        batch_imgpaths = img_paths[i:i+batch_size]
        os.makedirs("batch", exist_ok=True)
        for img_path in batch_imgpaths:
            shutil.copy(img_path, "batch/")
        
        logger.info(f"Started annonymizing ...")
        subprocess.run(cmd, check=True)
        logger.info(f"Finished annonymizing {i} to {i+batch_size}")
        if os.path.exists(save_file):
            logger.info(f"{save_file} already exist hence reading and updating it with the annonymized batched images")
            with open(save_file, "r") as fp:
                processed_data = json.load(fp)
            processed_images = processed_data["image_paths"]
            processed_images.append(batch_imgpaths)
            processed_data["image_paths"] = processed_images
        else:
            logger.info(f"{save_file} does not exist hence creating processed_data to be saved in it")
            processed_data = {"image_paths": batch_imgpaths}
        with open(save_file, "w") as fp:
            json.dump(processed_data, fp, indent=4)
        logger.info(f"Successfully saved processed data in {save_file}")           
        shutil.rmtree("batch")
        logger.info(f"Removed batch directory")
        
            
if __name__ == "__main__":
    human_dir = "/home/lin/codebase/human"
    annonymize_in_batches(source_dir=human_dir, target_dir="output_annonymized", batch_size=10)