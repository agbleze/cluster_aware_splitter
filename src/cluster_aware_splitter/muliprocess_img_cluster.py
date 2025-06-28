
#%%
from tqdm import tqdm
import multiprocessing
from feat import get_imgs_and_extract_features_wrapper, ImgPropertySetReturnType
import numpy as np
from clusteval import clusteval
import os
import pandas as pd
from glob import glob

def run_multiprocess(img_property_set,
                    feature_extractor_class = None,
                    seed=2024, img_resize_width=224,
                    img_resize_height=224,
                    model_family="efficientnet",
                    model_name="EfficientNetB0",
                    img_normalization_weight="imagenet",
                    ):
    img_paths = sorted(img_property_set.img_paths)
    args = [{"img_path": img_path, "img_resize_width": img_resize_width,
                 "img_resize_height": img_resize_height, "model_family": model_family,
                 "model_name":model_name, 
                 "img_normalization_weight": img_normalization_weight,
                 "seed": seed, "return_img_path": True
                 } for img_path in img_paths
                ]
    
    num_processes = multiprocessing.cpu_count()
    chunksize = max(1, len(args) // num_processes)
    from tqdm import tqdm
    with multiprocessing.Pool(num_processes) as p:
        results = list(
                    tqdm(
                        p.imap_unordered(get_imgs_and_extract_features_wrapper, 
                                         args, chunksize=chunksize
                                        ),
                        total=len(img_paths),
                    )
                )
    print("multiprocess of imaged feature extration completed")
    images_read = []
    features = []
    image_names = []
    print(f"started clustering")
    for res in results:
        images_read.append(res[0])
        features.append(res[1])
        image_names.append(os.path.basename(res[2]))
    featarray = np.array(features)
    ce = clusteval()
    cluster_results = ce.fit(featarray)
    clusters = cluster_results["labx"]
    imgcluster_dict = {"image_names":image_names, "clusters": clusters}
    imgclust_df = pd.DataFrame.from_dict(imgcluster_dict)
    print("completed clustering")
    return imgclust_df

#%%
if __name__ == '__main__':
    #%%
    img_dir = "field_crop_with_disease"
    img_dir = "/home/lin/codebase/__cv_with_roboflow_data/field_crop_with_disease"
    img_paths_list = sorted(glob(f"{img_dir}/*"))
    img_names = [os.path.basename(img) for img in img_paths_list]
    img_property_set = ImgPropertySetReturnType(img_paths=img_paths_list, img_names=img_names, total_num_imgs=100, max_num_clusters=4)
    
    imgclust_df = run_multiprocess(img_property_set=img_property_set)
    imgclust_df.to_csv("clustering_result.csv")



# %%
