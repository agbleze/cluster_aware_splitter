#%%
import pandas as pd
import os
from glob import glob
import numpy as np
from clusteval import clusteval
from .feat import (ImgPropertySetReturnType, img_feature_extraction_implementor,
                  extract_object_features_per_image
                  )

img_dir = "field_crop_with_disease"
img_dir = "/home/lin/codebase/__cv_with_roboflow_data/field_crop_with_disease"
img_paths_list = sorted(glob(f"{img_dir}/*"))
img_names = [os.path.basename(img) for img in img_paths_list]
img_property_set = ImgPropertySetReturnType(img_paths=img_paths_list, img_names=img_names, total_num_imgs=100, max_num_clusters=4)


img_property_set = img_feature_extraction_implementor(img_property_set=img_property_set,
                                                    use_cropped_imgs=False
                                                    )

#%%
if __name__ == "__main__":
    print(f"started work")
    featarray = np.array(img_property_set.features)
    ce = clusteval()
    results = ce.fit(featarray)
    clusters = results["labx"]
    imgcluster_dict = {"image_names": img_property_set.img_names, "clusters": clusters}
    imgclust_df = pd.DataFrame.from_dict(imgcluster_dict)
