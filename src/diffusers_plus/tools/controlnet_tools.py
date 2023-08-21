# generate pose
#%%
import sys
sys.path.append("../")

from models.annotator.openpose import OpenposeDetector
openpose = OpenposeDetector()
from diffusers.utils import load_image
import numpy as np
from PIL import Image

#%%
original_pose_img_path = r"D:\az_git_folder\diffusers_plus\az_tests\test_image\pose1.png"
original_pose_img = load_image(original_pose_img_path)
original_pose_img_array = np.array(original_pose_img)

pose = openpose(
    original_pose_img_array
    , hand = True
    , face = True 
)
pose = Image.fromarray(pose)

#%%
pose


#%%
pose.save(r"D:\az_git_folder\azcode\az_projects\azsd\diffusers_plus_test\test_image\detected_pose.png")