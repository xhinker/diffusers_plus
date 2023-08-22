from ..models.annotator.openpose import OpenposeDetector
from diffusers.utils import load_image
import numpy as np
from PIL import Image

openpose = OpenposeDetector()

def get_pose_img(
    image
    , detect_hand = True
    , detect_face = True
):
    source_img = load_image(image)
    source_img_array = np.array(source_img)
    pose = openpose(
        source_img_array
        , hand = detect_hand
        , face = detect_face
    )
    return Image.fromarray(pose)

