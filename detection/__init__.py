from .object_detection import detection
from .image_manipulation import image_man
from .animal_inference import animal_inference
from .small_inference import small_inference
from .sensors import mic, pir
from .deterrents import siren
import os.path


# Delete previous images
if os.path.exists("detection\\images\\person.png"):
    os.remove("detection\\images\\person.png")
if os.path.exists("detection\\images\\cropped_person.png"):
    os.remove("detection\\images\\cropped_person.png")
if os.path.exists("detection\\images\\animal.png"):
    os.remove("detection\\images\\animal.png")
if os.path.exists("detection\\images\\cropped_animal.png"):
    os.remove("detection\\images\\cropped_animal.png")
if os.path.exists("detection\\images\\small_animal.png"):
    os.remove("detection\\images\\small_animal.png")
if os.path.exists("detection\\images\\downsample_small_animal.png"):
    os.remove("detection\\images\\downsample_small_animal.png")
if os.path.exists("detection\\images\\downsample_animal.png"):
    os.remove("detection\\images\\downsample_animal.png")
if os.path.exists("detection\\images\\cropped_small_animal.png"):
    os.remove("detection\\images\\cropped_small_animal.png")