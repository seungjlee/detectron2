#%%
import logging
import json
import numpy

from itertools import chain
from tqdm import tqdm
from detectron2.data import get_detection_dataset_dicts
from detectron2.data.datasets import register_coco_instances
from pycocotools.cocoeval import Params

# %%
MIN_SHOTS_TARGET = 1
data_path = "../tools/nightowls/nightowls_training_reannotated_yolov7_e6e_vitdet_h75ep.json"
data = json.load(open(data_path))

logging.basicConfig(level=logging.INFO)
# register_coco_instances(f"nightowls_reannotated_train", {},
#                         data_path,
#                         "nightowls/nightowls_training")
# reannotated_dataset = get_detection_dataset_dicts(names=("nightowls_reannotated_train",))

# %% [markdown]
# ### Using a subset of the full data where the number of annotations is relatively large.
CATEGORY_SET = { "person", "bicycle", "car", "motorcycle", "bus", "truck", "traffic light", "stop sign" }
categories = []
for cat in data["categories"]:
    if cat["name"] in CATEGORY_SET:
        categories.append(cat)

id2class = dict()
for category in categories:
    id2class[category["id"]] = category["name"]

BBOX = "bbox"
SIZES = ("S", "M", "L")
category_count = {category["id"]: {size:0 for size in SIZES} for category in categories}

coco_params = Params(iouType=BBOX)

def AreaToSizeString(area):
    if area < coco_params.areaRng[1][1]:
        return "S"
    if area < coco_params.areaRng[2][1]:
        return "M"
    return "L"

# %%
id2img = {}
for i in data["images"]:
    id2img[i["id"]] = i

image_annotations = {}
for a in tqdm(data["annotations"]):
    if a["image_id"] in image_annotations:
        image_annotations[a["image_id"]].append(a)
    else:
        image_annotations[a["image_id"]] = [a]

# %% [markdown]
# ### Assuming that the timestamp is the typical 90kHz based time in MPEG videos.
MIN_TIME_DELTA = 450000 # 5 seconds

last_image_timestamp = 0
index = 0
used_ids = set()

class_id_to_index = {id: index for index, id in enumerate(id2class.keys())}

sample_annotations = [[] for _ in id2class.keys()]
sample_images = [[] for _ in id2class.keys()]

min_shots = 0
while min_shots < MIN_SHOTS_TARGET:
    for img_id, annotations in tqdm(image_annotations.items()):
        if not img_id in used_ids:
            image_info = id2img[img_id]
            image_timestamp = image_info["timestamp"]

            if abs(image_timestamp - last_image_timestamp) < MIN_TIME_DELTA:
                continue

            last_image_timestamp = image_timestamp

            for index, c in enumerate(id2class.keys()):
                for size in list(reversed(SIZES)):
                    if category_count[c][size] < MIN_SHOTS_TARGET:
                        found = False
                        for annotation in annotations:
                            if annotation["category_id"] is c and AreaToSizeString(annotation["area"]) is size:
                                found = True

                        if found:
                            for annotation in annotations:
                                class_id = annotation["category_id"]
                                if class_id in class_id_to_index.keys():
                                    sample_annotations[class_id_to_index[class_id]].append(annotation)
                                    category_count[class_id][AreaToSizeString(annotation["area"])] += 1
                            sample_images[index].append(image_info)
                            used_ids.add(img_id)
                            break
                if img_id in used_ids:
                    break

    min_shots = min([min(counts.values()) for counts in category_count.values()])
    print(f"min_shots = {min_shots}")

assert min_shots >= MIN_SHOTS_TARGET

# Validate samples.
# The numbers printed when running get_detection_dataset_dicts() can be lower since detectron2 removes
# instances where iscrowd==True.
for index, class_id in enumerate(id2class.keys()):
    for annotation in sample_annotations[index]:
        assert annotation["category_id"] == class_id

total_shots = sum([sum(counts.values()) for counts in category_count.values()])

new_data = {
    "images": list(chain(*sample_images)),
    "annotations": list(chain(*sample_annotations)),
    "categories": data["categories"],
}

data_set_name = f"nightowls_reannotated_train_{total_shots}_shots"
save_path = f"nightowls_reannotated_train_{total_shots}_shots.json"

print(f"### **{save_path}**  ")
print("| Category             |   S  |   M  |   L  | Total |")
print("|:---------------------|-----:|-----:|-----:|------:|")
total_counts = [0,0,0,0]
for category_id, counts in category_count.items():
    total_counts = numpy.sum([total_counts, [counts['S'], counts['M'], counts['L'], sum(counts.values())]], axis=0)
    print(f"| {id2class[category_id]:20s} | {counts['S']:-4d} | {counts['M']:-4d} | {counts['L']:-4d} | {sum(counts.values()):-5d} |")
print("|                      |      |      |      |       |")
print(f"| Totals               | {total_counts[0]:-4d} | {total_counts[1]:-4d} | {total_counts[2]:-4d} | {total_counts[3]:-5d} |")

with open(save_path, "w") as f:
    json.dump(new_data, f)

# %%
logging.basicConfig(level=logging.INFO)
try:
    register_coco_instances(data_set_name, {},
                            save_path,
                            "nightowls/nightowls_training")
except AssertionError as e:
    print(str(e))

# %%
dataset = get_detection_dataset_dicts(names=(data_set_name,))

# %%
