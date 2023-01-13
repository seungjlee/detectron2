# %%
import logging
import json

from itertools import chain
from tqdm import tqdm
from detectron2.data import get_detection_dataset_dicts
from detectron2.data.datasets import register_coco_instances

# %%
MIN_SHOTS = 100
data_path = "../nightowls_json/nightowls_training.json"
data = json.load(open(data_path))

# %% [markdown]
# ### Using only the first 3 classes which are valid for training.

# %% 
categories = [
    {"name":"pedestrian","id":1},
    {"name":"bicycledriver","id":2},
    {"name":"motorbikedriver","id":3},
    # {"name":"ignore","id":4}
]

id2class = dict()
for category in categories:
    id2class[category['id']] = category['name']

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
MIN_TIME_DELTA = 1000000 # Roughly over 10 seconds

last_image_timestamp = 0
index = 0
used_ids = set()

class_id_to_index = {id: index for index, id in enumerate(id2class.keys())}

sample_annotations = [[] for _ in id2class.keys()]
sample_images = [[] for _ in id2class.keys()]

min_shots = 0
while min_shots < MIN_SHOTS:
    for img_id, annotations in tqdm(image_annotations.items()):
        if not img_id in used_ids:
            image_info = id2img[img_id]
            image_timestamp = image_info["timestamp"]

            if abs(image_timestamp - last_image_timestamp) < MIN_TIME_DELTA:
                continue

            last_image_timestamp = image_timestamp

            for index, c in enumerate(id2class.keys()):
                if len(sample_annotations[index]) < MIN_SHOTS:
                    class_found = False
                    for annotation in annotations:
                        if annotation["category_id"] is c:
                            class_found = True

                    if class_found:
                        for annotation in annotations:
                            class_id = annotation["category_id"]
                            if class_id != 4:
                                sample_annotations[class_id_to_index[class_id]].append(annotation)
                        sample_images[index].append(image_info)
                        used_ids.add(img_id)
                        break

    min_shots = min([len(x) for x in sample_annotations])
    print(f"min_shots = {min_shots}")

assert min_shots >= MIN_SHOTS

# Validate samples.
# The numbers printed when running get_detection_dataset_dicts() can be lower since detectron2 removes
# instances where iscrowd==True.
for index, class_id in enumerate(id2class.keys()):
    for annotation in sample_annotations[index]:
        assert annotation["category_id"] == class_id

total_shots = sum([len(x) for x in sample_annotations])

new_data = {
    "images": list(chain(*sample_images)),
    "annotations": list(chain(*sample_annotations)),
    "categories": categories,
}

# Not including nightowls in data_set_name here to print the full breakdown with get_detection_dataset_dicts
data_set_name = f"_train_{total_shots}_shots"
save_path = f"nightowls_training_{total_shots}_shots.json"

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
