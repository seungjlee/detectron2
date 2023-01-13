#%%
import logging
import json

from itertools import chain
from tqdm import tqdm
from detectron2.data import get_detection_dataset_dicts
from detectron2.data.datasets import register_coco_instances

# %%
SHOTS = 50
data_path = "../tools/datasets/coco/annotations/instances_train2017.json"
data = json.load(open(data_path))

#%%
categories = []
for cat in data["categories"]:
    categories.append(cat)

id2class = dict()
for category in categories:
    id2class[category['id']] = category['name']

id2img = {}
for i in data["images"]:
    id2img[i["id"]] = i

#%%
image_annotations = {}
for a in tqdm(data["annotations"]):
    a.pop("segmentation")
    if a["image_id"] in image_annotations:
        image_annotations[a["image_id"]].append(a)
    else:
        image_annotations[a["image_id"]] = [a]

# %%
index = 0
used_ids = set()

class_id_to_index = {id: index for index, id in enumerate(id2class.keys())}

sample_annotations = [[] for _ in id2class.keys()]
sample_images = [[] for _ in id2class.keys()]

for img_id, annotations in tqdm(image_annotations.items()):
    if not img_id in used_ids:
        for index, c in enumerate(id2class.keys()):
            if len(sample_annotations[index]) < SHOTS:
                class_found = False
                for annotation in annotations:
                    if annotation["category_id"] is c:
                        class_found = True

                if class_found:
                    for annotation in annotations:
                        class_id = annotation["category_id"]
                        sample_annotations[class_id_to_index[class_id]].append(annotation)
                    sample_images[index].append(id2img[img_id])
                    used_ids.add(img_id)
                    break

# Validate samples.
# The numbers printed when running get_detection_dataset_dicts() can be lower since detectron2 removes
# instances where iscrowd==True.
for index, class_id in enumerate(id2class.keys()):
    for annotation in sample_annotations[index]:
        assert annotation["category_id"] == class_id

min_shots = min([len(x) for x in sample_annotations])
print(f"min_shots = {min_shots}")
assert min_shots == SHOTS

new_data = {
    "images": list(chain(*sample_images)),
    "annotations": list(chain(*sample_annotations)),
    "categories": categories,
}

save_path = f"instances_train2017_{SHOTS}_plus_shots.json"
with open(save_path, "w") as f:
    json.dump(new_data, f)

# %%
logging.basicConfig(level=logging.INFO)
try:
    register_coco_instances(f"coco_2017_train_{SHOTS}_plus_shots", {},
                            save_path,
                            "datasets/coco/images/train2017")
except AssertionError as e:
    print(str(e))

# %%
dataset = get_detection_dataset_dicts(names=(f"coco_2017_train_{SHOTS}_plus_shots",))

# %%
