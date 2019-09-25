import os
import json


train_imgs = ['../rnd-human-detection/images/2017-08-29/16bit', '../rnd-human-detection/images/2017-08-30/16bit']
val_imgs = ['../rnd-human-detection/images/2017-08-16/16bit']
train_annotations = ['../rnd-human-detection/annotations/hd_20170829.json', '../rnd-human-detection/annotations/hd_20170830.json']
val_annotations = ['../rnd-human-detection/annotations/hd_20170816.json']

f = open('train.txt', 'w')
# Iterate over all datasets.
for images_dir, annotations_filename in zip(train_imgs, train_annotations):
    # Load the JSON file.
    with open(annotations_filename, 'r') as file:
        annotations = json.load(file)

    # Iterate over all images in the dataset.
    for img in annotations:
        img_name = img['filename'].split('/')[-1]
        filename = os.path.join(images_dir, img_name)
        f.write(filename)

        for annotation in img['annotations']:
            # Since the dataset only contains one class, the class ID is always 0 (i.e. 'Person')
            class_id = 0
            xmin = max(annotation['x'], 0)
            ymin = max(annotation['y'], 0)
            width = annotation['width']
            height = annotation['height']
            # Compute `xmax` and `ymax`.
            xmax = min(xmin + width, 640)
            ymax = min(ymin + height, 480)
            box_info = " %d,%d,%d,%d,%d" % (
                xmin, ymin, xmax, ymax, class_id)

            f.write(box_info)
        f.write('\n')

f.close()
print('Finished writing train.txt')

f = open('val.txt', 'w')
# Iterate over all datasets.
for images_dir, annotations_filename in zip(val_imgs, val_annotations):
    # Load the JSON file.
    with open(annotations_filename, 'r') as file:
        annotations = json.load(file)

    # Iterate over all images in the dataset.
    for img in annotations:
        img_name = img['filename'].split('/')[-1]
        filename = os.path.join(images_dir, img_name)
        f.write(filename)

        for annotation in img['annotations']:
            # Since the dataset only contains one class, the class ID is always 0 (i.e. 'Person')
            class_id = 0
            xmin = max(annotation['x'], 0)
            ymin = max(annotation['y'], 0)
            width = annotation['width']
            height = annotation['height']
            # Compute `xmax` and `ymax`.
            xmax = min(xmin + width, 640)
            ymax = min(ymin + height, 480)
            box_info = " %d,%d,%d,%d,%d" % (
                xmin, ymin, xmax, ymax, class_id)

            f.write(box_info)
        f.write('\n')

f.close()
print('Finished writing val.txt')