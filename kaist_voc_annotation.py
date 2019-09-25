import xml.etree.ElementTree as ET
import os
from os import getcwd, listdir

dir = '../rgbt-ped-detection/data/kaist-rgbt'
sets = [('00', 9), ('01', 6), ('02', 5), ('03', 2), ('04', 2), ('05', 1), ('06', 5), ('07', 3), ('08', 3), ('09', 1), ('10', 2), ('11', 2)] # 00 - 11

classes = ['person']


def convert_annotation(set, seq, image_id, list_file):
    in_file = open(os.path.join(dir, 'annotations-xml/set%s/V00%d/%s.xml'%(set, seq, image_id)))
    tree=ET.parse(in_file)
    root = tree.getroot()

    for obj in root.iter('object'):
        difficult = obj.find('difficult').text
        cls = obj.find('name').text
        if cls not in classes or int(difficult)==1:
            continue
        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        b = (float(xmlbox.find('xmin').text), float(xmlbox.find('ymin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymax').text))
        list_file.write(" " + ",".join([str(a) for a in b]) + ',' + str(cls_id))

c = 0
for set, n_seq in sets:
    print('Started set%s' % set)
    i = 0
    while i < n_seq:
        image_ids = [s.split('.')[0] for s in sorted(listdir(os.path.join(dir, 'images/set%s/Visible_V00%d/'%(set, i))))]
        list_file = open('train_kaist.txt', 'a')
        if int(set) > 5:
            list_file = open('test_kaist.txt', 'a')
        for image_id in image_ids:
            list_file.write(os.path.join(dir, 'images/set%s/V00%d/%s.png'%(set, i, image_id)))
            convert_annotation(set, i, image_id, list_file)
            list_file.write('\n')
            c += 1
        list_file.close()

        i += 1

print('Wrote %d annotations'%c)

