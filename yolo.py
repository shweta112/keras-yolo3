# -*- coding: utf-8 -*-
"""
Class definition of YOLO_v3 style detection model on image and video
"""

import colorsys
from timeit import default_timer as timer

import numpy as np
from keras import backend as K
from keras.models import load_model
from keras.layers import Input
from PIL import Image, ImageFont, ImageDraw
import cv2

from yolo3.model import yolo_eval, yolo_body, tiny_yolo_body
from yolo3.utils import letterbox_image
import os, glob
from keras.utils import multi_gpu_model


class YOLO(object):
    _defaults = {
        "model_path": 'model_data/yolo.h5',
        "anchors_path": 'model_data/yolo_anchors.txt',
        "classes_path": 'model_data/coco_classes.txt',
        "score" : 0.3,
        "iou" : 0.45,
        "model_image_size" : (416, 416),
        "gpu_num" : 1,
        "no_of_channels" : 3,
    }

    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"

    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults) # set up default values
        self.__dict__.update(kwargs) # and update with user overrides
        self.class_names = self._get_class()
        self.anchors = self._get_anchors()
        self.sess = K.get_session()
        self.boxes, self.scores, self.classes = self.generate()

    def _get_class(self):
        classes_path = os.path.expanduser(self.classes_path)
        with open(classes_path) as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        return class_names

    def _get_anchors(self):
        anchors_path = os.path.expanduser(self.anchors_path)
        with open(anchors_path) as f:
            anchors = f.readline()
        anchors = [float(x) for x in anchors.split(',')]
        return np.array(anchors).reshape(-1, 2)

    def generate(self):
        model_path = os.path.expanduser(self.model_path)
        assert model_path.endswith('.h5'), 'Keras model or weights must be a .h5 file.'

        # Load model, or construct model and load weights.
        num_anchors = len(self.anchors)
        num_classes = len(self.class_names)
        is_tiny_version = num_anchors==6 # default setting
        try:
            self.yolo_model = load_model(model_path, compile=False)
        except:
            self.yolo_model = tiny_yolo_body(Input(shape=(None,None,self.no_of_channels)), num_anchors//2, num_classes) \
                if is_tiny_version else yolo_body(Input(shape=(None,None,self.no_of_channels)), num_anchors//3, num_classes)
            self.yolo_model.load_weights(self.model_path) # make sure model, anchors and classes match
        else:
            assert self.yolo_model.layers[-1].output_shape[-1] == \
                num_anchors/len(self.yolo_model.output) * (num_classes + 5), \
                'Mismatch between model and given anchor and class sizes'

        print('{} model, anchors, and classes loaded.'.format(model_path))

        # Generate colors for drawing bounding boxes.
        hsv_tuples = [(x / len(self.class_names), 1., 1.)
                      for x in range(len(self.class_names))]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(
            map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
                self.colors))
        np.random.seed(10101)  # Fixed seed for consistent colors across runs.
        np.random.shuffle(self.colors)  # Shuffle colors to decorrelate adjacent classes.
        np.random.seed(None)  # Reset seed to default.

        # Generate output tensor targets for filtered bounding boxes.
        self.input_image_shape = K.placeholder(shape=(2, ))
        if self.gpu_num>=2:
            self.yolo_model = multi_gpu_model(self.yolo_model, gpus=self.gpu_num)
        boxes, scores, classes = yolo_eval(self.yolo_model.output, self.anchors,
                len(self.class_names), self.input_image_shape,
                score_threshold=self.score, iou_threshold=self.iou)
        return boxes, scores, classes

    def detect_image(self, image, draw_bb=False):
        start = timer()

        if self.model_image_size != (None, None):
            assert self.model_image_size[0]%32 == 0, 'Multiples of 32 required'
            assert self.model_image_size[1]%32 == 0, 'Multiples of 32 required'
            boxed_image = letterbox_image(image, tuple(reversed(self.model_image_size)))
        else:
            new_image_size = (image.width - (image.width % 32),
                              image.height - (image.height % 32))
            boxed_image = letterbox_image(image, new_image_size)
        image_data = np.array(boxed_image)
        # image_data = np.array(boxed_image, dtype='float32')

        # print(image_data.shape)
        # image_data /= 255.
        image_data = np.expand_dims(image_data, 0)  # Add batch dimension.

        out_boxes, out_scores, out_classes = self.sess.run(
            [self.boxes, self.scores, self.classes],
            feed_dict={
                self.yolo_model.input: image_data,
                self.input_image_shape: [image.size[1], image.size[0]],
                K.learning_phase(): 0
            })

        print('Found {} boxes for {}'.format(len(out_boxes), 'img'))

        if draw_bb:
            font = ImageFont.truetype(font='font/FiraMono-Medium.otf',
                        size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
            thickness = (image.size[0] + image.size[1]) // 300

        detections = []
        for i, c in reversed(list(enumerate(out_classes))):
            predicted_class = self.class_names[c]
            box = out_boxes[i]
            score = out_scores[i]
            label = '{} {:.2f}'.format(predicted_class, score)

            if draw_bb:
                draw = ImageDraw.Draw(image)
                label_size = draw.textsize(label, font)

            top, left, bottom, right = box
            top = max(0, np.floor(top + 0.5).astype('int32'))
            left = max(0, np.floor(left + 0.5).astype('int32'))
            bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
            right = min(image.size[0], np.floor(right + 0.5).astype('int32'))
            print(label, (left, top), (right, bottom))

            if draw_bb:
                if top - label_size[1] >= 0:
                    text_origin = np.array([left, top - label_size[1]])
                else:
                    text_origin = np.array([left, top + 1])

                # My kingdom for a good redistributable image drawing library.
                for i in range(thickness):
                    draw.rectangle(
                        [left + i, top + i, right - i, bottom - i],
                        outline=self.colors[c])
                draw.rectangle(
                    [tuple(text_origin), tuple(text_origin + label_size)],
                    fill=self.colors[c])
                draw.text(text_origin, label, fill=(0, 0, 0), font=font)
                del draw

            # <left> <top> <right> <bottom> <class_id> <confidence>
            detections.append([left, top, right, bottom, c, score])

        end = timer()
        print(end - start)
        return image, detections

    def detect_image_cv(self, image, draw_bb=True, disp_img=None):
        start = timer()

        # if self.model_image_size != (None, None):
        #     assert self.model_image_size[0]%32 == 0, 'Multiples of 32 required'
        #     assert self.model_image_size[1]%32 == 0, 'Multiples of 32 required'
        #     boxed_image = letterbox_image(image, tuple(reversed(self.model_image_size)))
        # else:
        #     new_image_size = (image.width - (image.width % 32),
        #                       image.height - (image.height % 32))
        #     boxed_image = letterbox_image(image, new_image_size)
        # cv_image = np.array(boxed_image)
        # image_data = np.array(boxed_image, dtype='float32')

        # print(image_data.shape)
        # image_data /= 255.

        h, w = image.shape[:2]

        # image_cv = cv2.copyMakeBorder(image_cv, 0, 480 - h, 0, 640 - w, cv2.BORDER_CONSTANT, 0)
        image_cv = cv2.resize(image, (self.model_image_size[1], self.model_image_size[0]) , interpolation=cv2.INTER_AREA)
        # image_cv = np.stack([image_cv] * 3, axis=-1)
        # print(image_cv.shape)
        image_data = np.expand_dims(image_cv, 0)  # Add batch dimension.

        out_boxes, out_scores, out_classes = self.sess.run(
            [self.boxes, self.scores, self.classes],
            feed_dict={
                self.yolo_model.input: image_data,
                self.input_image_shape: [self.model_image_size[0], self.model_image_size[1]],
                K.learning_phase(): 0
            })

        print('Found {} boxes for {}'.format(len(out_boxes), 'img'))

        detections = []
        for i, c in reversed(list(enumerate(out_classes))):
            predicted_class = self.class_names[c]
            box = out_boxes[i]
            score = out_scores[i]
            label = '{} {:.2f}'.format(predicted_class, score)

            top, left, bottom, right = box
            top = max(0, np.floor(top + 0.5)) * h / self.model_image_size[0]
            left = max(0, np.floor(left + 0.5)) * w / self.model_image_size[1]
            bottom = min(self.model_image_size[0], np.floor(bottom + 0.5))  * h / self.model_image_size[0]
            right = min(self.model_image_size[1], np.floor(right + 0.5)) * w / self.model_image_size[1]


            print(label, (left, top), (right, bottom))

            if draw_bb:
                # if top - label_size[1] >= 0:
                #     text_origin = np.array([left, top - label_size[1]])
                # else:
                #     text_origin = np.array([left, top + 1])

                # My kingdom for a good redistributable image drawing library.
                if disp_img is None:
                    disp_img = image
                cv2.rectangle(disp_img, (int(left), int(top)), (int(right), int(bottom)), (0, 255, 0), 1)
                cv2.putText(disp_img, label, (int(left), int(top)), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1, cv2.LINE_AA)

            # <left> <top> <right> <bottom> <class_id> <confidence>
            detections.append([left, top, right, bottom, c, score])

        end = timer()
        print(end - start)
        return disp_img, detections

    def close_session(self):
        self.sess.close()

def detect_video(yolo, video_path, output_path=""):
    vid = cv2.VideoCapture(video_path)
    if not vid.isOpened():
        raise IOError("Couldn't open webcam or video")
    video_FourCC    = int(vid.get(cv2.CAP_PROP_FOURCC))
    video_fps       = vid.get(cv2.CAP_PROP_FPS)
    video_size      = (int(vid.get(cv2.CAP_PROP_FRAME_WIDTH)),
                        int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    isOutput = True if output_path != "" else False
    if isOutput:
        print("!!! TYPE:", type(output_path), type(video_FourCC), type(video_fps), type(video_size))
        out = cv2.VideoWriter(output_path, video_FourCC, video_fps, video_size)
    accum_time = 0
    curr_fps = 0
    fps = "FPS: ??"
    prev_time = timer()
    while True:
        return_value, frame = vid.read()
        image = Image.fromarray(frame)
        image = yolo.detect_image(image)
        result = np.asarray(image)
        curr_time = timer()
        exec_time = curr_time - prev_time
        prev_time = curr_time
        accum_time = accum_time + exec_time
        curr_fps = curr_fps + 1
        if accum_time > 1:
            accum_time = accum_time - 1
            fps = "FPS: " + str(curr_fps)
            curr_fps = 0
        cv2.putText(result, text=fps, org=(3, 15), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.50, color=(255, 0, 0), thickness=2)
        cv2.namedWindow("result", cv2.WINDOW_NORMAL)
        cv2.imshow("result", result)
        if isOutput:
            out.write(result)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    yolo.close_session()

def save_and_detect_imgs(yolo, img_base_path, save_base_path, disp_img_folder=None):
    result_detections = []
    result_images = []

    files = glob.glob(img_base_path + '*.png')
    save_img_path = os.path.join(save_base_path, 'images')
    if not os.path.exists(save_img_path):
        os.mkdir(save_img_path)

    start = timer()
    for f in files:
        # print('img_path', f)
        # image = Image.open(f).resize((640, 480), Image.BILINEAR)

        image_org = cv2.imread(f, cv2.IMREAD_UNCHANGED)
        disp_image = image_org
        if disp_img_folder is not None:
            disp_image = cv2.imread(os.path.join(disp_img_folder, f.split('/')[-1]), cv2.IMREAD_UNCHANGED)
        # image = Image.fromarray(image_cv.astype('uint8'))
        # image.filename = f

        r_image, detections = yolo.detect_image_cv(image_org, disp_img=disp_image)
        cv2.imwrite(os.path.join(save_img_path, f.split('/')[-1]), r_image)
        result_images.append(f)
        result_detections.append(detections)

    end = timer()

    print('FPS: %f'%(len(result_images)/(end - start)))

    print('Saving in ', save_base_path)

    for i, img in enumerate(files):
        o_name = img.split('/')[-1].split('.')[0] + '.txt'
        detections_string = ''
        for d in result_detections[i]:
            # <class_name> <left> <top> <right> <bottom>
            detections_string += '{} {} {} {} {} {}\n'.format(yolo._get_class()[d[4]], d[5], d[0], d[1], d[2], d[3])

        f_name = os.path.join(save_base_path, o_name)
        with open(f_name, 'w') as output_f:
            output_f.write(detections_string)


def detect_kaist(yolo, kaist_path, save_base_path=None):
    if save_base_path is None:
        save_base_path = kaist_path

    # sets = [('10', 2), ('11', 2)]  # 00 - 11
    sets = [('06', 5), ('07', 3), ('08', 3),
            ('09', 1), ('10', 2), ('11', 2)]  # 06 - 11

    for set, n_seq in sets:
        print('Started set%s' % set)
        i = 0

        while i < n_seq:
            image_folder = os.path.join(kaist_path, 'images/set%s/V00%d/'%(set, i))
            disp_img_folder = os.path.join(kaist_path, 'images/set%s/Visible_V00%d/'%(set, i))
            save_folder = os.path.join(save_base_path, 'set%s/V00%d/'%(set, i))
            if not os.path.exists(save_folder):
                os.makedirs(save_folder, exist_ok=True)

            save_and_detect_imgs(yolo, image_folder, save_folder, disp_img_folder)

            i += 1


if __name__ == '__main__':
    # logs/kaist_stacked/ep117-loss1.99-val_loss2.55.h5
    # logs/kaist_stacked_wt/ep111-loss1.84-val_loss2.41.h5

    yolo = YOLO(model_path = 'logs/kaist_stacked_wt/ep111-loss1.84-val_loss2.41.h5',
                classes_path = 'model_data/hd_classes.txt',
                anchors_path = 'model_data/hd_tiny_yolo_anchors.txt',
                model_image_size = (512,640),
                score = 0.3,
                iou = 0.45,
                no_of_channels = 4)
    img_folder = '/media/shweta.mahajan/Daten/Human_Detection/Datasets/flir_17_Sept_2013/Sempach-7/8bit/'
    save_folder = '/media/shweta.mahajan/Daten/GitHub/mAP/predicted/'
    # img_folder = '/media/shweta.mahajan/Daten/rosbags_meppen/AutoAuge_RosbagsUAV2/meppen/meppen_2017-10-17-17-45-23_filtered/8bit/'
    # save_folder = '/media/shweta.mahajan/Daten/rosbags_meppen/AutoAuge_RosbagsUAV2/meppen/meppen_2017-10-17-17-45-23_filtered/pred_tiny_yolov3/'
    # img_folder = '../rnd-human-detection/images/2017-08-29/8bit/'
    # save_folder = '../rnd-human-detection/images/2017-08-29/8bit/pred_tiny_yolov3/'

    # save_and_detect_imgs(yolo, img_folder, save_folder)

    dir = '../rgbt-ped-detection/data/kaist-rgbt'
    detect_kaist(yolo, dir, os.path.join(dir, 'pred_kaist_stacked_wt'))

    yolo.close_session()

    # train_f = '../rnd-human-detection/images/2017-08-29/8bit/frame00775.png'
    # train_img = Image.open(train_f).resize((640, 480), Image.BILINEAR)
    #
    # val_f = '../rnd-human-detection/images/2017-08-16/8bit/frame0667.png'
    # val_img = Image.open(val_f).resize((640, 480), Image.BILINEAR)
    #
    # det_img, det = yolo.detect_image(train_img, True)
    # det_img, det = yolo.detect_image(val_img, True)