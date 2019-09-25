"""
Retrain the YOLO model for your own dataset.
"""

import numpy as np
import time
import keras.backend as K
from keras.layers import Input, Lambda
from keras.models import Model
from keras.optimizers import Adam, SGD
from keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping, CSVLogger
from keras.utils import Sequence

from yolo3.model import preprocess_true_boxes, yolo_body, tiny_yolo_body, yolo_loss
from yolo3.utils import get_random_data, get_random_data_hd


def _main():
    tic = time.time()
    annotation_path = 'train_kaist.txt'
    log_dir = 'logs/kaist_stacked_wt/'
    classes_path = 'model_data/hd_classes.txt'
    anchors_path = 'model_data/hd_tiny_yolo_anchors.txt'
    class_names = get_classes(classes_path)
    num_classes = len(class_names)
    anchors = get_anchors(anchors_path)

    input_shape = (512,640) # multiple of 32, hw

    is_tiny_version = len(anchors)==6 # default setting
    if is_tiny_version:
        model = create_tiny_model(input_shape, anchors, num_classes,
            freeze_body=2, weights_path='model_data/yolov3-tiny.h5', channels=4)
    else:
        model = create_model(input_shape, anchors, num_classes,
            freeze_body=2, weights_path='model_data/yolov3.h5') # make sure you know what you freeze

    logging = TensorBoard(log_dir=log_dir)
    checkpoint = ModelCheckpoint(log_dir + 'ep{epoch:03d}-loss{loss:.2f}-val_loss{val_loss:.2f}.h5',
        monitor='val_loss', save_weights_only=True, save_best_only=True, period=3)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, verbose=1)
    early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1)
    csv_logger = CSVLogger('training.log', append=True)

    val_split = 0.1
    with open(annotation_path) as f:
        lines = f.readlines()
    np.random.seed(10101)
    np.random.shuffle(lines)
    np.random.seed(None)
    num_val = int(len(lines)*val_split)
    num_train = len(lines) - num_val

    # Train with frozen layers first, to get a stable loss.
    # Adjust num epochs to your dataset. This step is enough to obtain a not bad model.
    if True:
        model.compile(optimizer=Adam(lr=1e-3), loss={
            # use custom yolo_loss Lambda layer.
            'yolo_loss': lambda y_true, y_pred: y_pred})
        batch_size = 16
        print('Train on {} samples, val on {} samples, with batch size {}.'.format(num_train, num_val, batch_size))
        # model.fit_generator(data_generator_wrapper(lines[:num_train], batch_size, input_shape, anchors, num_classes),
        #         steps_per_epoch=max(1, num_train//batch_size),
        #         validation_data=data_generator_wrapper(lines[num_train:], batch_size, input_shape, anchors, num_classes),
        #         validation_steps=max(1, num_val//batch_size),
        #         epochs=30,
        #         initial_epoch=0,
        #         callbacks=[logging, checkpoint, csv_logger])
        model.fit_generator(KAISTMultiSpecSequence(lines[:num_train], batch_size, input_shape, anchors, num_classes),
                steps_per_epoch=max(1, num_train//batch_size),
                validation_data=KAISTMultiSpecSequence(lines[num_train:], batch_size, input_shape, anchors, num_classes),
                validation_steps=max(1, num_val//batch_size),
                epochs=30,
                initial_epoch=0,
                workers=3,
                use_multiprocessing=True,
                callbacks=[logging, checkpoint, csv_logger])
        model.save_weights(log_dir + 'trained_weights_stage_1.h5')

    # Unfreeze and continue training, to fine-tune.
    # Train longer if the result is not good.
    if True:
        for i in range(len(model.layers)):
            model.layers[i].trainable = True
        model.compile(optimizer=Adam(lr=1e-4), loss={'yolo_loss': lambda y_true, y_pred: y_pred}) # recompile to apply the change
        print('Unfreeze all of the layers.')

        batch_size = 16 # note that more GPU memory is required after unfreezing the body
        print('Train on {} samples, val on {} samples, with batch size {}.'.format(num_train, num_val, batch_size))
        # model.fit_generator(data_generator_wrapper(lines[:num_train], batch_size, input_shape, anchors, num_classes),
        #     steps_per_epoch=max(1, num_train//batch_size),
        #     validation_data=data_generator_wrapper(lines[num_train:], batch_size, input_shape, anchors, num_classes),
        #     validation_steps=max(1, num_val//batch_size),
        #     epochs=80,
        #     initial_epoch=30,
        #     callbacks=[logging, checkpoint, reduce_lr, early_stopping, csv_logger])
        model.fit_generator(KAISTMultiSpecSequence(lines[:num_train], batch_size, input_shape, anchors, num_classes),
            steps_per_epoch=max(1, num_train//batch_size),
            validation_data=KAISTMultiSpecSequence(lines[num_train:], batch_size, input_shape, anchors, num_classes),
            validation_steps=max(1, num_val//batch_size),
            epochs=80,
            initial_epoch=30,
            workers=3,
            use_multiprocessing=True,
            callbacks=[logging, checkpoint, reduce_lr, early_stopping, csv_logger])
        model.save_weights(log_dir + 'trained_weights_stage_2.h5')

    # Further training if needed.
    if True:
        for i in range(len(model.layers)):
            model.layers[i].trainable = True
        model.compile(optimizer=SGD(lr=0.0003, decay=1e-6, momentum=0.9, nesterov=True, clipnorm=5),
                      loss={'yolo_loss': lambda y_true, y_pred: y_pred}) # recompile to apply the change
        print('Unfreeze all of the layers.')

        batch_size = 16 # note that more GPU memory is required after unfreezing the body
        print('Train on {} samples, val on {} samples, with batch size {}.'.format(num_train, num_val, batch_size))
        # model.fit_generator(data_generator_wrapper(lines[:num_train], batch_size, input_shape, anchors, num_classes),
        #     steps_per_epoch=max(1, num_train//batch_size),
        #     validation_data=data_generator_wrapper(lines[num_train:], batch_size, input_shape, anchors, num_classes),
        #     validation_steps=max(1, num_val//batch_size),
        #     epochs=150,
        #     initial_epoch=80,
        #     callbacks=[logging, checkpoint, reduce_lr, early_stopping, csv_logger])
        model.fit_generator(KAISTMultiSpecSequence(lines[:num_train], batch_size, input_shape, anchors, num_classes),
            steps_per_epoch=max(1, num_train//batch_size),
            validation_data=KAISTMultiSpecSequence(lines[num_train:], batch_size, input_shape, anchors, num_classes),
            validation_steps=max(1, num_val//batch_size),
            epochs=150,
            initial_epoch=80,
            workers=3,
            use_multiprocessing=True,
            callbacks=[logging, checkpoint, reduce_lr, early_stopping, csv_logger])
        model.save_weights(log_dir + 'trained_weights_final.h5')

    print('Total training time: %f hrs'%((time.time() - tic)/3600))



def fine_tune_yolo():
    annotation_path = 'train_kaist.txt'
    log_dir = 'logs/003/'
    classes_path = 'model_data/hd_classes.txt'
    anchors_path = 'model_data/hd_tiny_yolo_anchors.txt'
    class_names = get_classes(classes_path)
    num_classes = len(class_names)
    anchors = get_anchors(anchors_path)

    input_shape = (512, 640)  # multiple of 32, hw

    model = create_tiny_model(input_shape, anchors, num_classes,
                             freeze_body=0, weights_path=log_dir + 'trained_weights_final.h5', channels=4)  # make sure you know what you freeze

    logging = TensorBoard(log_dir=log_dir)
    checkpoint = ModelCheckpoint(log_dir + 'ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5',
                                 monitor='val_loss', save_weights_only=True, save_best_only=True, period=3)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, verbose=1)
    early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1)
    csv_logger = CSVLogger('training.log', append=True)

    val_split = 0.2
    with open(annotation_path) as f:
        lines = f.readlines()
    np.random.seed(10101)
    np.random.shuffle(lines)
    np.random.seed(None)
    num_val = int(len(lines) * val_split)
    num_train = len(lines) - num_val

    # Unfreeze and continue training, to fine-tune.
    # Train longer if the result is not good.
    if True:
        for i in range(len(model.layers)):
            model.layers[i].trainable = True
        # model.compile(optimizer=Adam(lr=1e-4),
        #               loss={'yolo_loss': lambda y_true, y_pred: y_pred})  # recompile to apply the change
        model.compile(optimizer=SGD(lr=0.0003, decay=1e-6, momentum=0.9, nesterov=True, clipnorm=5),
                      loss={'yolo_loss': lambda y_true, y_pred: y_pred})
        print('Unfreeze all of the layers.')

        batch_size = 16  # note that more GPU memory is required after unfreezing the body
        print('Train on {} samples, val on {} samples, with batch size {}.'.format(num_train, num_val, batch_size))
        model.fit_generator(data_generator_wrapper(lines[:num_train], batch_size, input_shape, anchors, num_classes),
                            steps_per_epoch=max(1, num_train // batch_size),
                            validation_data=data_generator_wrapper(lines[num_train:], batch_size, input_shape, anchors,
                                                                   num_classes),
                            validation_steps=max(1, num_val // batch_size),
                            epochs=120,
                            initial_epoch=80,
                            callbacks=[logging, checkpoint, reduce_lr, early_stopping, csv_logger])
        model.save_weights(log_dir + 'trained_weights_final_fine.h5')


def get_classes(classes_path):
    '''loads the classes'''
    with open(classes_path) as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names

def get_anchors(anchors_path):
    '''loads the anchors from a file'''
    with open(anchors_path) as f:
        anchors = f.readline()
    anchors = [float(x) for x in anchors.split(',')]
    return np.array(anchors).reshape(-1, 2)


def create_model(input_shape, anchors, num_classes, load_pretrained=True, freeze_body=2,
            weights_path='model_data/yolo_weights.h5'):
    '''create the training model'''
    K.clear_session() # get a new session
    image_input = Input(shape=(None, None, 3))
    h, w = input_shape
    num_anchors = len(anchors)

    y_true = [Input(shape=(h//{0:32, 1:16, 2:8}[l], w//{0:32, 1:16, 2:8}[l], \
        num_anchors//3, num_classes+5)) for l in range(3)]

    model_body = yolo_body(image_input, num_anchors//3, num_classes)
    print('Create YOLOv3 model with {} anchors and {} classes.'.format(num_anchors, num_classes))

    if load_pretrained:
        model_body.load_weights(weights_path, by_name=True, skip_mismatch=True)
        print('Load weights {}.'.format(weights_path))
        if freeze_body in [1, 2]:
            # Freeze darknet53 body or freeze all but 3 output layers.
            num = (185, len(model_body.layers)-3)[freeze_body-1]
            for i in range(num): model_body.layers[i].trainable = False
            print('Freeze the first {} layers of total {} layers.'.format(num, len(model_body.layers)))

    model_loss = Lambda(yolo_loss, output_shape=(1,), name='yolo_loss',
        arguments={'anchors': anchors, 'num_classes': num_classes, 'ignore_thresh': 0.5})(
        [*model_body.output, *y_true])
    model = Model([model_body.input, *y_true], model_loss)

    return model

def create_tiny_model(input_shape, anchors, num_classes, load_pretrained=True, freeze_body=2,
            weights_path='model_data/tiny_yolo_weights.h5', channels=3):
    '''create the training model, for Tiny YOLOv3'''
    K.clear_session() # get a new session
    image_input = Input(shape=(None, None, channels))
    h, w = input_shape
    num_anchors = len(anchors)

    y_true = [Input(shape=(h//{0:32, 1:16}[l], w//{0:32, 1:16}[l], \
        num_anchors//2, num_classes+5)) for l in range(2)]

    model_body = tiny_yolo_body(image_input, num_anchors//2, num_classes)
    print('Create Tiny YOLOv3 model with {} anchors and {} classes.'.format(num_anchors, num_classes))

    if load_pretrained:
        model_body.load_weights(weights_path, by_name=True, skip_mismatch=True)
        print('Load weights {}.'.format(weights_path))

        load_mistmatched_layers_tiny(model_body, weights_path, num_anchors)
        # print(model_body.summary())
        # layer_1_wts = model_body.get_layer('conv2d_1').get_weights()[0]
        # layer_1_wts[:, :, 3, :] = layer_1_wts[:, :, 2, :]
        # print(layer_1_wts[:, :, 2, :])
        # print(layer_1_wts[:, :, 3, :])
        # model_body.get_layer('conv2d_1').set_weights([layer_1_wts])
        # print(model_body.get_weights())

        if freeze_body in [1, 2]:
            # Freeze the darknet body or freeze all but 2 output layers.
            num = (20, len(model_body.layers)-2)[freeze_body-1]
            for i in range(num): model_body.layers[i].trainable = False
            print('Freeze the first {} layers of total {} layers.'.format(num, len(model_body.layers)))

    model_loss = Lambda(yolo_loss, output_shape=(1,), name='yolo_loss',
        arguments={'anchors': anchors, 'num_classes': num_classes, 'ignore_thresh': 0.7})(
        [*model_body.output, *y_true])
    model = Model([model_body.input, *y_true], model_loss)

    return model

def load_mistmatched_layers_tiny(model, weights_path, num_anchors=6):
    ''' load the mismatched layers:
        conv2d_1 (3, 3, 4, None) vs (None, 3, 3, 3)
        conv2d_10 (1, 1, 512, 18) vs (255, 512, 1, 1)
        conv2d_13 (1, 1, 256, 18) vs (255, 256, 1, 1)
    '''
    print('Load mismatched weights.')

    # Create the original model
    org_model = tiny_yolo_body(Input(shape=(None, None, 3)), num_anchors//2, 80)
    org_model.load_weights(weights_path, by_name=True, skip_mismatch=True)

    wts_1 = np.asarray(model.get_layer('conv2d_1').get_weights())
    wts_1[:, :, :, :3, :] = np.asarray(org_model.get_layer(index=1).get_weights())[:, :, :, :3, :]
    wts_1[:, :, :, 3, :] = wts_1[:, :, :, 2, :]
    model.get_layer('conv2d_1').set_weights(wts_1)

    # print(len(org_model.get_layer(index=42).get_weights()))
    # print(len(org_model.get_layer(index=42).get_weights()[0]))
    # print(len(org_model.get_layer(index=42).get_weights()[1]))
    # print(len(model.get_layer(index=42).get_weights()))
    # print(len(model.get_layer(index=42).get_weights()[0]))
    # print(len(model.get_layer(index=42).get_weights()[1]))
    # wts_2 = np.asarray(model.get_layer('conv2d_10').get_weights())
    # print(model.get_layer('conv2d_10').get_weights())
    # wts_2[0] = org_model.get_layer(index=42).get_weights()[0]
    # wts_2[1] = org_model.get_layer(index=42).get_weights()[1][:18]
    # print(wts_2)
    # model.get_layer('conv2d_10').set_weights(wts_2)
    #
    # wts_3 = model.get_layer('conv2d_13').get_weights()
    # wts_3[0] = org_model.get_layer(index=43).get_weights()[0]
    # wts_3[1] = org_model.get_layer(index=43).get_weights()[1][:18]
    # model.get_layer('conv2d_13').set_weights(wts_3)

def data_generator(annotation_lines, batch_size, input_shape, anchors, num_classes):
    '''data generator for fit_generator'''
    n = len(annotation_lines)
    i = 0
    while True:
        image_data = []
        box_data = []
        for b in range(batch_size):
            if i==0:
                np.random.shuffle(annotation_lines)
            # image, box = get_random_data(annotation_lines[i], input_shape)
            image, box = get_random_data_hd(annotation_lines[i], input_shape, fusion=0, equalize=False, brightness=None)
            image_data.append(image)
            box_data.append(box)
            i = (i+1) % n
        image_data = np.array(image_data)
        box_data = np.array(box_data)
        y_true = preprocess_true_boxes(box_data, input_shape, anchors, num_classes)
        # print(image_data.shape)
        yield [image_data, *y_true], np.zeros(batch_size)

def data_generator_wrapper(annotation_lines, batch_size, input_shape, anchors, num_classes):
    n = len(annotation_lines)
    if n==0 or batch_size<=0: return None
    return data_generator(annotation_lines, batch_size, input_shape, anchors, num_classes)


class KAISTMultiSpecSequence(Sequence):
    def __init__(self, annotation_lines, batch_size, input_shape, anchors, num_classes, shuffle=True):
        self.annotation_lines = annotation_lines
        self.num_classes = num_classes
        self.input_shape, self.anchors = input_shape, anchors
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __len__(self):
        return int(np.ceil(len(self.annotation_lines) / float(self.batch_size)))

    def __getitem__(self, idx):
        image_data = []
        box_data = []
        for b in range(idx * self.batch_size, min(len(self.annotation_lines), (idx + 1) * self.batch_size)):
            image, box = get_random_data_hd(self.annotation_lines[b], self.input_shape, fusion=0, equalize=False, brightness=None)
            image_data.append(image)
            box_data.append(box)

        image_data = np.array(image_data)
        box_data = np.array(box_data)
        y_true = preprocess_true_boxes(box_data, self.input_shape, self.anchors, self.num_classes)
        # print(image_data.shape)
        return [image_data, *y_true], np.zeros(image_data.shape[0])

    def on_epoch_end(self):
        # Can do away with this is steps_per_epoch is set to None
        if self.shuffle:
            np.random.shuffle(self.annotation_lines)


if __name__ == '__main__':
    _main()
    # fine_tune_yolo()
