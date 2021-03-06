import os
import cv2
import copy
import numpy as np
import imgaug as ia
from imgaug import augmenters as iaa
from keras.utils import Sequence
import xml.etree.ElementTree as ET
from utils import BoundBox, bbox_iou


class BatchGeneratorVideo(Sequence):
    def __init__(self, images, 
                       config, 
                       shuffle=True, 
                       jitter=True, 
                       norm=None,
                       sequence_length=8):
        self.generator = None

        self.images = images
        self.config = config

        self.images_sets = dict()
        for i in images:
            vid = i['filename'].split('/')[-1].split('-')[0]
            if vid in self.images_sets:
                self.images_sets[vid].append((i['filename'], i))
            else:
                self.images_sets[vid] = [(i['filename'], i)]

        self.images_video_lengths = dict()
        for k in self.images_sets:
            self.images_video_lengths[k] = len(self.images_sets[k])

        for k in self.images_sets:
            self.images_sets[k] = sorted(self.images_sets[k])

        self.shuffle = shuffle
        self.jitter  = jitter
        self.norm    = norm

        self.sequence_length = sequence_length

        self.anchors = [BoundBox(0, 0, config['ANCHORS'][2*i], config['ANCHORS'][2*i+1]) for i in range(int(len(config['ANCHORS'])//2))]

        ### augmentors by https://github.com/aleju/imgaug
        sometimes = lambda aug: iaa.Sometimes(0.5, aug)

        # Define our sequence of augmentation steps that will be applied to every image
        # All augmenters with per_channel=0.5 will sample one value _per image_
        # in 50% of all cases. In all other cases they will sample new values
        # _per channel_.
        self.aug_pipe = iaa.Sequential(
            [
                # apply the following augmenters to most images
                #iaa.Fliplr(0.5), # horizontally flip 50% of all images
                #iaa.Flipud(0.2), # vertically flip 20% of all images
                #sometimes(iaa.Crop(percent=(0, 0.1))), # crop images by 0-10% of their height/width
                sometimes(iaa.Affine(
                    #scale={"x": (0.8, 1.2), "y": (0.8, 1.2)}, # scale images to 80-120% of their size, individually per axis
                    #translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)}, # translate by -20 to +20 percent (per axis)
                    #rotate=(-5, 5), # rotate by -45 to +45 degrees
                    #shear=(-5, 5), # shear by -16 to +16 degrees
                    #order=[0, 1], # use nearest neighbour or bilinear interpolation (fast)
                    #cval=(0, 255), # if mode is constant, use a cval between 0 and 255
                    #mode=ia.ALL # use any of scikit-image's warping modes (see 2nd image from the top for examples)
                )),
                # execute 0 to 5 of the following (less important) augmenters per image
                # don't execute all of them, as that would often be way too strong
                iaa.SomeOf((0, 5),
                    [
                        #sometimes(iaa.Superpixels(p_replace=(0, 1.0), n_segments=(20, 200))), # convert images into their superpixel representation
                        iaa.OneOf([
                            iaa.GaussianBlur((0, 3.0)), # blur images with a sigma between 0 and 3.0
                            iaa.AverageBlur(k=(2, 7)), # blur image using local means with kernel sizes between 2 and 7
                            iaa.MedianBlur(k=(3, 11)), # blur image using local medians with kernel sizes between 2 and 7
                        ]),
                        iaa.Sharpen(alpha=(0, 1.0), lightness=(0.75, 1.5)), # sharpen images
                        #iaa.Emboss(alpha=(0, 1.0), strength=(0, 2.0)), # emboss images
                        # search either for all edges or for directed edges
                        #sometimes(iaa.OneOf([
                        #    iaa.EdgeDetect(alpha=(0, 0.7)),
                        #    iaa.DirectedEdgeDetect(alpha=(0, 0.7), direction=(0.0, 1.0)),
                        #])),
                        iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255), per_channel=0.5), # add gaussian noise to images
                        iaa.OneOf([
                            iaa.Dropout((0.01, 0.1), per_channel=0.5), # randomly remove up to 10% of the pixels
                            #iaa.CoarseDropout((0.03, 0.15), size_percent=(0.02, 0.05), per_channel=0.2),
                        ]),
                        #iaa.Invert(0.05, per_channel=True), # invert color channels
                        iaa.Add((-10, 10), per_channel=0.5), # change brightness of images (by -10 to 10 of original value)
                        iaa.Multiply((0.5, 1.5), per_channel=0.5), # change brightness of images (50-150% of original value)
                        iaa.ContrastNormalization((0.5, 2.0), per_channel=0.5), # improve or worsen the contrast
                        #iaa.Grayscale(alpha=(0.0, 1.0)),
                        #sometimes(iaa.ElasticTransformation(alpha=(0.5, 3.5), sigma=0.25)), # move pixels locally around (with random strengths)
                        #sometimes(iaa.PiecewiseAffine(scale=(0.01, 0.05))) # sometimes move parts of the image around
                    ],
                    random_order=True
                )
            ],
            random_order=True
        )

        if shuffle: np.random.shuffle(self.images)

    def __len__(self):
        return int(np.ceil(float(len(self.images))/self.config['BATCH_SIZE']))   

    def num_classes(self):
        return len(self.config['LABELS'])

    def size(self):
        return len(self.images)    

    def load_annotation(self, i):
        annots = []

        for obj in self.images[i]['object']:
            annot = [obj['xmin'], obj['ymin'], obj['xmax'], obj['ymax'], self.config['LABELS'].index(obj['name'])]
            annots += [annot]

        if len(annots) == 0: annots = [[]]

        return np.array(annots)

    def load_image(self, i):
        return cv2.imread(self.images[i]['filename'])

    def __getitem__(self, idx):
        l_bound = idx*self.config['BATCH_SIZE']
        r_bound = (idx+1)*self.config['BATCH_SIZE']

        if r_bound > len(self.images):
            r_bound = len(self.images)
            l_bound = r_bound - self.config['BATCH_SIZE']

        instance_count = 0

        x_batch = np.zeros((r_bound - l_bound, self.sequence_length, self.config['IMAGE_H'], self.config['IMAGE_W'], 3))                         # input images
        b_batch = np.zeros((r_bound - l_bound, self.sequence_length, 1     , 1     , 1    ,  self.config['TRUE_BOX_BUFFER'], 4))   # list of self.config['TRUE_self.config['BOX']_BUFFER'] GT boxes
        y_batch = np.zeros((r_bound - l_bound, self.sequence_length, self.config['GRID_H'],  self.config['GRID_W'], self.config['BOX'], 4+1+len(self.config['LABELS'])))                # desired network output

        for ni in range(l_bound, r_bound):
            # augment input image and fix object's position and size

            train_instance = self.images[ni]

            img, all_objs = self.get_seq(ni, jitter=self.jitter)
            
            # construct output from object's x, y, w, h
            true_box_index = 0
            
            for frame_id in range(self.sequence_length):
                for obj in all_objs[frame_id]:
                    if obj['xmax'] > obj['xmin'] and obj['ymax'] > obj['ymin'] and obj['name'] in self.config['LABELS']:
                        center_x = .5*(obj['xmin'] + obj['xmax'])
                        center_x = center_x / (float(self.config['IMAGE_W']) / self.config['GRID_W'])
                        center_y = .5*(obj['ymin'] + obj['ymax'])
                        center_y = center_y / (float(self.config['IMAGE_H']) / self.config['GRID_H'])

                        grid_x = int(np.floor(center_x))
                        grid_y = int(np.floor(center_y))

                        if grid_x < self.config['GRID_W'] and grid_y < self.config['GRID_H']:
                            obj_indx  = self.config['LABELS'].index(obj['name'])
                            
                            center_w = (obj['xmax'] - obj['xmin']) / (float(self.config['IMAGE_W']) / self.config['GRID_W']) # unit: grid cell
                            center_h = (obj['ymax'] - obj['ymin']) / (float(self.config['IMAGE_H']) / self.config['GRID_H']) # unit: grid cell
                            
                            box = [center_x, center_y, center_w, center_h]

                            # find the anchor that best predicts this box
                            best_anchor = -1
                            max_iou     = -1
                            
                            shifted_box = BoundBox(0, 
                                                   0,
                                                   center_w,                                                
                                                   center_h)
                            
                            for i in range(len(self.anchors)):
                                anchor = self.anchors[i]
                                iou    = bbox_iou(shifted_box, anchor)
                                
                                if max_iou < iou:
                                    best_anchor = i
                                    max_iou     = iou
                                    
                            # assign ground truth x, y, w, h, confidence and class probs to y_batch
                            y_batch[instance_count, frame_id, grid_y, grid_x, best_anchor, 0:4] = box
                            y_batch[instance_count, frame_id, grid_y, grid_x, best_anchor, 4  ] = 1.
                            y_batch[instance_count, frame_id, grid_y, grid_x, best_anchor, 5+obj_indx] = 1
                            
                            # assign the true box to b_batch
                            b_batch[instance_count, frame_id, 0, 0, 0, true_box_index] = box
                            
                            true_box_index += 1
                            true_box_index = true_box_index % self.config['TRUE_BOX_BUFFER']
                            
            # assign input image to x_batch
            if self.norm != None: 
                for frame_id in range(self.sequence_length):
                    x_batch[instance_count, frame_id] = self.norm(img[frame_id])
            else:
                # plot image and bounding boxes for sanity check
                #for obj in all_objs:
                    # if obj['xmax'] > obj['xmin'] and obj['ymax'] > obj['ymin']:
                    #     cv2.rectangle(img[:,:,::-1], (obj['xmin'],obj['ymin']), (obj['xmax'],obj['ymax']), (255,0,0), 3)
                    #     cv2.putText(img[:,:,::-1], obj['name'], 
                    #                 (obj['xmin']+2, obj['ymin']+12), 
                    #                 0, 1.2e-3 * img.shape[0], 
                    #                 (0,255,0), 2)
                x_batch[instance_count, frame_id] = img[frame_id]

            # increase instance counter in current batch
            instance_count += 1  

        #print(' new batch created', idx)

        # x_batch = np.concatenate([x_batch[:, None]] * 8, axis=1)
        # b_batch = np.concatenate([b_batch[:, None]] * 8, axis=1)
        # y_batch = np.concatenate([y_batch[:, None]] * 8, axis=1)

        #return [x_batch[:, None], b_batch[:, None]], y_batch[:, None]
        return [x_batch, b_batch], y_batch

    def on_epoch_end(self):
        if self.shuffle: np.random.shuffle(self.images)


    def get_seq(self, ni, jitter):
        jitter = False
        image_name = self.images[ni]['filename']

        image = cv2.imread(image_name)

        if image is None: print('Cannot find ', image_name)

        h, w, c = image.shape
        images = np.zeros((self.sequence_length, h, w, c))


        result = np.zeros((self.sequence_length, self.config['IMAGE_H'], self.config['IMAGE_W'], c))
        result_objs = list()
        image_path = '/'.join(self.images[ni]['filename'].split('/')[:-1])
        image_name = self.images[ni]['filename'].split('/')[-1]
        video_id = image_name.split('-')[0]
        frame_id = int(image_name.split('-')[1][:-4])
        for i in range(self.sequence_length):
            if frame_id + i >= self.images_video_lengths[video_id]:
                new_i = self.images_video_lengths[video_id] - 1
            else:
                new_i = frame_id + i

            print(video_id, new_i, self.images_video_lengths[video_id])
            name, train_instance = self.images_sets[video_id][new_i]

            image = train_instance['filename']
            image = cv2.imread(name)
            if image is None: print('Cannot find ', name)

            image = cv2.resize(image, (self.config['IMAGE_H'], self.config['IMAGE_W']))
            image = image[:,:,::-1]

            all_objs = copy.deepcopy(train_instance['object'])
            
            for obj in all_objs:
                for attr in ['xmin', 'xmax']:
                    if jitter: obj[attr] = int(obj[attr] * scale - offx)
                        
                    obj[attr] = int(obj[attr] * float(self.config['IMAGE_W']) / w)
                    obj[attr] = max(min(obj[attr], self.config['IMAGE_W']), 0)
                    
                for attr in ['ymin', 'ymax']:
                    if jitter: obj[attr] = int(obj[attr] * scale - offy)
                        
                    obj[attr] = int(obj[attr] * float(self.config['IMAGE_H']) / h)
                    obj[attr] = max(min(obj[attr], self.config['IMAGE_H']), 0)

                if jitter and flip > 0.5:
                    xmin = obj['xmin']
                    obj['xmin'] = self.config['IMAGE_W'] - obj['xmax']
                    obj['xmax'] = self.config['IMAGE_W'] - xmin

            result[i] = image.copy()
            result_objs.append(all_objs)

        return result, result_objs

