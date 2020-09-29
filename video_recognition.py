#!/usr/bin/env python
#
# This runs the network on the live video feed from the web-cam
# Most parts are borrowed from evaluate.py / eval.py, except OCV stuff

# import the opencv library 
import cv2 

import argparse
import os
import sys
import time

import numpy as np

import keras

# Allow relative imports when being executed as script.
if __name__ == "__main__" and __package__ is None:
	sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
	import keras_retinanet.bin  # noqa: F401
	__package__ = "keras_retinanet.bin"

# Change these to absolute imports if you copy this script outside the keras_retinanet package.
from .. import models
from ..preprocessing.csv_generator import CSVGenerator
from ..preprocessing.pascal_voc import PascalVocGenerator
from ..utils.anchors import make_shapes_callback
from ..utils.config import read_config_file, parse_anchor_parameters, parse_pyramid_levels
from ..utils.eval import evaluate
from ..utils.gpu import setup_gpu
from ..utils.keras_version import check_keras_version
from ..utils.tf_version import check_tf_version
from ..utils.visualization import draw_detections, draw_annotations

import tensorflow as tf

# this fixes "Could not create cudnn handle: CUDNN_STATUS_INTERNAL_ERROR"
physical_devices = tf.config.experimental.list_physical_devices('GPU')
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
tf_config = tf.config.experimental.set_memory_growth(physical_devices[0], True)

# this fixes "failed to allocate 3.05G (3276114944 bytes) from device: CUDA_ERROR_OUT_OF_MEMORY: out of memory"
gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.66)
sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options))

def create_generator(args, preprocess_image):
    """ Create generators for evaluation.
    """
    common_args = {
        'preprocess_image': preprocess_image,
    }

    if args.dataset_type == 'coco':
        # import here to prevent unnecessary dependency on cocoapi
        from ..preprocessing.coco import CocoGenerator

        validation_generator = CocoGenerator(
            args.coco_path,
            'val2017',
            image_min_side=args.image_min_side,
            image_max_side=args.image_max_side,
            config=args.config,
            shuffle_groups=False,
            no_resize=args.no_resize,
            **common_args
        )
    elif args.dataset_type == 'pascal':
        validation_generator = PascalVocGenerator(
            args.pascal_path,
            'test',
            image_extension=args.image_extension,
            image_min_side=args.image_min_side,
            image_max_side=args.image_max_side,
            config=args.config,
            shuffle_groups=False,
            no_resize=args.no_resize,
            **common_args
        )
    elif args.dataset_type == 'csv':
        validation_generator = CSVGenerator(
            args.annotations,
            args.classes,
            image_min_side=args.image_min_side,
            image_max_side=args.image_max_side,
            config=args.config,
            shuffle_groups=False,
            no_resize=args.no_resize,
            **common_args
        )
    else:
        raise ValueError('Invalid data type received: {}'.format(args.dataset_type))

    return validation_generator

def parse_args(args):
	""" Parse the arguments.
	"""
	parser	 = argparse.ArgumentParser(description='Evaluation script for a RetinaNet network.')
	subparsers = parser.add_subparsers(help='Arguments for specific dataset types.', dest='dataset_type')
	subparsers.required = True

	coco_parser = subparsers.add_parser('coco')
	coco_parser.add_argument('coco_path', help='Path to dataset directory (ie. /tmp/COCO).')

	pascal_parser = subparsers.add_parser('pascal')
	pascal_parser.add_argument('pascal_path', help='Path to dataset directory (ie. /tmp/VOCdevkit).')
	pascal_parser.add_argument('--image-extension',   help='Declares the dataset images\' extension.', default='.jpg')

	csv_parser = subparsers.add_parser('csv')
	csv_parser.add_argument('annotations', help='Path to CSV file containing annotations for evaluation.')
	csv_parser.add_argument('classes', help='Path to a CSV file containing class label mapping.')

	parser.add_argument('model',			  help='Path to RetinaNet model.')
	parser.add_argument('video',			  help='Path to video to process.')
	parser.add_argument('--convert-model',	help='Convert the model to an inference model (ie. the input is a training model).', action='store_true')
	parser.add_argument('--backbone',		 help='The backbone of the model.', default='resnet50')
	parser.add_argument('--gpu',			  help='Id of the GPU to use (as reported by nvidia-smi).', type=int)
	parser.add_argument('--score-threshold',  help='Threshold on score to filter detections with (defaults to 0.05).', default=0.05, type=float)
	parser.add_argument('--iou-threshold',	help='IoU Threshold to count for a positive detection (defaults to 0.5).', default=0.5, type=float)
	parser.add_argument('--max-detections',   help='Max Detections per image (defaults to 100).', default=100, type=int)
	parser.add_argument('--save-path',		help='Path for saving images with detections (doesn\'t work for COCO).')
	parser.add_argument('--image-min-side',   help='Rescale the image so the smallest side is min_side.', type=int, default=800)
	parser.add_argument('--image-max-side',   help='Rescale the image if the largest side is larger than max_side.', type=int, default=1333)
	parser.add_argument('--batch-size',       help='Size of the batches.', default=1, type=int)
	parser.add_argument('--no-resize',		help='Don''t rescale the image.', action='store_true')
	parser.add_argument('--config',		   help='Path to a configuration parameters .ini file (only used with --convert-model).')

	return parser.parse_args(args)

def process_frame( frame, generator, model ) :
	raw_image	= frame
	image		= generator.preprocess_image(raw_image.copy())
	image, scale = generator.resize_image(image)
	print scale,

	score_threshold = 0.5	# was 0.05
	max_detections = 100

	if keras.backend.image_data_format() == 'channels_first':
		image = image.transpose((2, 0, 1))

	# run network
	start = time.time()
	boxes, scores, labels = model.predict_on_batch(np.expand_dims(image, axis=0))[:3]
	inference_time = time.time() - start

	#print boxes, scores

	# correct boxes for image scale
	boxes /= scale

	# select indices which have a score above the threshold
	indices = np.where(scores[0, :] > score_threshold)[0]

	# select those scores
	scores = scores[0][indices]

	# find the order with which to sort the scores
	scores_sort = np.argsort(-scores)[:max_detections]

	# select detections
	image_boxes		= boxes[0, indices[scores_sort], :]
	image_scores	= scores[scores_sort]
	image_labels	= labels[0, indices[scores_sort]]
	#image_labels	= [0] * len(scores_sort)

	#image_detections = np.concatenate([image_boxes, np.expand_dims(image_scores, axis=1), np.expand_dims(image_labels, axis=1)], axis=1)

	#draw_annotations(raw_image, generator.load_annotations(i), label_to_name=generator.label_to_name)

	draw_detections(raw_image, image_boxes, image_scores, image_labels, label_to_name=generator.label_to_name, score_threshold=score_threshold)

	return raw_image, inference_time

def process_frames( frames, generator, model ) :
	images = []
	for frame in frames :
		raw_image	= frame
		image		= generator.preprocess_image(raw_image.copy())
		image, scale = generator.resize_image(image)

		if keras.backend.image_data_format() == 'channels_first':
			image = image.transpose((2, 0, 1))

		images.append(image)
		print scale,

	score_threshold = 0.5	# was 0.05
	max_detections = 100

	# run network
	start = time.time()
#	boxes, scores, labels = model.predict_on_batch(np.expand_dims(image, axis=0))[:3]
	boxes, scores, labels = model.predict_on_batch(np.array(images))[:3]
	inference_time = time.time() - start

	'''
	#print boxes, scores

	# correct boxes for image scale
	boxes /= scale

	# select indices which have a score above the threshold
	indices = np.where(scores[0, :] > score_threshold)[0]

	# select those scores
	scores = scores[0][indices]

	# find the order with which to sort the scores
	scores_sort = np.argsort(-scores)[:max_detections]

	# select detections
	image_boxes		= boxes[0, indices[scores_sort], :]
	image_scores	= scores[scores_sort]
	image_labels	= labels[0, indices[scores_sort]]
	#image_labels	= [0] * len(scores_sort)

	#image_detections = np.concatenate([image_boxes, np.expand_dims(image_scores, axis=1), np.expand_dims(image_labels, axis=1)], axis=1)

	#draw_annotations(raw_image, generator.load_annotations(i), label_to_name=generator.label_to_name)

	draw_detections(raw_image, image_boxes, image_scores, image_labels, label_to_name=generator.label_to_name, score_threshold=score_threshold)
	'''

	return inference_time


def main(args=None):
	# parse arguments
	if args is None:
		args = sys.argv[1:]
	args = parse_args(args)

	# make sure keras and tensorflow are the minimum required version
	check_keras_version()
	check_tf_version()

	# optionally choose specific GPU
	if args.gpu:
		setup_gpu(args.gpu)

	# optionally load config parameters
	if args.config:
		args.config = read_config_file(args.config)

	# create the generator
	backbone = models.backbone(args.backbone)
	generator = create_generator(args, backbone.preprocess_image)

	# optionally load anchor parameters
	anchor_params = None
	pyramid_levels = None
	if args.config and 'anchor_parameters' in args.config:
		anchor_params = parse_anchor_parameters(args.config)
	if args.config and 'pyramid_levels' in args.config:
		pyramid_levels = parse_pyramid_levels(args.config)

	# load the model
	print('Loading model, this may take a second...')
	model = models.load_model(args.model, backbone_name=args.backbone)
	generator.compute_shapes = make_shapes_callback(model)

	# optionally convert the model
	if args.convert_model:
		model = models.convert_model(model, anchor_params=anchor_params, pyramid_levels=pyramid_levels)

	# define a video capture object
	vid = cv2.VideoCapture(args.video) if args.video else cv2.VideoCapture(0)

	counter = 0
	start = time.time()
	while(True):

		if args.batch_size == 1 :
			# Capture the video frame by frame
			ret, frame = vid.read()
			if not ret :
				print
				break

			infer_time = 0
			frame, infer_time = process_frame(frame, generator, model)

			# Display the resulting frame
			cv2.imshow('frame', frame)

			# the 'q' button is set as the
			# quitting button you may use any
			# desired button of your choice
			if cv2.waitKey(1) & 0xFF == ord('q'):
				break
		else :
			frames = []
			for _ in range(args.batch_size) :
				ret, frame = vid.read()
				if not ret :
					print
					break
				frames.append(frame)

			if len(frames) < args.batch_size : break

			infer_time = process_frames(frames, generator, model)

		counter += args.batch_size
		print 'FPS: %.2f, inference: %.1f ms' % (counter/(time.time() - start), infer_time*1000), '\r',

	# After the loop release the cap object 
	vid.release() 
	# Destroy all the windows 
	cv2.destroyAllWindows() 

if __name__ == '__main__':
	main()

'''
ResNet
https://github.com/fizyr/keras-models/releases/download/v0.0.1/ResNet-50-model.keras.h5 (36M params)
https://github.com/fizyr/keras-models/releases/download/v0.0.1/ResNet-101-model.keras.h5 (55M params)
https://github.com/fizyr/keras-models/releases/download/v0.0.1/ResNet-152-model.keras.h5 (70M params)

mobilenet
mobilenet128_1.0 : https://github.com/fchollet/deep-learning-models/releases/download/v0.6/mobilenet_1_0_128_tf_no_top.h5 (13M params)

'''
