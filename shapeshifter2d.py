#!/usr/bin/env python3.6
#
# Copyright (C) 2019 Intel Corporation
#
# SPDX-License-Identifier: BSD-3-Clause
#
import os
import sys
sys.path.append("../YOLOv3_TensorFlow")

import time
import math
import logging
import numpy as np
import tensorflow as tf

from functools import partial
from PIL import Image

from shapeshifter import parse_args
from shapeshifter import load_and_tile_images
from shapeshifter import create_model
from shapeshifter import create_textures
from shapeshifter import create_attack
from shapeshifter import create_evaluation
from shapeshifter import batch_accumulate
#from shapeshifter import plot

from model import yolov3
from utils.misc_utils import parse_anchors, read_class_names
from utils.data_utils import process_box

def main():
    # Parse args
    args = parse_args(object_roll_min=-5, object_roll_max=5,
                      object_y_min=-500, object_y_max=500,
                      object_z_min=0.5, object_z_max=1.1,

                      texture_roll_min=-10, texture_roll_max=10,
                      texture_x_min=-200, texture_x_max=50,
                      texture_y_min=-100, texture_y_max=150,
                      texture_z_min=0.3, texture_z_max=0.35)

    # Setup logging
    logger = logging.getLogger()
    logger.setLevel(logging.ERROR)
    tf.logging.set_verbosity(tf.logging.ERROR)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    log = logging.getLogger('shapeshifter')
    log.setLevel(logging.DEBUG if args.verbose else logging.INFO)
    formatter = logging.Formatter('%(asctime)s: %(levelname)s %(name)s: %(message)s')
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    log.addHandler(handler)
    log.propagate = False

    # Set seeds from the start
    if args.seed:
        log.debug("Setting seed")
        np.random.seed(args.seed)
        tf.set_random_seed(args.seed)

    # Load textures, backgrounds and objects (and their masks)
    log.debug("Loading backgrounds, textures, textures masks, objects, and objects masks")
    backgrounds = load_and_tile_images(args.backgrounds)

    textures = load_and_tile_images(args.textures)
    textures = textures[:, :, :, :3]
    textures_masks = (load_and_tile_images(args.textures_masks)[:, :, :, :1] >= 0.5).astype(np.float32)
    assert(textures.shape[:3] == textures_masks.shape[:3])

    objects = load_and_tile_images(args.objects)
    objects_masks = (objects[:, :, :, 3] >= 0.5).astype(np.float32)
    objects = objects[:, :, :, :3]
    assert(objects.shape[:3] == objects_masks.shape[:3])

    # YOLOv3 model related parameters
    class_name_path = '../YOLOv3_TensorFlow/data/coco.names'
    anchor_path = '../YOLOv3_TensorFlow/data/yolo_anchors.txt'
    yolo_checkpoint = '../YOLOv3_TensorFlow/data/darknet_weights/yolov3.ckpt'
    anchors = parse_anchors(anchor_path)
    classes = read_class_names(class_name_path)
    class_num = len(classes)
    yolo_model = yolov3(class_num, anchors)

    # Create test data
    generate_data_partial = partial(generate_data,
                                    backgrounds=backgrounds,
                                    objects=objects,
                                    objects_masks=objects_masks,
                                    objects_class=args.target_class,
                                    objects_transforms={'yaw_range': args.object_yaw_range,
                                                        'yaw_bins': args.object_yaw_bins,
                                                        'yaw_fn': args.object_yaw_fn,

                                                        'pitch_range': args.object_pitch_range,
                                                        'pitch_bins': args.object_pitch_bins,
                                                        'pitch_fn': args.object_pitch_fn,

                                                        'roll_range': args.object_roll_range,
                                                        'roll_bins': args.object_roll_bins,
                                                        'roll_fn': args.object_roll_fn,

                                                        'x_range': args.object_x_range,
                                                        'x_bins': args.object_x_bins,
                                                        'x_fn': args.object_x_fn,

                                                        'y_range': args.object_y_range,
                                                        'y_bins': args.object_y_bins,
                                                        'y_fn': args.object_y_fn,

                                                        'z_range': args.object_z_range,
                                                        'z_bins': args.object_z_bins,
                                                        'z_fn': args.object_z_fn},
                                    textures_transforms={'yaw_range': args.texture_yaw_range,
                                                         'yaw_bins': args.texture_yaw_bins,
                                                         'yaw_fn': args.texture_yaw_fn,

                                                         'pitch_range': args.texture_pitch_range,
                                                         'pitch_bins': args.texture_pitch_bins,
                                                         'pitch_fn': args.texture_pitch_fn,

                                                         'roll_range': args.texture_roll_range,
                                                         'roll_bins': args.texture_roll_bins,
                                                         'roll_fn': args.texture_roll_fn,

                                                         'x_range': args.texture_x_range,
                                                         'x_bins': args.texture_x_bins,
                                                         'x_fn': args.texture_x_fn,

                                                         'y_range': args.texture_y_range,
                                                         'y_bins': args.texture_y_bins,
                                                         'y_fn': args.texture_y_fn,

                                                         'z_range': args.texture_z_range,
                                                         'z_bins': args.texture_z_bins,
                                                         'z_fn': args.texture_z_fn},
                                    seed=args.seed,
                                    class_num=class_num,
                                    anchors=anchors)

    # Create adversarial textures, composite them on object, and pass composites into model. Finally, create summary statistics.
    log.debug("Creating perturbable textures")
    textures_var_, textures_ = create_textures(textures, 1.0, # initial_texture doesn't really matter
                                               use_spectral=args.spectral,
                                               soft_clipping=args.soft_clipping)

    log.debug("Creating composited input images")
    input_images_ = create_composited_images(args.batch_size, textures_, textures_masks)

    log.debug("Creating object detection model")
    predictions, detections, losses = create_model(input_images_, args.model_config, args.model_checkpoint, is_training=True)

    log.debug("Creating YOLOv3 loss")
    losses_yolo = create_yolo_loss(input_images_, yolo_model, is_training=True)

    log.debug("Creating attack losses")
    #victim_class_, target_class_, losses_summary_ = create_attack(textures_, textures_var_, predictions, losses,
    #                                                              optimizer_name=args.optimizer, clip=args.sign_gradients)
    victim_class_, target_class_, losses_summary_ = create_attack_2(textures_, textures_var_, predictions, losses,
                                                                    losses_yolo, optimizer_name=args.optimizer,
                                                                    clip=args.sign_gradients)

    log.debug("Creating evaluation metrics")
    metrics_summary_, texture_summary_ = create_evaluation(victim_class_, target_class_,
                                                           textures_, textures_masks, input_images_, detections)

    summaries_ = tf.summary.merge([losses_summary_, metrics_summary_])

    global_init_op_ = tf.global_variables_initializer()
    local_init_op_ = tf.local_variables_initializer()

    # Create tensorboard file writer for train and test evaluations
    saver = tf.train.Saver([textures_var_, tf.train.get_global_step()])
    train_writer = None
    test_writer = None

    if args.logdir is not None:
        log.debug(f"Tensorboard logging: {args.logdir}")
        os.makedirs(args.logdir, exist_ok=True)

        arguments_summary_ = tf.summary.text('Arguments', tf.constant('```' + ' '.join(sys.argv[1:]) + '```'))
        # TODO: Save argparse

        graph = None
        if args.save_graph:
            log.debug("Graph will be saved to tensorboard")
            graph = tf.get_default_graph()

        train_writer = tf.summary.FileWriter(args.logdir + '/train', graph=graph)
        test_writer = tf.summary.FileWriter(args.logdir + '/test')

        # Find existing checkpoint
        os.makedirs(args.logdir + '/checkpoints', exist_ok=True)
        checkpoint_path = tf.train.latest_checkpoint(args.logdir + '/checkpoints')
        args.checkpoint = checkpoint_path

    # Create session
    log.debug("Creating session")
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    sess = tf.Session(config=config)
    sess.run(global_init_op_)
    sess.run(local_init_op_)

    # Set initial texture
    if args.checkpoint is not None:
        log.debug(f"Restoring from checkpoint: {args.checkpoint}")
        saver.restore(sess, args.checkpoint)
    else:
        if args.gray_start:
            log.debug("Setting texture to gray")
            textures = np.zeros_like(textures) + 128/255

        if args.random_start > 0:
            log.debug(f"Adding uniform random perturbation texture with at most {args.random_start}/255 per pixel")
            textures = textures + np.random.randint(size=textures.shape, low=-args.random_start, high=args.random_start)/255

        sess.run('project_op', { 'textures:0': textures  })

    # Get global step
    step = sess.run('global_step:0')

    if train_writer is not None:
        log.debug("Running arguments summary")
        summary = sess.run(arguments_summary_)

        train_writer.add_summary(summary, step)
        test_writer.add_summary(summary, step)

    loss_tensors = ['total_loss:0',
                    'total_rpn_cls_loss:0', 'total_rpn_loc_loss:0',
                    'total_rpn_foreground_loss:0', 'total_rpn_background_loss:0',
                    'total_rpn_cw_loss:0',
                    'total_box_cls_loss:0', 'total_box_loc_loss:0',
                    'total_box_target_loss:0', 'total_box_victim_loss:0',
                    'total_box_target_cw_loss:0', 'total_box_victim_cw_loss:0',
                    'total_sim_loss:0',
                    'grad_l2:0', 'grad_linf:0']

    metric_tensors = ['proposal_average_precision:0', 'victim_average_precision:0', 'target_average_precision:0']

    output_tensors = loss_tensors + metric_tensors

    log.info('global_step [%s]', ', '.join([tensor.replace(':0', '').replace('total_', '') for tensor in output_tensors]))

    test_feed_dict = { 'learning_rate:0': args.learning_rate,
                       'momentum:0': args.momentum,
                       'decay:0': args.decay,

                       'rpn_iou_thresh:0': args.rpn_iou_threshold,
                       'rpn_cls_weight:0': args.rpn_cls_weight,
                       'rpn_loc_weight:0': args.rpn_loc_weight,
                       'rpn_foreground_weight:0': args.rpn_foreground_weight,
                       'rpn_background_weight:0': args.rpn_background_weight,
                       'rpn_cw_weight:0': args.rpn_cw_weight,
                       'rpn_cw_conf:0': args.rpn_cw_conf,

                       'box_iou_thresh:0': args.box_iou_threshold,
                       'box_cls_weight:0': args.box_cls_weight,
                       'box_loc_weight:0': args.box_loc_weight,
                       'box_target_weight:0': args.box_target_weight,
                       'box_victim_weight:0': args.box_victim_weight,
                       'box_target_cw_weight:0': args.box_target_cw_weight,
                       'box_target_cw_conf:0': args.box_target_cw_conf,
                       'box_victim_cw_weight:0': args.box_victim_cw_weight,
                       'box_victim_cw_conf:0': args.box_victim_cw_conf,

                       'sim_weight:0': args.sim_weight,

                       'victim_class:0': args.victim_class,
                       'target_class:0': args.target_class }

    # Keep attacking until CTRL+C. The only issue is that we may be in the middle of some operation.
    try:
        log.debug("Entering attacking loop (use ctrl+c to exit)")
        while True:
            # Run summaries as necessary
            if args.logdir and step % args.save_checkpoint_every == 0:
                log.debug("Saving checkpoint")
                saver.save(sess, args.logdir + '/checkpoints/texture', global_step=step, write_meta_graph=False, write_state=True)

            if step % args.save_texture_every == 0 and test_writer is not None:
                log.debug("Writing texture summary")
                test_texture = sess.run(texture_summary_)
                test_writer.add_summary(test_texture, step)

            if step % args.save_test_every == 0:
                log.debug("Runnning test summaries")
                start_time = time.time()

                sess.run(local_init_op_)
                batch_accumulate(sess, test_feed_dict,
                                 args.test_batch_size, args.batch_size,
                                 generate_data_partial,
                                 detections, predictions, args.categories)

                end_time = time.time()
                log.debug(f"Loss accumulation took {end_time - start_time} seconds")

                test_output = sess.run(output_tensors, test_feed_dict)
                log.info('test %d %s', step, test_output)

                if test_writer is not None:
                    log.debug("Writing test summaries")
                    test_summaries = sess.run(summaries_, test_feed_dict)
                    test_writer.add_summary(test_summaries, step)

            # Create train feed_dict
            train_feed_dict = test_feed_dict.copy()

            train_feed_dict['image_channel_multiplicative_noise:0'] = args.image_multiplicative_channel_noise_range
            train_feed_dict['image_channel_additive_noise:0'] = args.image_additive_channel_noise_range
            train_feed_dict['image_pixel_multiplicative_noise:0'] = args.image_multiplicative_pixel_noise_range
            train_feed_dict['image_pixel_additive_noise:0'] = args.image_additive_pixel_noise_range
            train_feed_dict['image_gaussian_noise_stddev:0'] = args.image_gaussian_noise_stddev_range

            train_feed_dict['texture_channel_multiplicative_noise:0'] = args.texture_multiplicative_channel_noise_range
            train_feed_dict['texture_channel_additive_noise:0'] = args.texture_additive_channel_noise_range
            train_feed_dict['texture_pixel_multiplicative_noise:0'] = args.texture_multiplicative_pixel_noise_range
            train_feed_dict['texture_pixel_additive_noise:0'] = args.texture_additive_pixel_noise_range
            train_feed_dict['texture_gaussian_noise_stddev:0'] = args.texture_gaussian_noise_stddev_range

            # Zero out gradient accumulation, losses, and metrics, then accumulate batches
            log.debug("Starting gradient accumulation...")
            start_time = time.time()


            sess.run(local_init_op_)
            batch_accumulate(sess, train_feed_dict,
                             args.train_batch_size, args.batch_size,
                             generate_data_partial,
                             detections, predictions, args.categories)

            end_time = time.time()
            log.debug(f"Gradient accumulation took {end_time - start_time} seconds")

            train_output = sess.run(output_tensors, train_feed_dict)
            log.info('train %d %s', step, train_output)

            if step % args.save_train_every == 0 and train_writer is not None:
                log.debug("Writing train summaries")
                train_summaries = sess.run(summaries_, test_feed_dict)
                train_writer.add_summary(train_summaries, step)

            # Update textures and project texture to feasible set
            # TODO: We can probably run these together but probably need some control dependency
            log.debug("Projecting attack")
            sess.run('attack_op', train_feed_dict)
            sess.run('project_op')
            step = sess.run('global_step:0')

    except KeyboardInterrupt:
        log.warn('Interrupted')

    finally:
        if test_writer is not None:
            test_writer.close()

        if train_writer is not None:
            train_writer.close()

        if sess is not None:
            sess.close()

def create_composited_images(batch_size, textures, masks, interpolation='BILINEAR'):
    backgrounds_ = tf.placeholder(tf.float32, [batch_size, None, None, 3], name='backgrounds')
    transforms_ = tf.placeholder(tf.float32, shape=[batch_size, 8], name='transforms')

    texture_channel_multiplicative_noise_ = tf.placeholder_with_default([1., 1.], [2], name='texture_channel_multiplicative_noise')
    texture_channel_additive_noise_ = tf.placeholder_with_default([0., 0.], [2], name='texture_channel_additive_noise')

    texture_pixel_multiplicative_noise_ = tf.placeholder_with_default([1., 1.], [2], name='texture_pixel_multiplicative_noise')
    texture_pixel_additive_noise_ = tf.placeholder_with_default([0., 0.], [2], name='texture_pixel_additive_noise')

    texture_gaussian_noise_stddev_ = tf.placeholder_with_default([0., 0.], [2], name='texture_gaussian_noise_stddev')

    image_channel_multiplicative_noise_ = tf.placeholder_with_default([1., 1.], [2], name='image_channel_multiplicative_noise')
    image_channel_additive_noise_ = tf.placeholder_with_default([0., 0.], [2], name='image_channel_additive_noise')

    image_pixel_multiplicative_noise_ = tf.placeholder_with_default([1., 1.], [2], name='image_pixel_multiplicative_noise')
    image_pixel_additive_noise_ = tf.placeholder_with_default([0., 0.], [2], name='image_pixel_additive_noise')

    image_gaussian_noise_stddev_ = tf.placeholder_with_default([0., 0.], [2], name='image_gaussian_noise_stddev')

    backgrounds_shape_ = tf.shape(backgrounds_)

    transformed_textures_ = textures
    transformed_masks_ = masks

    # Add noise to textures
    # TODO: Add blur to transformed textures
    transformed_textures_ = transformed_textures_ * tf.random_uniform([3], texture_channel_multiplicative_noise_[0], texture_channel_multiplicative_noise_[1])
    transformed_textures_ = transformed_textures_ + tf.random_uniform([3], texture_channel_additive_noise_[0], texture_channel_additive_noise_[1])

    transformed_textures_ = transformed_textures_ * tf.random_uniform([], texture_pixel_multiplicative_noise_[0], texture_pixel_multiplicative_noise_[1])
    transformed_textures_ = transformed_textures_ + tf.random_uniform([], texture_pixel_additive_noise_[0], texture_pixel_additive_noise_[1])

    transformed_textures_ = transformed_textures_ + tf.random_normal(transformed_textures_.shape, stddev=tf.random_uniform([], texture_gaussian_noise_stddev_[0], texture_gaussian_noise_stddev_[1]))

    #transformed_textures_ = tf.clip_by_value(transformed_textures_, 0.0, 1.0)

    # Center texture and mask in image the same size as backgrounds
    target_height = backgrounds_shape_[1]
    target_width = backgrounds_shape_[2]
    offset_height = (target_height - textures.shape[1]) // 2
    offset_width = (target_width - textures.shape[2]) // 2

    transformed_textures_ = tf.image.pad_to_bounding_box(transformed_textures_, offset_height, offset_width, target_height, target_width)
    transformed_masks_ = tf.image.pad_to_bounding_box(transformed_masks_, offset_height, offset_width, target_height, target_width)

    # Transform texture and mask ensure output remains the same size as backgrounds
    transformed_textures_ = tf.contrib.image.transform(transformed_textures_, transforms_, interpolation, backgrounds_shape_[1:3])
    transformed_masks_ = tf.contrib.image.transform(transformed_masks_, transforms_, interpolation, backgrounds_shape_[1:3])

    transformed_textures_ = tf.identity(transformed_textures_, name='transformed_textures')
    transformed_masks_ = tf.identity(transformed_masks_, name='transformed_masks')

    input_images_ = backgrounds_ * (1. - transformed_masks_) + transformed_textures_ * (transformed_masks_)

    # Add noise to image
    # TODO: Add blur to composite images
    input_images_ = input_images_ * tf.random_uniform([3], image_channel_multiplicative_noise_[0], image_channel_multiplicative_noise_[1])
    input_images_ = input_images_ + tf.random_uniform([3], image_channel_additive_noise_[0], image_channel_additive_noise_[1])

    input_images_ = input_images_ * tf.random_uniform([], image_pixel_multiplicative_noise_[0], image_pixel_multiplicative_noise_[1])
    input_images_ = input_images_ + tf.random_uniform([], image_pixel_additive_noise_[0], image_pixel_additive_noise_[1])

    input_images_ = input_images_ + tf.random_normal(tf.shape(input_images_), stddev=tf.random_uniform([], image_gaussian_noise_stddev_[0], image_gaussian_noise_stddev_[1]))

    #input_images_ = tf.clip_by_value(input_images_, 0.0, 1.0)

    input_images_ = tf.fake_quant_with_min_max_args(input_images_, min=0., max=1., num_bits=8)
    input_images_ = tf.identity(input_images_, name='input_images')

    return input_images_

"""
def _apply_blur(image):
    image_ = tf.expand_dims(image, axis=0)

    kernel_ = tf.expand_dims(tf.expand_dims(tf.eye(3), 0), 0)

    # Horizontal blur with distance sampled from beta distribution
    horizontal_blur_ = tf.random_gamma([], horizontal_blur_alpha_)
    horizontal_blur_ = horizontal_blur_ / (horizontal_blur_ + tf.random_gamma([], horizontal_blur_beta_))
    horizontal_blur_ = tf.cast(horizontal_blur_ * horizontal_blur_max_, tf.int32) + 1

    horizontal_blur_kernel_ = tf.tile(kernel_, (1, horizontal_blur_, 1, 1))
    horizontal_blur_kernel_ = horizontal_blur_kernel_ / tf.cast(horizontal_blur_, tf.float32)

    image_ = tf.nn.conv2d(image_, horizontal_blur_kernel_, [1, 1, 1, 1], 'SAME')

    # Vertical blur with distance sampled from beta distribution
    vertical_blur_ = tf.random_gamma([], vertical_blur_alpha_)
    vertical_blur_ = vertical_blur_ / (vertical_blur_ + tf.random_gamma([], vertical_blur_beta_))
    vertical_blur_ = tf.cast(vertical_blur_ * vertical_blur_max_, tf.int32) + 1

    vertical_blur_kernel_ = tf.tile(kernel_, (vertical_blur_, 1, 1, 1))
    vertical_blur_kernel_ = vertical_blur_kernel_ / tf.cast(vertical_blur_, tf.float32)

    image_ = tf.nn.conv2d(image_, vertical_blur_kernel_, [1, 1, 1, 1], 'SAME')

    return image_[0]
"""

def create_yolo_loss(input_images, yolo_model, is_training=False):
    true_feature_maps = [tf.placeholder(tf.float32, shape=(None, 8), name=f"truth_feature_maps_{i}") for i in range(input_images.shape[0])]
    pred_feature_maps = []
    with tf.variable_scope('yolov3'):
        for i in range(input_images.shape[0]):
            pred_feature_maps.append(yolo_model.forward(input_images[i], is_training=is_training))
    losses = [yolo_model.compute_loss(pred_feature_maps[i], true_feature_maps[i]) for i in range(input_images.shape[0])]

    return losses

def generate_data(count, backgrounds, objects, objects_masks, objects_class, objects_transforms, textures_transforms, class_num, anchors, seed=None):
    if seed:
        np.random.seed(seed)

    assert(objects.shape[0] == 1)
    assert(objects_masks.shape[0] == 1)

    backgrounds_count, output_height, output_width = backgrounds.shape[0:3]
    origin = (output_width // 2, output_height // 2)

    # Extract bounding boxes after transforming mask
    object_img = Image.fromarray((255*objects[0]).astype(np.uint8))
    mask_img = Image.fromarray((255*objects_masks[0]).astype(np.uint8))

    bboxes = []
    composited_backgrounds = []
    bboxes2 = []

    objects_transforms = generate_transforms(origin, count, **objects_transforms)

    for objects_transform in objects_transforms:
        target_left = (output_width - object_img.width) // 2
        target_upper = (output_height - object_img.height) // 2

        # Transform mask to background dimensions, then compute normalized bounding box
        transformed_mask_img = Image.new('L', (output_width, output_height), (0))
        transformed_mask_img.paste(mask_img, (target_left, target_upper), mask_img)
        transformed_mask_img = transformed_mask_img.transform((output_width, output_height), Image.PERSPECTIVE, objects_transform, Image.BILINEAR)

        # Composite object onto random background and transform
        transformed_object_img = Image.new('RGB', (output_width, output_height), (0, 0, 0))
        transformed_object_img.paste(object_img, (target_left, target_upper), mask_img)
        transformed_object_img = transformed_object_img.transform((output_width, output_height), Image.PERSPECTIVE, objects_transform, Image.BILINEAR)

        composited_background_img = Image.fromarray((255*backgrounds[np.random.choice(backgrounds_count)]).astype(np.uint8))
        composited_background_img.paste(transformed_object_img, (0, 0), transformed_mask_img)

        composited_background = np.array(composited_background_img) / 255
        composited_backgrounds.append(composited_background)

        # Compute normalized bounding box using transformed_mask
        transformed_mask = np.array(transformed_mask_img) / 255
        rows = np.any(transformed_mask[:, :] > 0.5, axis=0)
        cols = np.any(transformed_mask[:, :] > 0.5, axis=1)

        ymin = np.argmax(cols)
        ymax = output_height - np.argmax(cols[::-1]) - 1
        xmin = np.argmax(rows)
        xmax = output_width - np.argmax(rows[::-1]) - 1

        ymin /= output_height
        ymax /= output_height
        xmin /= output_width
        xmax /= output_width

        bbox = np.array([[ymin, xmin, ymax, xmax]], dtype=np.float32)

        bboxes.append(bbox)

        bbox2 = np.array([[xmin, ymin, xmax, ymax]], dtype=np.float32)
        bbox2 = np.concatenate((bbox2, np.full(shape=(bbox2.shape[0], 1), fill_value=1., dtype=np.float32)), axis=-1)
        bboxes2.append(bbox2)

    bboxes = np.array(bboxes)
    composited_backgrounds = np.array(composited_backgrounds)

    # Create texture transforms and compose them with background transforms
    textures_transforms = generate_transforms(origin, count, **textures_transforms)

    objects_transforms = np.concatenate([objects_transforms, np.ones((objects_transforms.shape[0], 1))], axis=1)
    textures_transforms = np.concatenate([textures_transforms, np.ones((textures_transforms.shape[0], 1))], axis=1)

    objects_transforms = np.reshape(objects_transforms, (-1, 3, 3))
    textures_transforms = np.reshape(textures_transforms, (-1, 3, 3))

    transforms = np.matmul(textures_transforms, objects_transforms)
    transforms = np.reshape(transforms, (-1, 9))
    transforms = transforms[:, :8]

    true_feature_maps = []
    for bbox in bboxes2:
        true_feature_maps.append(process_box(bbox, [objects_class - 1], (output_width, output_height), class_num, anchors))

    data = {'transforms:0': transforms,
            'backgrounds:0': composited_backgrounds,
            'groundtruth_boxes_%d:0': bboxes,
            'groundtruth_classes_%d:0': np.full(bboxes.shape[0:2], objects_class - 1),
            'groundtruth_weights_%d:0': np.ones(bboxes.shape[0:2]),
            'true_feature_maps_%d:0': true_feature_maps}

    return data

def generate_transforms(origin, count,
                        yaws=None, yaw_range=(0, 0), yaw_bins=100, yaw_fn=np.linspace,
                        pitchs=None, pitch_range=(0, 0), pitch_bins=100, pitch_fn=np.linspace,
                        rolls=None, roll_range=(0, 0), roll_bins=100, roll_fn=np.linspace,
                        xs=None, x_range=(0, 0), x_bins=100, x_fn=np.linspace,
                        ys=None, y_range=(0, 0), y_bins=100, y_fn=np.linspace,
                        zs=None, z_range=(0, 0), z_bins=100, z_fn=np.linspace):
     # Discretize ranges
    if yaws is None:
        yaws = yaw_fn(*yaw_range, num=yaw_bins)
    if pitchs is None:
        pitchs = pitch_fn(*pitch_range, num=pitch_bins)
    if rolls is None:
        rolls = roll_fn(*roll_range, num=roll_bins)
    if xs is None:
        xs = x_fn(*x_range, num=x_bins)
    if ys is None:
        ys = y_fn(*y_range, num=y_bins)
    if zs is None:
        zs = z_fn(*z_range, num=z_bins)

    # Choose randomly
    yaws = np.random.choice(yaws, count)
    pitchs = np.random.choice(pitchs, count)
    rolls = np.random.choice(rolls, count)
    xs = np.random.choice(xs, count)
    ys = np.random.choice(ys, count)
    zs = np.random.choice(zs, count)

    # Get transforms for each options
    transforms = []

    for yaw, pitch, roll, x, y, z in zip(yaws, pitchs, rolls, xs, ys, zs):
        #transform = get_transform(origin, x_shift=x, y_shift=y, im_scale=z, rot_in_degrees=roll)
        transform = get_transform3d(origin, x_shift=x, y_shift=y, im_scale=z, yaw=yaw, pitch=pitch, roll=roll)
        transforms.append(transform)

    transforms = np.array(transforms).astype(np.float32)

    return transforms

# From: https://github.com/tensorflow/cleverhans/blob/master/examples/adversarial_patch/AdversarialPatch.ipynb
def get_transform(origin, x_shift, y_shift, im_scale, rot_in_degrees):
    """
    If one row of transforms is [a0, a1, a2, b0, b1, b2, c0, c1],
    then it maps the output point (x, y) to a transformed input point
    (x', y') = ((a0 x + a1 y + a2) / k, (b0 x + b1 y + b2) / k),
    where k = c0 x + c1 y + 1.
    The transforms are inverted compared to the transform mapping input points to output points.
    """
    rot = float(rot_in_degrees) / 90. * (math.pi/2)

    # Standard rotation matrix
    # (use negative rot because tf.contrib.image.transform will do the inverse)
    rot_matrix = np.array(
        [[math.cos(-rot), -math.sin(-rot)],
        [math.sin(-rot), math.cos(-rot)]]
    )

    # Scale it
    # (use inverse scale because tf.contrib.image.transform will do the inverse)
    inv_scale = 1. / im_scale
    xform_matrix = rot_matrix * inv_scale
    a0, a1 = xform_matrix[0]
    b0, b1 = xform_matrix[1]

    # At this point, the image will have been rotated around the top left corner,
    # rather than around the center of the image.
    #
    # To fix this, we will see where the center of the image got sent by our transform,
    # and then undo that as part of the translation we apply.
    #x_origin = float(width) / 2
    #y_origin = float(width) / 2
    x_origin, y_origin = origin

    x_origin_shifted, y_origin_shifted = np.matmul(
        xform_matrix,
        np.array([x_origin, y_origin]),
    )

    x_origin_delta = x_origin - x_origin_shifted
    y_origin_delta = y_origin - y_origin_shifted

    # Combine our desired shifts with the rotation-induced undesirable shift
    a2 = x_origin_delta - (x_shift/(2*im_scale))
    b2 = y_origin_delta - (y_shift/(2*im_scale))

    # Return these values in the order that tf.contrib.image.transform expects
    return np.array([a0, a1, a2, b0, b1, b2, 0, 0]).astype(np.float32)

def get_M(width, height, focal, theta, phi, gamma, dx, dy, dz):

        w = width
        h = height
        f = focal

        # Projection 2D -> 3D matrix
        A1 = np.array([ [1, 0, -w/2],
                        [0, 1, -h/2],
                        [0, 0, 1],
                        [0, 0, 1]])

        # Rotation matrices around the X, Y, and Z axis
        RX = np.array([ [1, 0, 0, 0],
                        [0, np.cos(theta), -np.sin(theta), 0],
                        [0, np.sin(theta), np.cos(theta), 0],
                        [0, 0, 0, 1]])

        RY = np.array([ [np.cos(phi), 0, -np.sin(phi), 0],
                        [0, 1, 0, 0],
                        [np.sin(phi), 0, np.cos(phi), 0],
                        [0, 0, 0, 1]])

        RZ = np.array([ [np.cos(gamma), -np.sin(gamma), 0, 0],
                        [np.sin(gamma), np.cos(gamma), 0, 0],
                        [0, 0, 1, 0],
                        [0, 0, 0, 1]])

        # Composed rotation matrix with (RX, RY, RZ)
        R = np.dot(np.dot(RX, RY), RZ)

        # Translation matrix
        T = np.array([  [1, 0, 0, dx],
                        [0, 1, 0, dy],
                        [0, 0, 1, dz],
                        [0, 0, 0, 1]])

        # Projection 3D -> 2D matrix
        A2 = np.array([ [f, 0, w/2, 0],
                        [0, f, h/2, 0],
                        [0, 0, 1, 0]])

        # Final transformation matrix
        return np.dot(A2, np.dot(T, np.dot(R, A1)))

def get_transform3d(origin, x_shift, y_shift, im_scale, yaw, pitch, roll):
    theta = float(yaw) * math.pi / 180.0
    phi = float(pitch) * math.pi / 180.0
    gamma = float(roll) * math.pi / 180.0

    x_origin, y_origin = origin
    d = np.sqrt((x_origin*2)**2 + (y_origin*2)**2)
    focal = d / (2 * np.sin(gamma) if np.sin(gamma) != 0 else 1)
    mat = get_M(x_origin * 2, y_origin * 2, focal, theta, phi, gamma, x_shift, y_shift, focal)
    mat = np.dot(mat, [[im_scale, 0, 0], [0, im_scale, 0], [0, 0, 1]])
    inv_mat = np.linalg.inv(mat)

    return (inv_mat.reshape(9, ) / inv_mat[2, 2])[:8]


def create_attack_2(textures, textures_var, predictions, losses, yolo_losses, optimizer_name='gd', clip=False):
    # Box Classsifier Loss
    box_cls_weight_ = tf.placeholder_with_default(1.0, [], name='box_cls_weight')
    box_cls_loss_ = losses['Loss/BoxClassifierLoss/classification_loss']
    weighted_box_cls_loss_ = tf.multiply(box_cls_loss_, box_cls_weight_, name='box_cls_loss')

    # Box Localizer Loss
    box_loc_weight_ = tf.placeholder_with_default(2.0, [], name='box_loc_weight')
    box_loc_loss_ = losses['Loss/BoxClassifierLoss/localization_loss']
    weighted_box_loc_loss_ = tf.multiply(box_loc_loss_, box_loc_weight_, name='box_loc_loss')

    # RPN Classifier Loss
    rpn_cls_weight_ = tf.placeholder_with_default(1.0, [], name='rpn_cls_weight')
    rpn_cls_loss_ = losses['Loss/RPNLoss/objectness_loss']
    weighted_rpn_cls_loss_ = tf.multiply(rpn_cls_loss_, rpn_cls_weight_, name='rpn_cls_loss')

    # RPN Localizer Loss
    rpn_loc_weight_ = tf.placeholder_with_default(2.0, [], name='rpn_loc_weight')
    rpn_loc_loss_ = losses['Loss/RPNLoss/localization_loss']
    weighted_rpn_loc_loss_ = tf.multiply(rpn_loc_loss_, rpn_loc_weight_, name='rpn_loc_loss')

    # Box Losses
    target_class_ = tf.placeholder(tf.int32, [], name='target_class')
    victim_class_ = tf.placeholder(tf.int32, [], name='victim_class')
    box_iou_thresh_ = tf.placeholder_with_default(0.5, [], name='box_iou_thresh')

    box_logits_ = predictions['class_predictions_with_background']
    box_logits_ = box_logits_[:, 1:] # Ignore background class
    victim_one_hot_ = tf.one_hot([victim_class_ - 1], box_logits_.shape[-1])
    target_one_hot_ = tf.one_hot([target_class_ - 1], box_logits_.shape[-1])
    box_iou_ = tf.get_default_graph().get_tensor_by_name('Loss/BoxClassifierLoss/Compare/IOU/Select:0')
    box_iou_ = tf.reshape(box_iou_, (-1,))
    box_targets_ = tf.cast(box_iou_ >= box_iou_thresh_, tf.float32)

    # Box Victim Loss
    box_victim_weight_ = tf.placeholder_with_default(0.0, [], name='box_victim_weight')

    box_victim_logits_ = box_logits_[:, victim_class_ - 1]
    box_victim_loss_ = box_victim_logits_*box_targets_
    box_victim_loss_ = tf.reduce_sum(box_victim_loss_)
    weighted_box_victim_loss_ = tf.multiply(box_victim_loss_, box_victim_weight_, name='box_victim_loss')

    # Box Target Loss
    box_target_weight_ = tf.placeholder_with_default(0.0, [], name='box_target_weight')

    box_target_logits_ = box_logits_[:, target_class_ - 1]
    box_target_loss_ = box_target_logits_*box_targets_
    box_target_loss_ = -1*tf.reduce_sum(box_target_loss_) # Maximize!
    weighted_box_target_loss_ = tf.multiply(box_target_loss_, box_target_weight_, name='box_target_loss')

    # Box Victim CW loss (untargeted, victim -> nonvictim)
    box_victim_cw_weight_ = tf.placeholder_with_default(0.0, [], name='box_victim_cw_weight')
    box_victim_cw_conf_ = tf.placeholder_with_default(0.0, [], name='box_victim_cw_conf')

    box_nonvictim_logits_ = tf.reduce_max(box_logits_ * (1 - victim_one_hot_) - 10000*victim_one_hot_, axis=-1)
    box_victim_cw_loss_ = tf.nn.relu(box_victim_logits_ - box_nonvictim_logits_ + box_victim_cw_conf_)
    box_victim_cw_loss_ = box_victim_cw_loss_*box_targets_
    box_victim_cw_loss_ = tf.reduce_sum(box_victim_cw_loss_)
    weighted_box_victim_cw_loss_ = tf.multiply(box_victim_cw_loss_, box_victim_cw_weight_, name='box_victim_cw_loss')

    # Box Target CW loss (targeted, nontarget -> target)
    box_target_cw_weight_ = tf.placeholder_with_default(0.0, [], name='box_target_cw_weight')
    box_target_cw_conf_ = tf.placeholder_with_default(0.0, [], name='box_target_cw_conf')

    box_nontarget_logits_ = tf.reduce_max(box_logits_ * (1 - target_one_hot_) - 10000*target_one_hot_, axis=-1)
    box_target_cw_loss_ = tf.nn.relu(box_nontarget_logits_ - box_target_logits_ + box_target_cw_conf_)
    box_target_cw_loss_ = box_target_cw_loss_*box_targets_
    box_target_cw_loss_ = tf.reduce_sum(box_target_cw_loss_)
    weighted_box_target_cw_loss_ = tf.multiply(box_target_cw_loss_, box_target_cw_weight_, name='box_target_cw_loss')

    # RPN Losses
    rpn_iou_thresh_ = tf.placeholder_with_default(0.5, [], name='rpn_iou_thresh')
    rpn_logits_ = predictions['rpn_objectness_predictions_with_background']
    rpn_logits_ = tf.reshape(rpn_logits_, (-1, rpn_logits_.shape[-1]))
    rpn_iou_ = tf.get_default_graph().get_tensor_by_name('Loss/RPNLoss/Compare/IOU/Select:0')
    rpn_iou_ = tf.reshape(rpn_iou_, (-1,))
    rpn_targets_ = tf.cast(rpn_iou_ >= rpn_iou_thresh_, tf.float32)

    # RPN Background Loss
    rpn_background_weight_ = tf.placeholder_with_default(0.0, [], name='rpn_background_weight')

    rpn_background_logits_ = rpn_logits_[:, 0]
    rpn_background_loss_ = rpn_background_logits_*rpn_targets_
    rpn_background_loss_ = -1*tf.reduce_sum(rpn_background_loss_) # Maximize!
    weighted_rpn_background_loss_ = tf.multiply(rpn_background_loss_, rpn_background_weight_, name='rpn_background_loss')

    # RPN Foreground Loss
    rpn_foreground_weight_ = tf.placeholder_with_default(0.0, [], name='rpn_foreground_weight')

    rpn_foreground_logits_ = rpn_logits_[:, 1]
    rpn_foreground_loss_ = rpn_foreground_logits_*rpn_targets_
    rpn_foreground_loss_ = tf.reduce_sum(rpn_foreground_loss_)
    weighted_rpn_foreground_loss_ = tf.multiply(rpn_foreground_loss_, rpn_foreground_weight_, name='rpn_foreground_loss')

    # RPN CW Loss (un/targeted foreground -> background)
    rpn_cw_weight_ = tf.placeholder_with_default(0.0, [], name='rpn_cw_weight')
    rpn_cw_conf_ = tf.placeholder_with_default(0.0, [], name='rpn_cw_conf')

    rpn_cw_loss_ = tf.nn.relu(rpn_foreground_logits_ - rpn_background_logits_ + rpn_cw_conf_)
    rpn_cw_loss_ = rpn_cw_loss_*rpn_targets_
    rpn_cw_loss_ = tf.reduce_sum(rpn_cw_loss_)
    weighted_rpn_cw_loss_ = tf.multiply(rpn_cw_loss_, rpn_cw_weight_, name='rpn_cw_loss')

    # Similiary Loss
    sim_weight_ = tf.placeholder_with_default(0.0, [], name='sim_weight')
    initial_textures_ = tf.get_default_graph().get_tensor_by_name('initial_textures:0')
    sim_loss_ = tf.nn.l2_loss(initial_textures_ - textures)
    weighted_sim_loss_ = tf.multiply(sim_loss_, sim_weight_, name='sim_loss')

    # YOLO Loss
    yolo_weight_ = tf.placeholder_with_default(0.0, [], name='yolo_weight')
    yolo_loss_ = yolo_losses[0]
    weighted_yolo_loss_ = tf.multiply(yolo_loss_, yolo_weight_, name='yolo_loss')

    loss_ = tf.add_n([weighted_box_cls_loss_, weighted_box_loc_loss_,
                      weighted_rpn_cls_loss_, weighted_rpn_loc_loss_,
                      weighted_box_victim_loss_, weighted_box_target_loss_,
                      weighted_box_victim_cw_loss_, weighted_box_target_cw_loss_,
                      weighted_rpn_foreground_loss_, weighted_rpn_background_loss_,
                      weighted_rpn_cw_loss_,
                      weighted_sim_loss_,
                      weighted_yolo_loss_], name='loss')

    # Support large batch accumulation metrics
    total_box_cls_loss_, update_box_cls_loss_ = tf.metrics.mean(losses['Loss/BoxClassifierLoss/classification_loss'])
    total_box_loc_loss_, update_box_loc_loss_ = tf.metrics.mean(losses['Loss/BoxClassifierLoss/localization_loss'])
    total_rpn_cls_loss_, update_rpn_cls_loss_ = tf.metrics.mean(losses['Loss/RPNLoss/objectness_loss'])
    total_rpn_loc_loss_, update_rpn_loc_loss_ = tf.metrics.mean(losses['Loss/RPNLoss/localization_loss'])
    total_box_target_loss_, update_box_target_loss_ = tf.metrics.mean(box_target_loss_)
    total_box_victim_loss_, update_box_victim_loss_ = tf.metrics.mean(box_victim_loss_)
    total_box_target_cw_loss_, update_box_target_cw_loss_ = tf.metrics.mean(box_target_cw_loss_)
    total_box_victim_cw_loss_, update_box_victim_cw_loss_ = tf.metrics.mean(box_victim_cw_loss_)
    total_rpn_foreground_loss_, update_rpn_foreground_loss_ = tf.metrics.mean(rpn_foreground_loss_)
    total_rpn_background_loss_, update_rpn_background_loss_ = tf.metrics.mean(rpn_background_loss_)
    total_rpn_cw_loss_, update_rpn_cw_loss_ = tf.metrics.mean(rpn_cw_loss_)
    total_sim_loss_, update_sim_loss_ = tf.metrics.mean(sim_loss_)
    total_yolo_loss_, update_yolo_loss_ = tf.metrics.mean(yolo_loss_)

    total_weighted_box_cls_loss_ = tf.multiply(total_box_cls_loss_, box_cls_weight_, name='total_box_cls_loss')
    total_weighted_box_loc_loss_ = tf.multiply(total_box_loc_loss_, box_loc_weight_, name='total_box_loc_loss')
    total_weighted_rpn_cls_loss_ = tf.multiply(total_rpn_cls_loss_, rpn_cls_weight_, name='total_rpn_cls_loss')
    total_weighted_rpn_loc_loss_ = tf.multiply(total_rpn_loc_loss_, rpn_loc_weight_, name='total_rpn_loc_loss')
    total_weighted_box_target_loss_ = tf.multiply(total_box_target_loss_, box_target_weight_, name='total_box_target_loss')
    total_weighted_box_victim_loss_ = tf.multiply(total_box_victim_loss_, box_victim_weight_, name='total_box_victim_loss')
    total_weighted_box_target_cw_loss_ = tf.multiply(total_box_target_cw_loss_, box_target_cw_weight_, name='total_box_target_cw_loss')
    total_weighted_box_victim_cw_loss_ = tf.multiply(total_box_victim_cw_loss_, box_victim_cw_weight_, name='total_box_victim_cw_loss')
    total_weighted_rpn_background_loss_ = tf.multiply(total_rpn_background_loss_, rpn_background_weight_, name='total_rpn_background_loss')
    total_weighted_rpn_foreground_loss_ = tf.multiply(total_rpn_foreground_loss_, rpn_foreground_weight_, name='total_rpn_foreground_loss')
    total_weighted_rpn_cw_loss_ = tf.multiply(total_rpn_cw_loss_, rpn_cw_weight_, name='total_rpn_cw_loss')
    total_weighted_sim_loss_ = tf.multiply(total_sim_loss_, sim_weight_, name='total_sim_loss')
    total_weighted_yolo_loss_ = tf.multiply(total_yolo_loss_, yolo_weight_, name='total_yolo_loss')

    total_loss_ = tf.add_n([total_weighted_box_cls_loss_, total_weighted_box_loc_loss_,
                            total_weighted_rpn_cls_loss_, total_weighted_rpn_loc_loss_,
                            total_weighted_box_target_loss_, total_weighted_box_victim_loss_,
                            total_weighted_box_target_cw_loss_, total_weighted_box_victim_cw_loss_,
                            total_weighted_rpn_foreground_loss_, total_weighted_rpn_background_loss_,
                            total_weighted_rpn_cw_loss_,
                            total_weighted_sim_loss_,
                            total_weighted_yolo_loss_], name='total_loss')

    #tf.summary.scalar('losses/box_cls_weight', box_cls_weight_)
    tf.summary.scalar('losses/box_cls_loss', total_box_cls_loss_)
    tf.summary.scalar('losses/box_cls_weighted_loss', total_weighted_box_cls_loss_)

    #tf.summary.scalar('losses/box_loc_weight', box_loc_weight_)
    tf.summary.scalar('losses/box_loc_loss', total_box_loc_loss_)
    tf.summary.scalar('losses/box_loc_weighted_loss', total_weighted_box_loc_loss_)

    #tf.summary.scalar('losses/rpn_cls_weight', rpn_cls_weight_)
    tf.summary.scalar('losses/rpn_cls_loss', total_rpn_cls_loss_)
    tf.summary.scalar('losses/rpn_cls_weighted_loss', total_weighted_rpn_cls_loss_)

    #tf.summary.scalar('losses/rpn_loc_weight', rpn_loc_weight_)
    tf.summary.scalar('losses/rpn_loc_loss', total_rpn_loc_loss_)
    tf.summary.scalar('losses/rpn_loc_weighted_loss', total_weighted_rpn_loc_loss_)

    #tf.summary.scalar('losses/box_target_weight', box_target_weight_)
    tf.summary.scalar('losses/box_target_loss', total_box_target_loss_)
    tf.summary.scalar('losses/box_target_weighted_loss', total_weighted_box_target_loss_)

    #tf.summary.scalar('losses/box_victim_weight', box_victim_weight_)
    tf.summary.scalar('losses/box_victim_loss', total_box_victim_loss_)
    tf.summary.scalar('losses/box_victim_weighted_loss', total_weighted_box_victim_loss_)

    #tf.summary.scalar('losses/box_target_cw_weight', box_target_cw_weight_)
    #tf.summary.scalar('losses/box_target_cw_conf', box_target_cw_conf_)
    tf.summary.scalar('losses/box_target_cw_loss', total_box_target_cw_loss_)
    tf.summary.scalar('losses/box_target_cw_weighted_loss', total_weighted_box_target_cw_loss_)

    #tf.summary.scalar('losses/box_victim_cw_weight', box_victim_cw_weight_)
    #tf.summary.scalar('losses/box_victim_cw_conf', box_victim_cw_conf_)
    tf.summary.scalar('losses/box_victim_cw_loss', total_box_victim_cw_loss_)
    tf.summary.scalar('losses/box_victim_cw_weighted_loss', total_weighted_box_victim_cw_loss_)

    #tf.summary.scalar('losses/rpn_foreground_weight', rpn_foreground_weight_)
    tf.summary.scalar('losses/rpn_foreground_loss', total_rpn_foreground_loss_)
    tf.summary.scalar('losses/rpn_foreground_weighted_loss', total_weighted_rpn_foreground_loss_)

    #tf.summary.scalar('losses/rpn_background_weight', rpn_background_weight_)
    tf.summary.scalar('losses/rpn_background_loss', total_rpn_background_loss_)
    tf.summary.scalar('losses/rpn_background_weighted_loss', total_weighted_rpn_background_loss_)

    #tf.summary.scalar('losses/rpn_cw_weight', rpn_cw_weight_)
    #tf.summary.scalar('losses/rpn_cw_conf', rpn_cw_conf_)
    tf.summary.scalar('losses/rpn_cw_loss', total_rpn_cw_loss_)
    tf.summary.scalar('losses/rpn_cw_weighted_loss', total_weighted_rpn_cw_loss_)

    tf.summary.scalar('losses/sim_loss', total_sim_loss_)
    tf.summary.scalar('losses/sim_weighted_loss', total_weighted_sim_loss_)

    tf.summary.scalar('losses/yolo_loss', total_yolo_loss_)
    tf.summary.scalar('losses/yolo_weighted_loss', total_weighted_yolo_loss_)

    tf.summary.scalar('loss/loss', total_loss_)

    learning_rate_ = tf.placeholder(tf.float32, [], name='learning_rate')
    #tf.summary.scalar('hyperparameters/learning_rate', learning_rate_)

    momentum_ = tf.placeholder(tf.float32, [], name='momentum')
    #tf.summary.scalar('hyperparameters/momentum', momentum_)

    decay_ = tf.placeholder(tf.float32, [], name='decay')
    #tf.summary.scalar('hyperparameters/decay', decay_)

    optimizer = None
    if optimizer_name == 'gd':
        optimizer = tf.train.GradientDescentOptimizer(learning_rate_)
    elif optimizer_name == 'momentum':
        optimizer = tf.train.MomentumOptimizer(learning_rate_, momentum=momentum_)
    elif optimizer_name == 'rmsprop':
        optimizer = tf.train.RMSPropOptimizer(learning_rate_, momentum=momentum_, decay=decay_)
    elif optimizer_name == 'adam':
        optimizer = tf.train.AdamOptimizer(learning_rate_)
    else:
        raise Exception("Unknown optimizer: {args.optimizer}")

    global_step_ = tf.train.get_or_create_global_step()
    grad_ = optimizer.compute_gradients(loss_, var_list=[textures_var])[0][0]
    grad_ = grad_

    # Create variable to store grad
    grad_total_ = tf.Variable(tf.zeros_like(textures_var), trainable=False, name='grad_total', collections=[tf.GraphKeys.GLOBAL_VARIABLES, tf.GraphKeys.LOCAL_VARIABLES])
    grad_count_ = tf.Variable(0., trainable=False, name='grad_count', collections=[tf.GraphKeys.GLOBAL_VARIABLES, tf.GraphKeys.LOCAL_VARIABLES])

    update_grad_total_ = tf.assign_add(grad_total_, grad_)
    update_grad_count_ = tf.assign_add(grad_count_, 1.)

    grad_ = tf.div(grad_total_, tf.maximum(grad_count_, 1.), name='grad')
    if clip:
        grad_ = tf.sign(grad_)

    accum_grad_op_ = tf.group([update_grad_total_, update_grad_count_], name='accum_op')

    grad_vec_ = tf.reshape(grad_, [-1], name='grad_vec')
    tf.summary.histogram('gradients/all', grad_vec_)

    grad_l1_ = tf.identity(tf.norm(grad_vec_, ord=1), name='grad_l1')
    tf.summary.scalar('gradients/all_l1', grad_l1_)

    grad_l2_ = tf.identity(tf.norm(grad_vec_, ord=2), name='grad_l2')
    tf.summary.scalar('gradients/all_l2', grad_l2_)

    grad_linf_ = tf.identity(tf.norm(grad_vec_, ord=np.inf), name='grad_linf')
    tf.summary.scalar('gradients/all_linf', grad_linf_)

    attack_op_ = optimizer.apply_gradients([(grad_, textures_var)], global_step=global_step_, name='attack_op')

    update_losses_ = tf.stack([update_box_cls_loss_, update_box_loc_loss_,
                               update_rpn_cls_loss_, update_rpn_loc_loss_,
                               update_box_target_loss_, update_box_victim_loss_,
                               update_box_target_cw_loss_, update_box_victim_cw_loss_,
                               update_rpn_foreground_loss_, update_rpn_background_loss_,
                               update_rpn_cw_loss_,
                               update_sim_loss_,
                               update_yolo_loss_],
                              name='update_losses_op')

    losses_summary_ = tf.summary.merge_all()

    return victim_class_, target_class_, losses_summary_

if __name__ == '__main__':
    main()
