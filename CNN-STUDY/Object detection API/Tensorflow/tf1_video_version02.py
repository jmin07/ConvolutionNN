# conda activate tf1.15.5
import numpy as np
import os
import sys
import tarfile
import tensorflow as tf
import zipfile
import cv2
from tqdm import tqdm

from matplotlib import pyplot as plt
from PIL import Image

# 아래는 tensorflow version 1.13의 utils에서 가져왔습니다.
from utils_tf1 import ops as utils_ops
from utils_tf1 import label_map_util
from utils_tf1 import visualization_utils as vis_util

def run_inference_for_single_image(sess, image):
    # Get handles to input and output tensors
    ops = tf.get_default_graph().get_operations()
    all_tensor_names = {output.name for op in ops for output in op.outputs}

    tensor_dict = {}
    for key in [
        'num_detections', 'detection_boxes', 'detection_scores',
        'detection_classes', 'detection_masks'
    ]:
        tensor_name = key + ':0'
        if tensor_name in all_tensor_names:
            tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(tensor_name)

    if 'detection_masks' in tensor_dict:
        # The following processing is only for single image
        detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
        detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])

        # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
        real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
        detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
        detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
        detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(detection_masks, detection_boxes,
                                                                              image.shape[0], image.shape[1])
        detection_masks_reframed = tf.cast(tf.greater(detection_masks_reframed, 0.5), tf.uint8)

        # Follow the convention by adding back the batch dimension
        tensor_dict['detection_masks'] = tf.expand_dims(detection_masks_reframed, 0)

    image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

    # Run inference
    output_dict = sess.run(tensor_dict, feed_dict={image_tensor: np.expand_dims(image, 0)})

    # all outputs are float32 numpy arrays, so convert types as appropriate
    output_dict['num_detections'] = int(output_dict['num_detections'][0])
    output_dict['detection_classes'] = output_dict['detection_classes'][0].astype(np.uint8)
    output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
    output_dict['detection_scores'] = output_dict['detection_scores'][0]

    if 'detection_masks' in output_dict:
        output_dict['detection_masks'] = output_dict['detection_masks'][0]

    return output_dict


def run_tf1_video(model_name, video_save, draw_image_save, ori_image_save, anno_save, sess):

    model = './' + model_name
    MODEL = model + '/frozen_inference_graph.pb'
    PATH_TO_LABELS = model + '/***.pbtxt'

    # 경로 고정
    PATH = 'C:/Users/***/PycharmProjects/pythonProject/models/ssd_model'
    DATA_FILE_PATH = 'inference/Data/Origin_data/video'
    SAVE_PATH = './inference/Data/Result_data/image/'

    category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)

    for file_name in os.listdir(DATA_FILE_PATH):
        if file_name.endswith('.avi') or file_name.endswith('.mp4'):

            if not os.path.exists(os.path.join(SAVE_PATH, file_name, model_name)):
                if not os.path.exists(os.path.join(SAVE_PATH, file_name)):
                    os.mkdir(os.path.join(SAVE_PATH, file_name))
                    os.mkdir(os.path.join(SAVE_PATH, file_name, model_name))
                else:
                    os.mkdir(os.path.join(SAVE_PATH, file_name, model_name))

            print(f"---> file name : {file_name}")

            video_cap = cv2.VideoCapture(os.path.join(DATA_FILE_PATH, file_name))
            frame_size = (
                           int(video_cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                           int(video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            )
            print(f'---> frame_size : {frame_size}')

            # 코덱 설정
            FOURCC = cv2.VideoWriter_fourcc(*'DIVX')
            # FOURCC = cv2.VideoWriter_fourcc(*'XVID')

            FPS = 25.0
            threshold = 0.5

            # 이미지 저장하기 위한 영상 파일 생성
            video_out = cv2.VideoWriter(SAVE_PATH + file_name + '/' + model_name + '/{}.mp4'.format(model_name), FOURCC, FPS, frame_size)

            total_frame_count = video_cap.get(cv2.CAP_PROP_FRAME_COUNT)
            print("전체 프레임 수: {}".format(total_frame_count))

            obj_counts = 0
            total_score = 0
            for count in tqdm(range(int(total_frame_count))):  # frame_count가 float형태이다
                retval, frame = video_cap.read()

                if not retval:
                    break

                # Actual detection.
                output_dict = run_inference_for_single_image(sess, frame)

                detection_scores = output_dict['detection_scores']
                detection_boxes = output_dict['detection_boxes']
                detection_classes = output_dict['detection_classes']
                im_height, im_width = frame.shape[0], frame.shape[1]

                if ori_image_save == 'yes':
                    if not os.path.exists(os.path.join(SAVE_PATH, file_name, model_name, 'Origin_image')):
                        os.mkdir(os.path.join(SAVE_PATH, file_name, model_name, 'Origin_image'))

                    cv2.imwrite(SAVE_PATH + file_name + '/' + model_name + '/Origin_image' + '/{}_{}.jpg'.format(file_name, '{0:010d}'.format(count)), frame)

                if draw_image_save == 'yes':
                    if not os.path.exists(os.path.join(SAVE_PATH, file_name, model_name, 'Draw_image')):
                        os.mkdir(os.path.join(SAVE_PATH, file_name, model_name, 'Draw_image'))

                    # Visualization of the results of a detection.
                    vis_util.visualize_boxes_and_labels_on_image_array(
                        frame,
                        output_dict['detection_boxes'],
                        output_dict['detection_classes'],
                        output_dict['detection_scores'],
                        category_index,
                        instance_masks=output_dict.get('detection_masks'),
                        use_normalized_coordinates=True,
                        line_thickness=3,
                        min_score_thresh=threshold
                    )

                    if anno_save == 'yes' and len(detection_classes) != 0:
                        for idx, score in list(filter(lambda val: val[1] > threshold, [val for val in enumerate(detection_scores)])):
                            if not os.path.exists(SAVE_PATH + file_name + '/' + model_name + '/Draw_image' + '/{}_{}.txt'.format(file_name, '{0:010d}'.format(count))):
                                with open(SAVE_PATH + file_name + '/' + model_name + '/Draw_image' + '/{}_{}.txt'.format(file_name, '{0:010d}'.format(count)), mode='w') as wf:
                                    (left, right, top, bottom) = (
                                        detection_boxes[idx][1] * im_width, detection_boxes[idx][3] * im_width,
                                        detection_boxes[idx][0] * im_height, detection_boxes[idx][2] * im_height
                                    )
                                    dec_class = detection_classes[idx]
                                    dec_score = detection_scores[idx]

                                    wf.write('{} {} {} {} {} {} {} {}\n'.format(dec_class, int(left), int(top), int(right), int(bottom), -1, -1, -1))

                                    obj_counts += 1
                                    total_score += dec_score
                            else:
                                with open(SAVE_PATH + file_name + '/' + model_name + '/Draw_image' + '/{}_{}.txt'.format(file_name, '{0:010d}'.format(count)), mode='a') as af:
                                    (left, right, top, bottom) = (
                                        detection_boxes[idx][1] * im_width, detection_boxes[idx][3] * im_width,
                                        detection_boxes[idx][0] * im_height, detection_boxes[idx][2] * im_height
                                    )
                                    dec_class = detection_classes[idx]
                                    dec_score = detection_scores[idx]

                                    af.write('{} {} {} {} {} {} {} {}\n'.format(dec_class, int(left), int(top), int(right), int(bottom), -1, -1, -1))

                                    obj_counts += 1
                                    total_score += dec_score

                    if obj_counts == 0:
                        mean_score = round(0, 4)
                    else:
                        mean_score = round((total_score / obj_counts), 4) * 100

                    text_frame = cv2.putText(frame, 'OBJECT: {}'.format(obj_counts), (frame_size[0] - 260, 80),  cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0))
                    text_frame = cv2.putText(text_frame, 'SCORE: {}%'.format(mean_score), (frame_size[0] - 260, 110), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0))

                    if video_save == 'yes':
                        video_out.write(text_frame)

                    cv2.imwrite(SAVE_PATH + file_name + '/' + model_name + '/Draw_image' + '/{}_{}.jpg'.format(file_name, '{0:010d}'.format(count)), text_frame)

def main():
    video_save = 'yes'
    draw_image_save = 'yes'
    ori_image_save = 'no'
    anno_save = 'yes'
    model_names = ['model_20728', 'model_19058', 'model_17085']

    PATH = 'C:/Users/****/PycharmProjects/pythonProject/models/ssd_model'

    if os.getcwd() != PATH:
        os.chdir(PATH)

    for model_name in model_names:
        model = './' + model_name
        MODEL = model + '/frozen_inference_graph.pb'

        # Load a (frozen) Tensorflow model into memory.
        detection_graph = tf.Graph()
        with detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.io.gfile.GFile(MODEL, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')

        with detection_graph.as_default():
            with tf.Session() as sess:
                run_tf1_video(model_name, video_save, draw_image_save, ori_image_save, anno_save, sess)


if __name__ == '__main__':
    main()