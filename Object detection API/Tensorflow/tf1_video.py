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

"""
사용버전 : Tensorflow object detection tf1.13 release tutorial 
사이트: https://github.com/tensorflow/models/blob/57e075203f8fba8d85e6b74f17f63d0a07da233a/research/object_detection/object_detection_tutorial.ipynb

"""


def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size

    return np.array(image.getdata()).reshape((im_height, im_width, 3)).astype(np.uint8)


def run_inference_for_single_image(image, graph):
    with graph.as_default():
        with tf.Session() as sess:
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
                    tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(
                        tensor_name)
            if 'detection_masks' in tensor_dict:
                # The following processing is only for single image
                detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
                detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
                # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
                real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
                detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
                detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
                detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
                    detection_masks, detection_boxes, image.shape[0], image.shape[1])
                detection_masks_reframed = tf.cast(
                    tf.greater(detection_masks_reframed, 0.5), tf.uint8)
                # Follow the convention by adding back the batch dimension
                tensor_dict['detection_masks'] = tf.expand_dims(
                    detection_masks_reframed, 0)
            image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

            # Run inference
            output_dict = sess.run(tensor_dict, feed_dict={image_tensor: np.expand_dims(image, 0)})

            # all outputs are float32 numpy arrays, so convert types as appropriate
            output_dict['num_detections'] = int(output_dict['num_detections'][0])
            output_dict['detection_classes'] = output_dict[
                'detection_classes'][0].astype(np.uint8)
            output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
            output_dict['detection_scores'] = output_dict['detection_scores'][0]
            if 'detection_masks' in output_dict:
                output_dict['detection_masks'] = output_dict['detection_masks'][0]

    return output_dict


def run_tf1_video(model_name, video_save, draw_image_save, ori_image_save, anno_save):
    model = './' + model_name
    MODEL = model + '/frozen_inference_graph.pb'
    PATH_TO_LABELS = model + '/****.pbtxt'

    # 경로 고정
    PATH = 'C:/Users/****/PycharmProjects/pythonProject/models/ssd_model'
    DATA_FILE_PATH = 'inference/Data/Origin_data/video'
    SAVE_PATH = './inference/Data/Result_data/image/draw_image/video_image/'

    if os.getcwd() != PATH:
        os.chdir(PATH)

    if not os.path.exists(os.path.join(SAVE_PATH, model_name)):
        os.mkdir(os.path.join(SAVE_PATH, model_name))

    # Load a (frozen) Tensorflow model into memory.
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(MODEL, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')

    category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)

    ###### 동영상 이미지 가져오기
    for file_name in os.listdir(DATA_FILE_PATH):
        if '.mp4' in file_name:
            print(file_name)
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
            video_out = cv2.VideoWriter('./inference/Data/Result_data/video/{}.mp4'.format(file_name.split(".mp4")[0]),
                                        FOURCC, FPS, frame_size)

            total_frame_count = video_cap.get(cv2.CAP_PROP_FRAME_COUNT)
            print("전체 프레임 수: {}".format(total_frame_count))

            obj_counts = 0
            total_score = 0
            for count in tqdm(range(int(total_frame_count))):  # frame_count가 float형태이다
                retval, frame = video_cap.read()
                if not retval:
                    break

                # Actual detection.
                output_dict = run_inference_for_single_image(frame, detection_graph)

                detection_scores = output_dict['detection_scores']
                detection_boxes = output_dict['detection_boxes']
                detection_classes = output_dict['detection_classes']
                im_height, im_width = frame.shape[0], frame.shape[1]

                if ori_image_save == 'yes':
                    cv2.imwrite('./inference/Data/Result_data/image/video_image/{}_{}.jpg'.format(file_name,
                                                                                                  '{0:010d}'.format(
                                                                                                      count)), frame)

                if draw_image_save == 'yes':

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
                        for idx, score in list(
                                filter(lambda val: val[1] > threshold, [val for val in enumerate(detection_scores)])):
                            if not os.path.exists(
                                    './inference/Data/Result_data/image/video_image/{}_{}.txt'.format(file_name,
                                                                                                      '{0:010d}'.format(
                                                                                                              count))):
                                with open(
                                        './inference/Data/Result_data/image/draw_image/video_image/' + model_name + '/{}_{}.txt'.format(
                                                file_name, '{0:010d}'.format(count)), mode='w') as file:
                                    (left, right, top, bottom) = (
                                        detection_boxes[idx][1] * im_width, detection_boxes[idx][3] * im_width,
                                        detection_boxes[idx][0] * im_height, detection_boxes[idx][2] * im_height
                                    )
                                    dec_class = detection_classes[idx]
                                    dec_score = detection_scores[idx]

                                    file.write(
                                        '{} {} {} {} {} {} {} {}\n'.format(dec_class, int(left), int(top), int(right),
                                                                           int(bottom), -1, -1, -1))
                                    obj_counts += 1
                                    total_score += dec_score

                            else:
                                with open('./inference/Data/Result_data/image/video_image/{}_{}.txt'.format(file_name,
                                                                                                            '{0:010d}'.format(
                                                                                                                    count)),
                                          mode='at') as file:
                                    (left, right, top, bottom) = (
                                        detection_boxes[idx][1] * im_width, detection_boxes[idx][3] * im_width,
                                        detection_boxes[idx][0] * im_height, detection_boxes[idx][2] * im_height
                                    )
                                    dec_class = detection_classes[idx]
                                    dec_score = detection_scores[idx]

                                    file.write(
                                        '{} {} {} {} {} {} {} {}\n'.format(dec_class, int(left), int(top), int(right),
                                                                           int(bottom), -1, -1, -1))
                                    obj_counts += 1
                                    total_score += dec_score

                    if obj_counts == 0:
                        mean_score = round(0, 4)
                    else:
                        mean_score = round((total_score / obj_counts), 4) * 100

                    text_frame = cv2.putText(frame, 'OBJECT: {}'.format(obj_counts), (frame_size[0] - 260, 80),
                                             cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0))
                    text_frame = cv2.putText(text_frame, 'SCORE: {}%'.format(mean_score), (frame_size[0] - 260, 110),
                                             cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0))

                    if video_save == 'yes':
                        video_out.write(text_frame)

                    cv2.imwrite(SAVE_PATH + model_name + '/{}_{}.jpg'.format(file_name, '{0:010d}'.format(count)),
                                text_frame)


def main():
    video_save = 'yes'
    draw_image_save = 'yes'
    ori_image_save = 'yes'
    anno_save = 'yes'
    model_name = 'model_class_320610'

    run_tf1_video(model_name, video_save, draw_image_save, ori_image_save, anno_save)


if __name__ == '__main__':
    main()
