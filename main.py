# -*- coding: utf-8 -*-
"""People Counter."""
"""
 Copyright (c) 2018 Intel Corporation.
 Permission is hereby granted, free of charge, to any person obtaining
 a copy of this software and associated documentation files (the
 "Software"), to deal in the Software without restriction, including
 without limitation the rights to use, copy, modify, merge, publish,
 distribute, sublicense, and/or sell copies of the Software, and to
 permit person to whom the Software is furnished to do so, subject to
 the following conditions:
 The above copyright notice and this permission notice shall be
 included in all copies or substantial portions of the Software.
 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
 LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
 OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
 WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""


import os
import sys
import time
import socket
import subprocess
import json
import cv2
import logging as log
import paho.mqtt.client as mqtt
import numpy as np
from argparse import ArgumentParser
from inference import Network
from csv import DictWriter
from datetime import datetime
from collections import deque
FORMATTER = log.Formatter(
    "%(asctime)s — %(name)s — %(levelname)s — %(message)s")
console_handler = log.StreamHandler(sys.stdout)
console_handler.setFormatter(FORMATTER)
logger = log.getLogger(__name__)
logger.setLevel(log.ERROR)
# logger.setLevel(log.DEBUG)
logger.addHandler(console_handler)

MODEL_PATH = "/opt/intel/openvino/deployment_tools/demo/nd131-openvino-fundamentals-project-starter/TensorFlow/frozen_inference_graph.xml"
VIDEO_PATH = "resources/Pedestrian_Detect_2_1_1.mp4"

# MQTT server environment variables
HOSTNAME = socket.gethostname()
IPADDRESS = socket.gethostbyname(HOSTNAME)
MQTT_HOST = IPADDRESS
MQTT_PORT = 3001
MQTT_KEEPALIVE_INTERVAL = 60


def build_argparser():
    """
    Parse command line arguments.

    :return: command line arguments
    """
    parser = ArgumentParser()
    parser.add_argument("-m", "--model", required=True, type=str,
                        help="Path to an xml file with a trained model.")
    parser.add_argument("-i", "--input", required=True, type=str,
                        help="Path to image or video file")
    parser.add_argument("-d", "--device", type=str, default="CPU",
                        help="Specify the target device to infer on: "
                             "CPU, GPU, FPGA or MYRIAD is acceptable. Sample "
                             "will look for a suitable plugin for device "
                             "specified (CPU by default)")
    parser.add_argument('--rtsp', dest='use_rtsp',
                        help='use IP CAM (remember to also set --uri)',
                        action='store_true')
    parser.add_argument('--uri', dest='rtsp_uri',
                        help='RTSP URI, e.g. rtsp://192.168.1.64:554',
                        default=None, type=str)
    parser.add_argument('--latency', dest='rtsp_latency',
                        help='latency in ms for RTSP [200]',
                        default=200, type=int)
    parser.add_argument('--width', dest='image_width',
                        help='image width [1920]',
                        default=1920, type=int)
    parser.add_argument('--height', dest='image_height',
                        help='image height [1080]',
                        default=1080, type=int)
    parser.add_argument("-pt", "--prob_threshold", type=float, default=0.5,
                        help="Probability threshold for detections filtering"
                        "(0.5 by default)")
    return parser


def connect_mqtt():
    # Connect to the MQTT client ###
    client = mqtt.Client()
    client.connect(MQTT_HOST, MQTT_PORT, MQTT_KEEPALIVE_INTERVAL)
    return client


def infer_on_stream(args, client):
    """
    Initialize the inference network, stream video to network,
    and output stats and video.

    :param args: Command line arguments parsed by `build_argparser()`
    :param client: MQTT client
    :return: None
    """
    # Initialise the class
    plugin = Network()

    # Set Probability threshold for detections
    if not args.prob_threshold is None:
        prob_threshold = args.prob_threshold
    else:
        prob_threshold = 0.3

    # Load the model through `infer_network`
    plugin.load_model(model=args.model,
                      device=args.device)
    net_input_shape = plugin.get_input_shape()

    # Handle the input stream ###
    if args.input == 'CAM':
        input_stream = 0
        single_image_mode = False
    elif args.input[-4:] in [".jpg", ".bmp", ".png"]:
        single_image_mode = True
        input_stream = args.input
    else:
        single_image_mode = False
        input_stream = args.input
        assert os.path.isfile(input_stream)

    if args.use_rtsp:
        cap = open_rtsp_cam(args.rtsp_uri,
                            args.image_width,
                            args.image_height,
                            args.rtsp_latency)
        single_image_mode = False
    else:
        cap = cv2.VideoCapture(input_stream)
        cap.open(input_stream)

    if not cap.isOpened():
        log.error("Unable open video stream")
    logger.debug("Weight-Height: " + str(cap.get(cv2.CAP_PROP_FRAME_WIDTH)
                                         ) + "-" + str(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    data_list = []
    total_count = 0
    current_count = 0
    request_id = 0
    num_detected = 0
    inference_t = 0
    max_len = 50
    prev_count = 0
    duration = 0
    threshold = 0.1
    track = deque(maxlen=max_len)

    # Loop until stream is over ###
    while cap.isOpened():
        log_data = {}

        # Read from the video capture ###
        flag, frame = cap.read()

        if not flag:
            sys.stdout.flush()
            break

        width = int(cap.get(3))
        height = int(cap.get(4))
        displayFrame = frame.copy()

        # Pre-process the image as needed ###
        logger.debug("size: ".format(net_input_shape))
        p_frame = cv2.resize(frame, (net_input_shape[3], net_input_shape[2]))
        p_frame = p_frame.transpose((2, 0, 1))
        p_frame = p_frame.reshape(net_input_shape)

        # Start asynchronous inference for specified request ###
        t0 = time.time()
        plugin.exec_net(request_id, p_frame)

        # Wait for the result ###
        if plugin.wait(request_id) == 0:

            # Get the results of the inference request ###
            result = plugin.get_all_output()
            t1 = time.time()
            inference_t = t1 - t0

            # Extract any desired stats from the results ###
            output = result['DetectionOutput']
            counter = 0

            for detection in output[0][0]:
                image_id, label, conf, x_min, y_min, x_max, y_max = detection

                if conf > prob_threshold:
                    print("label " + str(label) + "imageid" + str(image_id))
                    x_min = int(x_min * width)
                    x_max = int(x_max * width)
                    y_min = int(y_min * height)
                    y_max = int(y_max * height)

                    try:
                        if conf > 0.85:
                            crop_target = frame[y_min:y_max, x_min:x_max]

                    except Exception as err:
                        print(err)
                        pass
                    print(err)

                    x_min_diff = last_x_min - x_min
                    x_max_diff = last_x_max - x_max

                    if x_min_diff > 0 and x_max_diff > 0:  
                        continue

                    y_min_diff = abs(last_y_min) - abs(y_min)

                    counter = counter + 1

                    last_x_min = x_min
                    last_x_max = x_max
                    last_y_max = y_max
                    last_y_min = y_min

                    cv2.rectangle(displayFrame, (x_min, y_min),
                                  (x_max, y_max), (0, 255, 0), 2)

                    cv2.putText(displayFrame, (x_max + 10, y_min + 50), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1,
                                (230, 50, 2),
                                lineType=cv2.LINE_8, thickness=1)

                    last_detection_time = datetime.now()
               
                    if start is None:
                        start = time.time()
                        time.clock()
            process_t = time.time() - t1

            # Calculate and send relevant information on ###
            ### current_count, total_count and duration to the MQTT server ###
            ### Topic "person": keys of "count" and "total" ###

            track.append(counter)
            num_detected = 0

            if np.sum(track)/max_len > threshold:
                num_detected = 1

            if num_detected > prev_count:
                start_time = time.time()
                num_persons = num_detected - prev_count
                total_count += num_persons
                prev_count = num_detected
                client.publish("person", json.dumps(
                    {"total": total_count}), retain=True)

            ### Topic "person/duration": key of "duration" ###
            if num_detected < prev_count:
                prev_count = num_detected

            if num_detected > 0:
                duration += (time.time() - start_time)/10
                logger.debug("Duration: {}".format(duration))
            if total_count > 0:
                avg_duration = time/total_count
                client.publish("person/duration",
                               json.dumps({"duration": int(avg_duration)}))

            client.publish("person", json.dumps(
                {"count": num_detected}), retain=True)

        log_data['time'] = time.strftime("%H:%M:%S", time.localtime())
        log_data['count'] = counter
        log_data['num_detected'] = num_detected
        log_data['num_persons'] = num_persons
        log_data['prev_count'] = prev_count
        log_data['total_count'] = total_count
        log_data['duration'] = duration
        log_data['avg_duration'] = avg_duration
        log_data['inference_t'] = inference_t
        log_data['process_t'] = process_t
        log_data['result'] = result
        data_list.append(log_data)

        key_pressed = cv2.waitKey(60)
        if key_pressed == 27:
            write_csv(data_list)
            cap.release()
            cv2.destroyAllWindows()
            client.disconnect()
            break

        # Send the frame to the FFMPEG server ###
        logger.debug("Image_size: {}".format(displayFrame.shape))
        sys.stdout.buffer.write(displayFrame)
        sys.stdout.flush()

        # Write an output image if `single_image_mode`
        if single_image_mode:
            cv2.imwrite("output.jpg",)

        write_csv(data_list)
        cap.release()
        cv2.destroyAllWindows()
        client.disconnect()

def write_csv(data):
    with open('./log.csv', 'w') as outfile:
        writer = DictWriter(outfile, ('time', 'count', 'num_detected',
                                      'num_persons', 'prev_count',
                                      'total_count', 'duration',
                                      'avg_duration', 'inference_t',
                                      'process_t', 'result'))
        writer.writeheader()
        writer.writerows(data)

def count_targets(detections, image):
    num_detections = 0
    draw_bounding_box = image
    if len(detections) > 0:
        draw_bounding_box, num_detections = draw_boxes(detections, image)
    return num_detections, draw_bounding_box

def draw_boxes(boxes, image):
    num_detections = 0
    for box in boxes:
        logger.debug("box: {}".format(box))
        if box['class_id'] == 0:
            if box['confidence'] > 0:
                cv2.rectangle(
                    image, (box['xmin'], box['ymin']), (box['xmax'], box['ymax']), (0, 255, 0), 1)
                num_detections += 1
    return image, num_detections

def open_rtsp_cam(uri, width, height, latency):
    gst_str = ('rtspsrc location={} latency={} ! '
               'rtph264depay ! h264parse ! omxh264dec ! '
               'nvvidconv ! '
               'video/x-raw, width=(int){}, height=(int){}, '
               'format=(string)BGRx ! '
               'videoconvert ! appsink').format(uri, latency, width, height)
    return cv2.VideoCapture(gst_str, cv2.CAP_GSTREAMER)

def open_usb_cam(dev, width, height):
    # Set width and height here, otherwise we could just do:
    #     return cv2.VideoCapture(dev)
    gst_str = ('v4l2src device=/dev/video{} ! '
               'video/x-raw, width=(int){}, height=(int){} ! '
               'videoconvert ! appsink').format(dev, width, height)
    return cv2.VideoCapture(gst_str, cv2.CAP_GSTREAMER)

def open_onboard_cam(width, height):
    gst_elements = str(subprocess.check_output('gst-inspect-1.0'))
    if 'nvcamerasrc' in gst_elements:
        # On versions of L4T prior to 28.1, add 'flip-method=2' into gst_str
        gst_str = ('nvcamerasrc ! '
                   'video/x-raw(memory:NVMM), '
                   'width=(int)2592, height=(int)1458, '
                   'format=(string)I420, framerate=(fraction)30/1 ! '
                   'nvvidconv ! '
                   'video/x-raw, width=(int){}, height=(int){}, '
                   'format=(string)BGRx ! '
                   'videoconvert ! appsink').format(width, height)
    elif 'nvarguscamerasrc' in gst_elements:
        gst_str = ('nvarguscamerasrc ! '
                   'video/x-raw(memory:NVMM), '
                   'width=(int)1920, height=(int)1080, '
                   'format=(string)NV12, framerate=(fraction)30/1 ! '
                   'nvvidconv flip-method=2 ! '
                   'video/x-raw, width=(int){}, height=(int){}, '
                   'format=(string)BGRx ! '
                   'videoconvert ! appsink').format(width, height)
    else:
        raise RuntimeError('onboard camera source not found!')
    return cv2.VideoCapture(gst_str, cv2.CAP_GSTREAMER)

def main():
    """
    Load the network and parse the output.

    :return: None
    """
    # Grab command line args
    args = build_argparser().parse_args()
    print('Called with args:')
    print(args)
    print('OpenCV version: {}'.format(cv2.__version__))

    # Connect to the MQTT server
    client = connect_mqtt()
    # Perform inference on the input stream
    infer_on_stream(args, client)

if __name__ == '__main__':
    main()
