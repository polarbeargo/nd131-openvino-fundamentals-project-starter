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
from sklearn.metrics.pairwise import cosine_similarity
FORMATTER = log.Formatter(
    "%(asctime)s — %(name)s — %(levelname)s — %(message)s")
console_handler = log.StreamHandler(sys.stdout)
console_handler.setFormatter(FORMATTER)
logger = log.getLogger(__name__)
logger.setLevel(log.ERROR)
# logger.setLevel(log.DEBUG)
logger.addHandler(console_handler)

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

def pre_process(frame, net_input_shape):
    logger.debug("size: ".format(net_input_shape))
    p_frame = cv2.resize(frame, (net_input_shape[3], net_input_shape[2]))
    p_frame = p_frame.transpose((2, 0, 1))
    p_frame = p_frame.reshape(net_input_shape)
    return p_frame

def is_previous_detected(plugin, crop_target, net_input_shape, total_unique_targets, conf):
    idetification_frame = pre_process(
        crop_target, net_input_shape=net_input_shape)
    plugin.exec_net(idetification_frame)
    if plugin.wait() == 0:
        ident_output = plugin.get_output()
        for i in range(len(ident_output)):
            if (len(total_unique_targets) == 0):

                # print(ident_output[i].reshape(1,-1).shape)
                total_unique_targets.append(ident_output[i].reshape(1, -1))
            else:

                # print("Checking SIMILARITY WITH PREVIOUS TARGET IF THEY MATCH THEN ALTERTING PERSON COMES SECONF TIME ELSE INCREMENTING TOTAL TARGET")
                newFound = True
                detected_target = ident_output[i].reshape(1, -1)

                # Checking that detected target is in list or not
                for index in range(len(total_unique_targets)):
                    similarity = cosine_similarity(
                        detected_target, total_unique_targets[index])[0][0]

                    print(similarity)
                    if similarity > 0.65:  # 0.58
                        print("SAME TARGET FOUD")

                        # print(str(similarity) + "at "+str(index))
                        newFound = False

                        # Update detetected one
                        total_unique_targets[index] = detected_target
                        break

                if newFound and conf > 0.90:
                    total_unique_targets.append(detected_target)
                    print('NEW TARGET FOUND')
        print(len(total_unique_targets))
        return total_unique_targets

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
    inference_t = 0
    duration = 0
    threshold = 0.1
    last_detection_time = None
    start = None
    total_unique_targets = []

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
        processed_frame = pre_process(frame, net_input_shape)

        # Start asynchronous inference for specified request ###
        t0 = time.time()
        plugin.exec_net(processed_frame)

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
                            total_unique_targets = is_previous_detected(plugin, crop_target,
                                                                    net_input_shape, total_unique_targets, conf)

                            process_t = time.time() - t1
                    except Exception as err:
                        print(err)
                        pass

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

                if last_detection_time is not None:
                    second_diff = (datetime.now() - last_detection_time).total_seconds()
    
                    if second_diff >= 1.5:
                        if start is not None:
                            elapsed = time.time() - start
                            duration = elapsed - second_diff

                             ### Topic "person/duration": key of "duration" ###
                            client.publish("person/duration", json.dumps({"duration": duration}))
                            last_detection_time = None
                            start = None
    
            # Calculate and send relevant information on ###
            ### current_count, total_count and duration to the MQTT server ###
            ### Topic "person": keys of "count" and "total" ###
            client.publish("person", json.dumps({"count": counter}), "total": len(total_unique_targets)})))

        log_data['time'] = time.strftime("%H:%M:%S", time.localtime())
        log_data['count'] = counter
        log_data['total_count'] = total_unique_targets
        log_data['duration'] = duration
        log_data['inference_t'] = inference_t
        log_data['process_t'] = process_t
        log_data['result'] = result
        data_list.append(log_data)

        key_pressed = cv2.waitKey(60)
        if key_pressed == 27:
            write_csv(data_list)
            print('Write CSV')
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
        writer = DictWriter(outfile, ('time', 'count',
                                      'total_count', 'duration',
                                      'inference_t', 'process_t', 'result'))
        writer.writeheader()
        writer.writerows(data)

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
