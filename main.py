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
import json
import cv2
import logging as log
import paho.mqtt.client as mqtt
from argparse import ArgumentParser
from inference import Network
CPU_EXTENSION = "/opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/libcpu_extension_sse4.so"
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
    parser.add_argument("-l", "--cpu_extension", required=False, type=str,
                        default=None,
                        help="MKLDNN (CPU)-targeted custom layers."
                             "Absolute path to a shared library with the"
                             "kernels impl.")
    parser.add_argument("-d", "--device", type=str, default="CPU",
                        help="Specify the target device to infer on: "
                             "CPU, GPU, FPGA or MYRIAD is acceptable. Sample "
                             "will look for a suitable plugin for device "
                             "specified (CPU by default)")
    parser.add_argument("-pt", "--prob_threshold", type=float, default=0.5,
                        help="Probability threshold for detections filtering"
                        "(0.5 by default)")
    return parser


def connect_mqtt():
    ### TODO: Connect to the MQTT client ###
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
    infer_network = Network()

    # Set Probability threshold for detections
    prob_threshold = args.prob_threshold

    # Load the model through `infer_network`
    infer_network.load_model(model=args.model,
                             device=args.device,
                             cpu_extension=args.cpu_extension)
    net_input_shape = infer_network.get_input_shape()

    # Handle the input stream ###
    input_stream = args.input
    cap = cv2.VideoCapture(input_stream)
    cap.open(input_stream)
    if not cap.isOpened():
        log.error("Unable open video stream")
    # Loop until stream is over ###
    while cap.isOpened():
        ### TODO: Read from the video capture ###
        flag, frame = cap.read()
        if not flag:
            break
         request_id = 0
        # Pre-process the image as needed ###
        logger.debug("size: ".format(net_input_shape))
        p_frame = cv2.resize(frame, (net_input_shape[3], net_input_shape[2]))
        p_frame = p_frame.transpose((2, 0, 1))
        p_frame = p_frame.reshape(net_input_shape)
        # Start asynchronous inference for specified request ###
        infer_network.exec_net(request_id, p_frame)
        # Wait for the result ###
        if plugin.wait() == 0:
            # Get the results of the inference request ###
            result = infer_network.exec_net(request_id, frame.shape,prob_threshold)

            # Extract any desired stats from the results ###
            count, box_frame = count_targets(result,frame)

            ### TODO: Calculate and send relevant information on ###
            ### current_count, total_count and duration to the MQTT server ###
            ### Topic "person": keys of "count" and "total" ###
            ### Topic "person/duration": key of "duration" ###
        key_pressed = cv2.waitKey(60)
        if key_pressed == 27:
            write_csv(data_list)
            cap.release()
            cv2.destroyAllWindows()
            client.disconnect()
            break

        # Send the frame to the FFMPEG server ###
        logger.debug("Image_size: {}".format(box_frame.shape))
        sys.stdout.buffer.write(box_frame)
        sys.stdout.flush()
    
        # Write an output image if `single_image_mode`
        if single_image_mode:
            cv2.imwrite("output.jpg",)

        write_csv(data_list)
        cap.release()
        cv2.destroyAllWindows()
        client.disconnect()

def write_csv(data):
    with open('./log.csv','w') as outfile:
        writer = DictWriter(outfile,())
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
                cv2.rectangle(image,(box['xmin'], box['ymin']), (box['xmax'], box['ymax']), (0,255,0), 1)
                num_detections +=1
    return image, num_detections

def main():
    """
    Load the network and parse the output.

    :return: None
    """
    # Grab command line args
    args = build_argparser().parse_args()
    # Connect to the MQTT server
    client = connect_mqtt()
    # Perform inference on the input stream
    infer_on_stream(args, client)


if __name__ == '__main__':
    main()
