# Project Write-Up  
## Explaining Custom Layers

The list of [supported layers](https://docs.openvinotoolkit.org/2019_R3/_docs_MO_DG_prepare_model_Supported_Frameworks_Layers.html) directly relates to whether a given layer is a custom layer. Any layer not in that list is automatically classified as a custom layer by the Model Optimizer.
To add custom layers, there are a few differences depending on the original model framework. In both TensorFlow and Caffe, the first option is to register the custom layers as extensions to the Model Optimizer:
- For Caffe, the second option is to register the layers as Custom, then use Caffe to calculate the output shape of the layer. You’ll need Caffe on your system to do this option.
- For TensorFlow, its second option is to actually replace the unsupported subgraph with a different subgraph. The final TensorFlow option is to actually offload the computation of the subgraph back to TensorFlow during inference.  
## Running the project

For running the video file
```
python main.py -i resources/Pedestrian_Detect_2_1_1.mp4 -m models/intel/person-detection-retail-0013/FP16/person-detection-retail-0013.xml -l /opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/libcpu_extension_sse4.so -d CPU -pt 0.6 | ffmpeg -v warning -f rawvideo -pixel_format bgr24 -video_size 768x432 -framerate 24 -i - http://0.0.0.0:3004/fac.ffm
```  

For running a camera  
```
python3 main.py -i CAM -m models/intel/person-detection-retail-0013/FP16/person-detection-retail-0013.xml -d CPU -pt 0.6 | ffmpeg -v warning -f rawvideo -pixel_format bgr24 -video_size 768x432 -framerate 24 -i - http://0.0.0.0:3004/fac.ffm
```  
For running a IP CAM
```
python3 main.py --rtsp --uri http://0.0.0.0:3004/fac.ffm  -m models/intel/person-detection-retail-0013/FP16/person-detection-retail-0013.xml -d CPU -pt 0.6 | ffmpeg -v warning -f rawvideo -pixel_format bgr24 -video_size 768x432 -framerate 24 -i - http://0.0.0.0:3004/fac.ffm
```


## Comparing Model Performance

My method to compare [ssd_mobilenet_v2_coco_2018_03_29](http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v2_coco_2018_03_29.tar.gz) before and after conversion to Intermediate Representations is to calculate `total_inference_time = inference_end_time - inference_start_time` around execuete the following code: 
- pre-conversion
```
inference_start_time = time.time()
sess.run([sess.graph.get_tensor_by_name('num_detections:0'),
                            sess.graph.get_tensor_by_name('detection_scores:0'),
                            sess.graph.get_tensor_by_name('detection_boxes:0'),
                            sess.graph.get_tensor_by_name('detection_classes:0')],
                           feed_dict={
                               'image_tensor:0': img.reshape(1,
                                                             img.shape[0],
                                                             img.shape[1], 3)})
inference_end_time = time.time()
total_inference_time = inference_end_time - inference_start_time
```    
- post-conversion  
```
t0 = time.time()
plugin.exec_net(p_frame)

# Wait for the result ###
if plugin.wait() == 0:

   # Get the results of the inference request ###
   result = plugin.get_output()
   t1 = time.time()
   inference_t = t1 - t0
```

were:  
- Comparing the size of both model
- Comparing the accuracy of model
- Comparing the inference time of pre- and post-conversio model

The difference between model accuracy, size and inference time of pre- and post-conversion as follow:

Parameters | pre-conversion | post-conversion
| ------------- | ------------- | -------------
size  | 69.7 MB  | 67.5 MB
accuracy  | 0.659482432  | 0.6492571
inference time  | 4828.266 ms  | 35.72 ms

## Assess Model Use Cases

Some of the potential use cases of the people counter app are...

1. Audience Statistics - By Detecting and count audience in theater and concert with this app, user only need to process CCTV footage and perform the statistics of population.
2. Retail - Directing people to less-congested queues in the store. 
3. Crowd Detection (specifically for current Covid-19) - By check on the number of people in the frame to keep control the acces at common zones such as offices or hospital.  
4. Security Systems - To restric people going to sensitive area or indivisual property.
5. Tenant Counting - Help house owner to know the people gathering in or around their house on daily basis with the function duration and total count.

## Assess Effects on End User Needs

Lighting, model accuracy, and camera focal length/image size have different effects on a
deployed edge model. The potential effects of each of these are as follows:

- Lighting: Lighting is essential factor affects to the result of model. In poor light enviroment, model can't predict accurately if input image is totally dark. So IP camera must place with enough light or we apply transfer learning train our model in low light environment but still if totally dark it wouldn't help.
- Model accuracy: Deployed model on edge must have high accuracy because edge device works in real time if we deployed low accuracy model then it would give great results which is not oleasant for end users.
- Camera focal length: It's depend on user reuqirements which type of camera is required. If users want to surveillance wider place than The shorter the focal length camera is better but model can extract less information about object's in picture so it can lower the accuracy. In compare if users want to surveillance narrow place then they can use longer the focal length camera. The longer the focal length, the narrower the angle of view and the higher the magnification. The shorter the focal length, the wider the angle of view and the lower the magnification. 
- Image size: Image size relate to the resolution of image. High quality images the size will be larger. Model can provide better result if image resolution is better but with higher resolution image model can take more time to inference and provide results than low quality images and also use more memory. If users have more memory and also can accept with some delay for more accurate result then higher resoltuion means larger image can be use. We usually need image size same as the image size used to train models to provide good accuracy.

## Model Research

In investigating potential people counter models, I tried each of the following three models:

- Model 1: ssd_inception_v2_coco
  - http://download.tensorflow.org/models/object_detection/ssd_inception_v2_coco_2018_01_28.tar.gz
  - I converted the model to an Intermediate Representation with the following arguments:
   ```
  python /opt/intel/openvino/deployment_tools/model_optimizer/mo.py --input_model frozen_inference_graph.pb --tensorflow_object_detection_api_pipeline_config pipeline.config --reverse_input_channels --tensorflow_use_custom_operations_config /opt/intel/openvino/deployment_tools/model_optimizer/extensions/front/tf/ssd_v2_support.json
  ```
  - The model was insufficient for the app because it does not work correctly and it can't count correctly the people in the frame.
  - I tried to improve the model for the app by adust prob_threshold but still nothing improved.  
- Model 2: faster_rcnn_inception_v2_coco
  - http://download.tensorflow.org/models/object_detection/faster_rcnn_inception_v2_coco_2018_01_28.tar.gz
  - I converted the model to an Intermediate Representation with the following arguments:
 ```
    python /opt/intel/openvino/deployment_tools/model_optimizer/mo.py --input_model model/faster_rcnn_inception_v2_coco/frozen_inference_graph.pb --tensorflow_object_detection_api_pipeline_config pipeline.config --reverse_input_channels --tensorflow_use_custom_operations_config /opt/intel/openvino/deployment_tools/model_optimizer/extensions/front/tf/faster_rcnn_support.json
 ```
 - The model was insufficient for the app because working better than previos one.
  - I tried to improve the model for the app by change prob_threshold to 0.3 but not obviously improved.  
- Model 3: ssd_mobilenet_v2_coco
  - http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v2_coco_2018_03_29.tar.gz
  - I converted the model to an Intermediate Representation with the following arguments:
  ```
  python /opt/intel/openvino/deployment_tools/model_optimizer/mo_tf.py --input_model frozen_inference_graph.pb --tensorflow_object_detection_api_pipeline_config pipeline.config --tensorflow_use_custom_operations_config /opt/intel/openvino/deployment_tools/model_optimizer/extensions/front/tf/ssd_v2_support.json --reverse_input_channel
  ```
  - The model was good for the app.
  - I tried to improve the model for the app by change prob_threshold to 0.3 and the model works well though missing draw box around people in some particular time.
  
## Result  
After run inference on above models, the suitable accurate model was the one provided by Intel® [person-detection-retail-0013](https://docs.openvinotoolkit.org/latest/_models_intel_person_detection_retail_0013_description_person_detection_retail_0013.html) with the already existing in Intermediate Representations.  
- Model 4: person-detection-retail-0013
[person-detection-retail-0013](https://docs.openvinotoolkit.org/latest/_models_intel_person_detection_retail_0013_description_person_detection_retail_0013.html)

Use this commad to dowload the model 

```
python3  /opt/intel/openvino/deployment_tools/tools/model_downloader/downloader.py --name person-detection-retail-0013 -o models/
```

Run the app 
```
python main.py -i resources/Pedestrian_Detect_2_1_1.mp4 -m models/intel/person-detection-retail-0013/FP16/person-detection-retail-0013.xml -l /opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/libcpu_extension_sse4.so -d CPU -pt 0.6 | ffmpeg -v warning -f rawvideo -pixel_format bgr24 -video_size 768x432 -framerate 24 -i - http://0.0.0.0:3004/fac.ffm
```  
[![IMAGE ALT TEXT HERE](https://img.youtube.com/vi/42AjrDQYfe4/0.jpg)](https://youtu.be/42AjrDQYfe4)
