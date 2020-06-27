python3 -m venv venv
source /opt/intel/openvino/bin/setupvars.sh -pyver 3.5
source venv/bin/activate
mkdir models
python3  /opt/intel/openvino/deployment_tools/tools/model_downloader/downloader.py --name person-detection-retail-0013 -o models/
pip3 install opencv-python
pip3 install paho-mqtt python-etcd
pip3 install -r requirements.txt
cd webservice/server
npm install
cd ../ui
npm install
cd ../../
cd webservice/server
kill $(sudo lsof -t -i:3001)
node server.js &
cd ../ui
kill $(sudo lsof -t -i:3000)
npm run dev &
cd ../../
kill $(sudo lsof -t -i:3004)
ffserver -f ./ffmpeg/server.conf &
python main.py -i resources/Pedestrian_Detect_2_1_1.mp4 -m models/intel/person-detection-retail-0013/FP16/person-detection-retail-0013.xml -l /opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/libcpu_extension_sse4.so -d CPU -pt 0.6 | ffmpeg -v warning -f rawvideo -pixel_format bgr24 -video_size 768x432 -framerate 24 -i - http://0.0.0.0:3004/fac.ffm