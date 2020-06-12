python3 -m venv venv
source /opt/intel/openvino/bin/setupvars.sh -pyver 3.5
source venv/bin/activate
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
python3 main.py -i resources/Pedestrian_Detect_2_1_1.mp4 -m frozen_inference_graph.xml -d CPU -pt 0.6 | ffmpeg -v warning -f rawvideo -pixel_format bgr24 -video_size 768x432 -framerate 24 -i - http://0.0.0.0:3004/fac.ffm