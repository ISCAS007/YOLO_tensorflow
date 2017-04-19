# opencv interface for python
source deactivate
source set-opencv2.sh
# ffmpeg for scikit-image
export PATH=/usr/bin:$PATH
python demo-ioucheck.py ../data/sherbrooke_video.avi out.json
