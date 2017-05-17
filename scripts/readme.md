# 任意曲线区域
```
%in carbon
source set-opencv2.sh
python image_draw.py
```

# 多边形区域
```
%in carbon
source activate env2
python labelme_gui.py
```

# yolo detection
```
%in dl
./run.sh

```

or 

```
# opencv interface for python
source deactivate
source set-opencv2.sh
# ffmpeg for scikit-image
export PATH=/usr/bin:$PATH
python demo-ioucheck.py ../data/sherbrooke_video.avi out.json
```
