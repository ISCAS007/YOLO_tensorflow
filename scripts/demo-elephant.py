# -*- coding: utf-8 -*-

from yolo_video import YOLO_VIDEO
yolo = YOLO_VIDEO()

#yolo.disp_console = True
#yolo.imshow = True
#yolo.tofile_video = '../test/output.avi'
#yolo.tofile_txt = '../test/output.txt'
#yolo.filewrite_video = True
#yolo.filewrite_txt = True
yolo.rules=[['car',200,100,100,100,0.3],['car',400,100,100,100,0.3]]
yolo.tofile_video = '../test/elephant.avi'
yolo.tofile_txt = '../test/elephant.txt'
yolo.MaxFrameNum = 500
yolo.detect_from_file('/home/nfs/elephant.mp4')
