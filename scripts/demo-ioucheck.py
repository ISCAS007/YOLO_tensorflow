# -*- coding: utf-8 -*-

from yolo_video import YOLO_VIDEO
yolo = YOLO_VIDEO()

#yolo.disp_console = True
#yolo.imshow = True
#yolo.tofile_video = '../test/output.avi'
#yolo.tofile_txt = '../test/output.txt'
#yolo.filewrite_video = True
#yolo.filewrite_txt = True
yolo.rule=['car',400,400,200,100,0.3]
yolo.detect_from_file('/media/sdb/CVDataset/UrbanTracker/rouen_video.avi')
