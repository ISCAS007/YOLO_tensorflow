# -*- coding: utf-8 -*-

from yolo_video import YOLO_VIDEO
yolo = YOLO_VIDEO()

#yolo.disp_console = True
#yolo.imshow = True
#yolo.tofile_video = '../test/output.avi'
#yolo.tofile_txt = '../test/output.txt'
#yolo.filewrite_video = True
#yolo.filewrite_txt = True
yolo.rules=[['car',400,400,100,100,0.3],['bicycle',400,300,100,100,0.3]]
yolo.tofile_video = '../test/rouen_video.avi'
yolo.tofile_txt = '../test/rouen_video.txt'
yolo.detect_from_file('/media/sdb/CVDataset/UrbanTracker/rouen_video.avi')

yolo.rules=[['car',300,300,100,100,0.3],['bicycle',200,200,100,100,0.3],['person',400,400,100,100,0.3]]
yolo.tofile_video = '../test/atrium_video.avi'
yolo.tofile_txt = '../test/atrium_video.txt'
yolo.detect_from_file('/media/sdb/CVDataset/UrbanTracker/atrium_video.avi')

yolo.rules=[['car',200,200,200,100,0.3],['bicycle',400,200,100,200,0.3]]
yolo.tofile_video = '../test/sherbrooke_video.avi'
yolo.tofile_txt = '../test/sherbrooke_video.txt'
yolo.detect_from_file('/media/sdb/CVDataset/UrbanTracker/sherbrooke_video.avi')
