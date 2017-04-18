# -*- coding: utf-8 -*-

from yolo_video import YOLO_VIDEO
yolo = YOLO_VIDEO()

#yolo.disp_console = True
#yolo.imshow = True
#yolo.tofile_video = '../test/output.avi'
#yolo.tofile_txt = '../test/output.txt'
#yolo.filewrite_video = True
#yolo.filewrite_txt = True
yolo.rules=[['aeroplane',600,400,100,100,0.3],['aeroplane',800,100,100,100,0.3]]
yolo.tofile_video = '../test/airplane1.avi'
yolo.tofile_txt = '../test/airplane1.txt'
yolo.MaxFrameNum = 500
yolo.detect_from_file('/home/nfs/airplane1.mp4')

yolo.rules=[['aeroplane',600,400,100,100,0.3],['aeroplane',800,100,100,100,0.3]]
yolo.tofile_video = '../test/airplane2.avi'
yolo.tofile_txt = '../test/airplane2.txt'
yolo.MaxFrameNum = 500
yolo.detect_from_file('/home/nfs/airplane2.mp4')
