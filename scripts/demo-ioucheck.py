# -*- coding: utf-8 -*-

from yolo_video import YOLO_VIDEO
import os,sys,json
import numpy as np

def yolo_video_process(inputfile,rulefile):
    yolo = YOLO_VIDEO()

    #yolo.disp_console = True
    #yolo.imshow = True
    #yolo.tofile_video = '../test/output.avi'
    #yolo.tofile_txt = '../test/output.txt'
    #yolo.filewrite_video = True
    #yolo.filewrite_txt = True
    if not os.path.exists(rulefile):
        print('rule file not exsit!')
        sys.exit()
    elif not rulefile.lower().endswith('json'):
        print('rule file is not json')
        sys.exit()

    json_rule=json.load(open(rulefile))
    yolo.rules=json_rule['shapes']

    yolo.tofile_video = '../test/rouen_video.avi'
    yolo.tofile_txt = '../test/rouen_video.txt'
    yolo.MaxFrameNum = 300
    #yolo.detect_from_file('/media/sdb/CVDataset/UrbanTracker/rouen_video.avi')
    if not os.path.exists(inputfile):
        print('input file not exists')
        sys.exit()

    yolo.detect_from_file(inputfile)

if __name__ == '__main__':
    if(len(sys.argv)<3):
        print('usage: python',sys.argv[0],'inputfile','rulefile')
        sys.exit()

    yolo_video_process(sys.argv[1],sys.argv[2])