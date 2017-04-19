# -*- coding: utf-8 -*-

import cv2
import numpy as np

def poly_check(polyA,polyB):
    FINAL_LINE_COLOR = (255)
    LINE_SIZE=1

    ##################################################### draw image
    matA=np.array(polyA)
    matB=np.array(polyB)

    sizeA=matA.max(0)
    sizeB=matB.max(0)

    width=max(sizeA[0],sizeB[0])
    height=max(sizeA[1],sizeB[1])
    img_size=(height,width)

    img1=np.zeros(img_size, np.uint8)
    img2=np.zeros(img_size, np.uint8)

    cv2.polylines(img1, np.array([polyA]), True, FINAL_LINE_COLOR, LINE_SIZE)
    cv2.polylines(img2, np.array([polyB]),True,FINAL_LINE_COLOR,LINE_SIZE)

    cv2.namedWindow('img1', flags=cv2.CV_WINDOW_AUTOSIZE)
    # And show it
    cv2.imshow('img1', img1)
    # Waiting for the user to press any key
    cv2.waitKey(20)

    cv2.namedWindow('img2', flags=cv2.CV_WINDOW_AUTOSIZE)
    # And show it
    cv2.imshow('img2', img2)
    # Waiting for the user to press any key
    cv2.waitKey(20)

    cv2.waitKey(0)

    ###################################################### fill poly
    cv2.fillPoly(img1, np.array([polyA]), FINAL_LINE_COLOR)
    cv2.fillPoly(img2, np.array([polyB]), FINAL_LINE_COLOR)

    # And show it
    cv2.imshow('img1', img1)
    # Waiting for the user to press any key
    cv2.waitKey(20)

    # And show it
    cv2.imshow('img2', img2)
    # Waiting for the user to press any key
    cv2.waitKey(20)

    ##################################################### get iou
    img_iou=img1 & img2

    cv2.namedWindow('img_iou', flags=cv2.CV_WINDOW_AUTOSIZE)
    # And show it
    cv2.imshow('img_iou', img_iou)
    # Waiting for the user to press any key
    cv2.waitKey(20)

    cv2.waitKey(0)

    area1=cv2.sumElems(img1)
    area2=cv2.sumElems(img2)
    area_iou=cv2.sumElems(img_iou)

    if area_iou[0] > 0:
        return False
    else:
        return True