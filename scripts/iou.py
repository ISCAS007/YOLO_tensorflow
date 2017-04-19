# -*- coding: utf-8 -*-

import csv
import cv2

def checkrule(rule,test):
    """
        return true if rule passed
        return false if rule not passed, which means overlap with the same type
        eg: rule=['dog',100,200,30,30,0.2] , [type,x_top_left,y_top_left,width,height,possibility]
        note: currently rule[5]=0.2 is not used!
    """
    if(rule[0]!=test[0]):
        return True

    rect1=rule[1:5]
    rect2=test[1:5]

    # convert [x1,y1,w,h] to [x1,y1,x2,y2]
    rect1[2]+=rect1[0]
    rect1[3]+=rect1[1]
    rect2[2]+=rect2[0]
    rect2[3]+=rect2[1]
#    print("rect1 is ",rect1)
#    print("rect2 is ",rect2)
    w=min(rect1[2],rect2[2])-max(rect1[0],rect2[0])
    h=min(rect1[3],rect2[3])-max(rect1[1],rect2[1])

    if w<=0 or h <=0:
        return True
    else:
        return False

def checkrules(rules_check,test_check):
    """
        return true if rule passed
        return false if rule not passed, which means overlap with the same type
        eg: rule=['dog',100,200,30,30,0.2]
        note: currently rule[5]=0.2 is not used!
    """
    rules=rules_check[:]
    test=test_check[:]
    for rule in rules:
        # convert center to top left
        # [x_c,y_c,w,h] -> [x1,y1,w,h]
        rule_c=rule[:]
        rule_c[1]=rule[1]-rule[3]//2
        rule_c[2]=rule[2]-rule[4]//2
        test_c=test[:]
        test_c[1]=test[1]-test[3]//2
        test_c[2]=test[2]-test[4]//2
        if(not checkrule(rule_c,test_c)):
            return False

    return True

if __name__ == '__main__':
    rule=['dog',100,200,30,30,0.2]

    img_cp=cv2.imread('test/a.jpg')
    x=rule[1]
    y=rule[2]
    w=rule[3]
    h=rule[4]

    cv2.rectangle(img_cp, (x - w, y - h), (x + w, y + h), (0, 0, 255), 2)
    cv2.rectangle(img_cp, (x - w, y - h - 20), (x + w, y - h), (125, 125, 125), -1)
    cv2.putText(img_cp, 'no'+rule[0] + ' : %.2f' % rule[5], (x - w + 5, y - h - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
            (0, 0, 0), 1)

    cv2.imshow('YOLO_small detection',img_cp)
    cv2.waitKey(0)

    csvfile = file('test/a.txt', 'r')
    reader = csv.reader(csvfile)

    for line in reader:
        print line
        line[1]=int(line[1])
        line[2]=int(line[2])
        line[3]=int(line[3])
        line[4]=int(line[4])
        line[5]=float(line[5])
        if(checkrule(rule,line)):
            print "not overlap"
        else:
            print "overlap"

    csvfile.close()
