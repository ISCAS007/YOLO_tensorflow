# -*- coding: utf-8 -*-

from poly_ioucheck import poly_check

polyA=[]
polyA.append((100,100))
polyA.append((100,200))
polyA.append((100,300))
polyA.append((100,400))
polyA.append((400,400))

polyB=[]
polyB.append((100,400))
polyB.append((200,400))
polyB.append((300,400))
polyB.append((400,400))
polyB.append((400,100))

assert(not poly_check(polyA,polyB))