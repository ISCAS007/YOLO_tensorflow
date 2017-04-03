import iou
rule=['dog',20,30,10,10,0.2]

check=['dog',15,25,10,10,0.2]
assert(not iou.checkrule(rule,check))

check=['dog',25,25,10,10,0.2]
assert(not iou.checkrule(rule,check))

check=['dog',35,25,10,10,0.2]
assert(iou.checkrule(rule,check))

check=['dog',15,35,10,10,0.2]
assert(not iou.checkrule(rule,check))

check=['dog',25,35,10,10,0.2]
assert(not iou.checkrule(rule,check))

check=['dog',35,35,10,10,0.2]
assert(iou.checkrule(rule,check))

check=['dog',15,45,10,10,0.2]
assert(iou.checkrule(rule,check))

check=['dog',25,45,10,10,0.2]
assert(iou.checkrule(rule,check))

check=['dog',35,45,10,10,0.2]
assert(iou.checkrule(rule,check))

check=['dog',25,35,1,1,0.2]
assert(not iou.checkrule(rule,check))

check=['dog',15,25,100,100,0.2]
assert(not iou.checkrule(rule,check))
