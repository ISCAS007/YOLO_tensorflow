'''
A simple Gooey example. One required field, one optional.
'''

import argparse
from gooey import Gooey, GooeyParser
import os, sys


@Gooey()
def main():
    parser = GooeyParser(description='A simple gui.')

    parser.add_argument(
        'input',
        metavar='Input Image/Video file to Label',
        help='Image(jpg,png,jpeg);Video(mp4,avi)',
        widget="FileChooser",
        default='/home/nfs/sherbrooke_video.avi')

    args = parser.parse_args()
    if not os.path.exists(args.input):
        print('input file not exists')
        sys.exit()

    # avoid override ask
    imgname = 'out.jpg'
    if os.path.exists(imgname):
        os.system('rm '+imgname)

    if args.input.lower().endswith(('avi', 'mp4', 'mov')):
        cmd = 'ffmpeg -i ' + args.input + ' -frames 1 ' + imgname
        os.system(cmd)
    elif args.input.lower().endswith(('jpg', 'png', 'jpeg')):
        imgname = args.input
    else:
        print('input is not image or video')
        sys.exit()

    # avoid override ask
    outfile = 'out.json'
    if os.path.exists(outfile):
        os.system('rm ' + outfile)

    cmd = './labelme_sh.sh ' + imgname + ' ' + outfile
    print(cmd)
    os.system(cmd)

    if args.input.lower().endswith(('avi', 'mp4', 'mov')):
        cmd = '/usr/bin/python demo-ioucheck.py ' + args.input + ' ' + outfile
        print(cmd)
        #os.system(cmd)


if __name__ == '__main__':
    main()
