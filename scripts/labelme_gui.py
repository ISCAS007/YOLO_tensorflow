'''
A simple Gooey example. One required field, one optional.
'''


import argparse
from gooey import Gooey, GooeyParser
import os,sys

@Gooey()
def main():
  parser = GooeyParser(description='A simple gui.')

  parser.add_argument(
    'input',
    metavar='Input Image/Video file to Label',
    help='Image(jpg,png,jpeg);Video(mp4,avi)',
    widget="FileChooser",
    default='/home/yzbx/Pictures/Car.png')

  args = parser.parse_args()
  if not os.path.exists(args.input):
    print('input file not exists')
    sys.exit()

  imgname='out.jpg'
  if args.input.lower().endswith(('avi','mp4','mov')):
    cmd='ffmpeg -i '+args.input+' -frames 1 '+imgname
    os.system(cmd)
  elif args.input.lower().endswith(('jpg','png','jpeg')):
    imgname=args.input
  else:
    print('input is not image or video')
    sys.exit()

  cmd='./labelme_sh.sh '+imgname+" out.json"
  print(cmd)
  os.system(cmd)

if __name__ == '__main__':
  main()
