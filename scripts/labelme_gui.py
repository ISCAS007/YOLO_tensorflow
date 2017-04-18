'''
A simple Gooey example. One required field, one optional.
'''


import argparse
from gooey import Gooey, GooeyParser
import os

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
  cmd='./labelme_sh.sh '+args.input+" out.json"
  print(cmd)
  os.system(cmd)

if __name__ == '__main__':
  main()
