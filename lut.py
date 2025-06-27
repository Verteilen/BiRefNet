from PIL import Image
from pillow_lut import load_cube_file
from argparse import ArgumentParser, Namespace

parser = ArgumentParser(description="Training script parameters")
parser.add_argument('-i', type=str, default = None)
parser.add_argument('-o', type=str, default = None)
args = parser.parse_args(sys.argv[1:])

lut = load_cube_file("B2048_add.cube")
im = Image.open(args.i)
im.filter(lut).save(args.o)
