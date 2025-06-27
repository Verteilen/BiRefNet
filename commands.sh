python test.py -i test -o out

ffmpeg -y -hide_banner -i test/34325707.png -i out/34325707.png -filter_complex "[0:v][1:v]alphamerge" -pix_fmt rgb24 final/test.png

ffmpeg -y -hide_banner -i final/test.png -vf lut3d="B2048_add.cube" -q:v 1 final/test2.png
