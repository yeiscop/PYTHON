import math
from PIL import Image, ImageDraw

import data_generator

points = data_generator.generate(250)

img = Image.new("RGB", (12*100,12*100), "white")
draw = ImageDraw.Draw(img)
dotSize = 5

for point in points:
  x = math.trunc((point[0]+6)*100)
  y = math.trunc((point[1]+6)*100)
  draw.rectangle([x,y,x+dotSize-1,y+dotSize-1], fill="orange" if point[2]==0 else "blue")

img.show() # View in default viewer