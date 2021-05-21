import random
import math

def generate(numPoints):
  result = []
  for i in range(0, numPoints):
    x = random.uniform(-6,6)
    y = random.uniform(-6,6)
    color = 1 if distance((0,0), (x,y)) < 3 else 0
    result.append((x,y,color))
  return result

def distance(pointFrom, pointTo):
  diff = (pointTo[0] - pointFrom[0], pointTo[1] - pointFrom[1])
  return math.sqrt(diff[0]*diff[0]+diff[1]*diff[1])

