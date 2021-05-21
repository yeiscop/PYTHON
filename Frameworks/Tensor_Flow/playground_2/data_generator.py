import random
import math

a = 0.5

def generate(numPoints):
  result = []
  for i in range(0, numPoints):
    t = random.uniform(0,6/a)
    if (i%2==0):
      x = a * t * math.cos(t) + random.uniform(-0.1,0.1)
      y = a * t * math.sin(t) + random.uniform(-0.1,0.1)
      color = 0
    else:
      x = -a * t * math.cos(t) + random.uniform(-0.1,0.1)
      y = -a * t * math.sin(t) + random.uniform(-0.1,0.1)
      color = 1
    result.append((x,y,color))
  return result

def distance(pointFrom, pointTo):
  diff = (pointTo[0] - pointFrom[0], pointTo[1] - pointFrom[1])
  return math.sqrt(diff[0]*diff[0]+diff[1]*diff[1])
