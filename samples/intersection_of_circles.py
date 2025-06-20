import math
a, b, c = 0, 0, 5
d, e, f = 7, 0, 6

g = d - a
h = e - b

i = g * g + h * h
j = math.sqrt(i)

if c + f < j:
  print("P")
else:
  print("Q")