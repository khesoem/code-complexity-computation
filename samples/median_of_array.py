array = [1, 2, 4, 5, 6, 10]
array.sort()

if len(array) % 2 == 1:
  median = array[len(array)//2]
else:
  median = (array[len(array)//2 - 1] + array[len(array)//2]) / 2

print(median)