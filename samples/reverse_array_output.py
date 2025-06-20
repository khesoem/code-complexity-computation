array = [1, 6, 4, 10, 2]

result = ""

for i in range(len(array)//2):
  tmp = array[len(array) - i - 1]
  array[len(array) - i - 1] = array[i]
  array[i] = tmp

for elem in array:
  result = str(elem) + " " + result
print(result)
