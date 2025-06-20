def largest_number_in_array():
  array = [2, 19, 5, 17]
  result = array[0]
  for x in array:
    if x > result:
        result = x
  print(result)
