def sum_of_three_digits():
  number = 323
  result = 0

  while number != 0:
    result = result + number % 10
    number = number // 10
  print(result)