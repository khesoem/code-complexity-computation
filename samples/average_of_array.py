numbers = [2, 4, 1, 9]
number1 = 0
number2 = 0

while number1 < len(numbers):
  number2 = number2 + numbers[number1]
  number1 = number1 + 1

result = number2 / number1
print(result)
