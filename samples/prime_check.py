number = 11
result = True

for i in range(2, number):
  if number % i == 0:
      result = False
      break

print(result)
