i = 14
result = ""

while i > 0:
  if i % 2 == 0:
      result = "0" + result
  else:
      result = "1" + result
  i = i // 2

print(result)