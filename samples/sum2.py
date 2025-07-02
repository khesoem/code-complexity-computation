matrix = [[1, 2, 3],[4, 5, 6],[7, 8, 9]]
rows = len ( matrix )
cols = len ( matrix [0]) if rows > 0 else 0
total_sum = 0
for index in range ( rows * cols ):
    total_sum += matrix [index // cols][index % cols]
print(total_sum)