n = int(input("Enter the value n: "))
loop1 = 0
loop2 = 0
loop3 = 0
for i in range(0, n):
    loop1 += 1
    print(f"loop1 = {loop1}")
    for j in range(i, i * i):
        loop2 += 1
        print(f"loop2 = {loop2}")
        if j % i == 0:
            for k in range(0, j):
                loop3 += 1
                print(f"loop3 = {loop3}")

print(f"loop1: {loop1}, loop2: {loop2}, loop3: {loop3}")
