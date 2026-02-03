arr = []
for i in range(101):
    arr.append(False)
for i in range(1,101):
    for j in range(0,101,i):
        if arr[j] == True:
            arr[j] = False
                
        elif arr[j] == False:
            arr[j] = True


for i in range(len(arr)):
    if arr[i] == True:
        print(f'array position{i} is true')


        
