
arr = [2,3,6,5,33,7,23]

def selectionSort(arr):
    for i in range(len(arr) - 1):
        # 记录最小的索引
        minIndex = i
        for j in range(i + 1, len(arr)):
            if arr[j] < arr[minIndex]:
                minIndex = j
        # i 不是最小数时， 将i 和最小数进行交换
        if i != minIndex:
            arr[i], arr[minIndex] = arr[minIndex], arr[i]
    return arr

print(selectionSort(arr))