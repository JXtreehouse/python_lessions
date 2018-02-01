# https://github.com/hustcc/JS-Sorting-Algorithm/blob/master/1.bubbleSort.md
arr = [2,3,6,5,33,7,23]

def bubbleSort(arr):
    for i in range(1, len(arr)):
        for j in range(0, len(arr)-i):
            if arr[j] > arr[j+i]:
                arr[j],arr[j + i] = arr[j + i], arr[j]
    return arr

print(bubbleSort(arr))