a, b = 0, 1
#swap without temp
a, b = b, a

# List comprehension beats append in python
squares = [x*x for x in range(10)]

arr = []
#counting frequencies fast
from collections import Counter
freq = Counter(arr)

# Recursive coin change problem: Find minimum number of coins to make a given amount
def min_coins(coins, amount):
    # Base case
    if amount == 0:
        return 0
    if amount < 0:
        return float('inf')
    
    # Initialize best to infinity
    best = float('inf')
    
    # Try each coin as a choice
    for coin in coins:
        # Recursively solve for remaining amount after taking this coin
        subproblem = min_coins(coins, amount - coin)
        
        # Update best if this choice leads to a better solution
        if subproblem != float('inf'):
            best = min(best, subproblem + 1)
    
    return best

# Example usage
coins = [1, 5, 10, 25]  # Coin denominations
amount = 63             # Target amount
print(f"Minimum coins needed: {min_coins(coins, amount)}")

#selection sort
def selection_sort(a):
    n = len(a)
    for i in range(n):
        mi = min (range(i,n), key = a.__getitem__)
        a[i], a[mi] = a[mi], a[i]
    return a

def merge_sort(arr):
    if len(arr) > 1:
        mid = len(arr) // 2
        L = arr[:mid]
        R = arr[mid:]

        merge_sort(L)
        merge_sort(R)

        i = j = k = 0

        while i < len(L) and j < len(R):
            if L[i] < R[j]:
                arr[k] = L[i]
                i += 1
            else:
                arr[k] = R[j]
                j += 1
            k += 1

        while i < len(L):
            arr[k] = L[i]
            i += 1
            k += 1

        while j < len(R):
            arr[k] = R[j]
            j += 1
            k += 1

    return arr

def quick_sort(arr):
    if len(arr) <= 1:
        return arr
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    return quick_sort(left) + middle + quick_sort(right)

def bubble_sort(arr):
    n = len(arr)
    for i in range(n):
        for j in range(0, n-i-1):
            if arr[j] > arr[j+1]:
                arr[j], arr[j+1] = arr[j+1], arr[j]
    return arr

def insertion_sort(arr):
    for i in range(1, len(arr)):
        key = arr[i]
        j = i - 1
        while j >= 0 and key < arr[j]:
            arr[j + 1] = arr[j]
            j -= 1
        arr[j + 1] = key
    return arr


