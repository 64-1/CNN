class AVLNode:
    __slots__ = ('key','left', 'right', 'h')
    def __init__(self, key):
        self.key, self.left, self.right, self.h = key, None, None, 1
    
def height(n): return n.h if n else 0
def update(n): n.h = 1 + max(height(n.left), height(n.right))

def rotate_right(y):
    x, T2 = y.left, y.left.right
    x.right, y.left = y, T2
    update(y); update(x)
    return x

def rotate_left(x):
    y, T2 = x.right, x.right.left
    y.left, x.right = x, T2
    update(x); update(y)
    return y

def balance(n):
    bf = height(n.left) - height(n.right)
    if bf > 1:
        if height(n.left.left) < height(n.left.right):
            n.left = rotate_right(n.left)
        return rotate_right(n)
    if bf < -1:
        if height(n.right.right) < height(n.right.left):
            n.right = rotate_right(n.right)
        return rotate_left(n)
    return n

def avl_insert(n, key):
    if not n:
        return AVLNode(key)
    if key < n.key:
        n.left = avl_insert(n.left, key)
    elif key > n.key:
        n.right = avl_insert(n.right, key)
    else:
        return n  # No duplicates allowed
    update(n)
    return balance(n)