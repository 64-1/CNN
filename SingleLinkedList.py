from dataclasses import dataclass
from typing import Any, Optional

@dataclass
class Node:
    val: Any
    next: Optional['Node'] = None

class SingleLinkedList:
    def __init__(self):
        self.head: Optional[Node] = None

    def push_front(self, x):
        self.head = Node(val=x, next=self.head)
        
    def pop_front(self):
        if not self.head: raise IndexError("List is empty")
        x = self.head.val
        self.head = self.head.next
        return x
    
    def __iter__(self):
        current = self.head
        while current:
            yield current.val
            current = current.next