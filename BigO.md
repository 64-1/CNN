| Operation              | Typical Cost         | Note         |
| ---------------------- | -------------------- | ------------ |
| `lst[i]` / `dict[key]` | $O(1)$               | average-case |
| `lst.append(x)`        | $O(1)$ amortised     |              |
| `lst.pop()`            | $O(1)$ tail only     |              |
| `heapq.heappush()`     | $O(\log n)$          |              |
| `sorted(lst)`          | $O(n\log n)$ Timsort |              |
