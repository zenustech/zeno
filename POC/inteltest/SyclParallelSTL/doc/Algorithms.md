# Algorithms

This document describes the implementation progress of the algorithms included in the Parallel STL standard. In addition, it includes notes on the iterators that our implementation can recieve/return, along with other notes on thier implementations. 

Note: The set of algorithms marked as "Implemented" are just those included within the "Dundee" release of the SYCL Parallel STL. A number of other algorithms have also been implemented, but the implementation should be considered "in-progress", and not ready for release, hence their exclusion from the set of algorithms marked as "Implemented".

If only one iterator is specified for the Ideal/Current minimum (or maxiumum) iterator, then the currently implemented iteratory is also the ideal iterator.

### Non-modifying sequence operations

| Algorithm | Implemented |  Ideal/Current minimum input iterator | Ideal/Current minimum output iterator | Notes |
| ----- | ----- | ----- | ----- | -----|
| `all_of`/`any_of`/`none_of` | no | - | - | - |
| `for_each` | yes | Input | Input | - |
| `for_each_n` | yes | Input | Input | - |
| `count`/`count_if` | yes | Input | Input | - |
| `mismatch` | no | - | - | - |
| `equal` | no | - | - | - |
| `find`/`find_if`/`find_if_not` | yes | Input | Input | - |
| `find_end` | no | - | - | - |
| `find_first_of` | no | - | - | - |
| `adjacent_find` | no | - | - | - |
| `search` | no | - | - | - |
| `search_n` | no | - | - | - |

### Modifying sequence operations

| Algorithm | Implemented |  Ideal/Current minimum input iterator | Ideal/Current minimum output iterator | Notes |
| ----- | ----- | ----- | ----- | -----|
| `copy`/`copy_if` | no | - | - | - |
| `copy_n` | no | - | - | - |
| `copy_backward` | no | - | - | - |
| `move` | no | - | - | - |
| `move_backward` | no | - | - | - |
| `fill` | yes | - | - | - |
| `fill_n` | yes | - | - | - |
| `transform` | yes | Input | Input | - |
| `generate` | no | - | - | - |
| `generate_n` | no | - | - | - |
| `swap_ranges` | no | - | - | - |
| `remove` | no | - | - | - |
| `remove_if` | no | - | - | - |
| `replace` | no | - | - | - |
| `replace_if` | no | - | - | - |
| `reverse` | no | - | - | - |
| `rotate` | no | - | - | - |
| `unique` | no | - | - | - |
| `remove_copy` | no | - | - | - |
| `remove_copy_if` | no | - | - | - |
| `replace_copy` | no | - | - | - |
| `replace_copy_if` | no | - | - | - |
| `reverse_copy` | no | - | - | - |
| `rotate_copy` | no | - | - | - |
| `unique_copy` | no | - | - | - |

### Operations on uninitialized storage

| Algorithm | Implemented |  Ideal/Current minimum input iterator | Ideal/Current minimum output iterator | Notes |
| ----- | ----- | ----- | ----- | -----|
| `uninitialized_copy` | no | - | - | - |
| `uninitialized_move` | no | - | - | - |
| `uninitialized_copy_n` | no | - | - | - |
| `uninitialized_move_n` | no | - | - | - |
| `uninitialized_fill` | no | - | - | - |
| `uninitialized_fill_n` | no | - | - | - |

### Partitioning operations

| Algorithm | Implemented |  Ideal/Current minimum input iterator | Ideal/Current minimum output iterator | Notes |
| ----- | ----- | ----- | ----- | -----|
| `is_partitioned` | no | - | - | - |
| `partition_point` | no | - | - | - |
| `partition` | no | - | - | - |
| `partition_copy` | no | - | - | - |
| `stable_partition` | no | - | - | - |

### Sorting operations

| Algorithm | Implemented |  Ideal/Current minimum input iterator | Ideal/Current minimum output iterator | Notes |
| ----- | ----- | ----- | ----- | -----|
| `is_sorted` | no | - | - | - |
| `is_sorted_until` | no | - | - | - |
| `sort` | yes | Input | Input | Although the algorithm performs random access operations, they are carried out on a sycl buffer, which the iterators are used to copy data into/out of |
| `stable_sort` | no | - | - | - |
| `partial_sort` | no | - | - | - |
| `partial_sort_copy` | no | - | - | - |
| `nth_element` | no | - | - | - |

### Binary search operations

| Algorithm | Implemented |  Ideal/Current minimum input iterator | Ideal/Current minimum output iterator | Notes |
| ----- | ----- | ----- | ----- | -----|
| `binary_search` | no | - | - | - |
| `equal_range` | no | - | - | - |

### Set operations (on sorted ranges)

| Algorithm | Implemented |  Ideal/Current minimum input iterator | Ideal/Current minimum output iterator | Notes |
| ----- | ----- | ----- | ----- | -----|
| `merge` | no | - | - | - |
| `inplace_merge` | no | - | - | - |
| `set_difference` | no | - | - | - |
| `set_intersection` | no | - | - | - |
| `set_symmetric_difference` | no | - | - | - |
| `set_union` | no | - | - | - |
| `includes` | no | - | - | - |

### Heap operations

| Algorithm | Implemented |  Ideal/Current minimum input iterator | Ideal/Current minimum output iterator | Notes |
| ----- | ----- | ----- | ----- | -----|
| `is_heap` | no | - | - | - |
| `is_heap_until` | no | - | - | - |
| `sort_heap` | no | - | - | - |

### Minimum/maximum operations

| Algorithm | Implemented |  Ideal/Current minimum input iterator | Ideal/Current minimum output iterator | Notes |
| ----- | ----- | ----- | ----- | -----|
| `max_element` | no | - | - | - |
| `min_element` | no | - | - | - |
| `minmax_element` | no | - | - | - |
| `lexicographical_compare` | no | - | - | - |

### Numeric operations

| Algorithm | Implemented |  Ideal/Current minimum input iterator | Ideal/Current minimum output iterator | Notes |
| ----- | ----- | ----- | ----- | -----|
| `inner_product` | yes | Input | Input | - |
| `adjacent_difference` | no | - | - | - |
| `reduce` | yes | Input | Input | - |
| `transform_reduce` | yes | Input | Input | - |
| `inclusive_scan` | yes | Input | Input | - |
| `exclusive_scan` | yes | Input | Input | - |
| `transform_inclusive_scan` | no | - | - | - |
| `transform_exclusive_scan` | no | - | - | - |
