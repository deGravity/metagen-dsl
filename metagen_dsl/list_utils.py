from collections.abc import Callable

INVALID_LIST_INDEX = -1

def find_indices(lst:list, condition:Callable):
    return [i for i, elem in enumerate(lst) if condition(elem)]

def find_first_index_of(lst:list, condition:Callable):
    elems_meeting_cond = find_indices(lst, condition)
    if len(elems_meeting_cond) > 0:
        return elems_meeting_cond[0]
    return INVALID_LIST_INDEX

def contains(s, substr) -> bool:
    return s.find(substr) != -1