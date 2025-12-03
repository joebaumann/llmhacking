import time
import psutil
import nvidia_smi
from datetime import datetime


def timer(func):
    """Decorator that prints the runtime of the decorated function"""

    def wrapper_timer(*args, **kwargs):
        start_time = time.perf_counter()
        value = func(*args, **kwargs)
        end_time = time.perf_counter()
        run_time = end_time - start_time
        timer_end_message = f"  :):):) Finished {func.__name__!r} in {run_time:.4f} seconds"
        print(timer_end_message)
        return value

    return wrapper_timer


def get_experiment_start_date():
    return datetime.now().strftime("%Y %m %d %H:%M:%S").replace(' ', '_').replace(':', '_')


def sort_strings_substrings_last(strings):
    def has_parent(s, remaining):
        return any(s in parent and s != parent for parent in remaining)

    if not strings:
        return []
    elif '100' in strings or '40' in strings or '10' in strings:
        # if strings are numbers from 0 to 100, do not reorder
        return strings

    # Find strings that aren't substrings of any remaining string
    roots = [s for s in strings if not has_parent(s, strings)]

    # If no roots found, pick one longest string
    if not roots:
        max_len = max(len(s) for s in strings)
        roots = [next(s for s in strings if len(s) == max_len)]

    # Remove roots from remaining strings and recurse
    remaining = [s for s in strings if s not in roots]

    return roots + sort_strings_substrings_last(remaining)

    # # Example usage:
    # strings = ['a', 'ab', 'abc']
    # result = sort_strings_substrings_last(strings)
    # print(result)  # Output: ['abc', 'ab', 'a']
