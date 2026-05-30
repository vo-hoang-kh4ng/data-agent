from solution_lcb_001_two_sum import *


def test_twoSum_basic():
    assert twoSum([2, 7, 11, 15], 9) == [0, 1]

def test_twoSum_middle():
    assert twoSum([3, 2, 4], 6) == [1, 2]

def test_twoSum_same():
    assert twoSum([3, 3], 6) == [0, 1]
