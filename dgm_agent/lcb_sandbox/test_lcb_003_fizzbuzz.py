from solution_lcb_003_fizzbuzz import *


def test_fizzbuzz_15():
    result = fizzBuzz(15)
    assert result[0] == '1'
    assert result[2] == 'Fizz'
    assert result[4] == 'Buzz'
    assert result[14] == 'FizzBuzz'
    assert len(result) == 15

def test_fizzbuzz_1():
    assert fizzBuzz(1) == ['1']
