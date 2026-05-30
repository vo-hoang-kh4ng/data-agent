from solution_lcb_002_palindrome import *


def test_palindrome_positive():
    assert isPalindrome(121) == True

def test_palindrome_negative():
    assert isPalindrome(-121) == False

def test_palindrome_ten():
    assert isPalindrome(10) == False

def test_palindrome_zero():
    assert isPalindrome(0) == True
