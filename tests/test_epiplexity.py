from core.inspector import compute_mdl_epiplexity
import os

def run_tests():
    print("Testing Epiplexity Calculation (MDL)")
    
    # Test 1: Trivial code
    trivial = "x=1"
    score1 = compute_mdl_epiplexity(trivial)
    print(f"Test 1 (trivial): {score1:.4f}")
    
    # Test 2: Moderate code
    moderate = '''import pandas as pd
df = pd.read_csv("data.csv")
print(df.describe())
df.plot(kind="bar")
'''
    score2 = compute_mdl_epiplexity(moderate)
    print(f"Test 2 (moderate): {score2:.4f}")
    
    # Test 3: Complex code
    with open('LAMBDA.py', 'r', encoding='utf-8') as f:
        complex_code = f.read()
    score3 = compute_mdl_epiplexity(complex_code)
    print(f"Test 3 (complex): {score3:.4f}")
    
    print("\nAll tests completed.")

if __name__ == "__main__":
    run_tests()
