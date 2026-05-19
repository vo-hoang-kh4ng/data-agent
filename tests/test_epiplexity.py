import os
import pytest
from core.inspector import compute_mdl_epiplexity, Verifier

def test_compute_mdl_epiplexity():
    # Test 1: Trivial code
    trivial = "x=1"
    score1 = compute_mdl_epiplexity(trivial)
    assert score1 >= 0.0
    
    # Test 2: Moderate code
    moderate = '''import pandas as pd
df = pd.read_csv("data.csv")
print(df.describe())
df.plot(kind="bar")
'''
    score2 = compute_mdl_epiplexity(moderate)
    assert score2 >= 0.0

def test_verifier_epiplexity_calculation():
    verifier = Verifier()
    
    # 1. Test case: Code matches task description exactly (too easy)
    task_easy = "Write a python script that prints Hello World"
    solution_easy = "print('Hello World')"
    score_easy = verifier.estimate_epiplexity(task_easy, solution_easy)
    
    # 2. Test case: Highly complex solution representing learnable information gain
    task_complex = "Perform data correlation, outlier detection, and plotting on a DataFrame"
    solution_complex = """
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
df = pd.read_csv("data.csv")
q1 = df['val'].quantile(0.25)
q3 = df['val'].quantile(0.75)
iqr = q3 - q1
df_clean = df[(df['val'] >= q1 - 1.5 * iqr) & (df['val'] <= q3 + 1.5 * iqr)]
corr = df_clean.corr()
plt.figure(figsize=(10, 6))
plt.imshow(corr, cmap='hot')
plt.colorbar()
plt.savefig("corr.png")
"""
    score_complex = verifier.estimate_epiplexity(task_complex, solution_complex)
    
    # 3. Goldilocks check
    verdict_easy = verifier.is_in_goldilocks_zone(task_easy, solution_easy)
    verdict_complex = verifier.is_in_goldilocks_zone(task_complex, solution_complex)
    
    assert "epiplexity_score" in verdict_easy
    assert "goldilocks_status" in verdict_easy
    assert "epiplexity_score" in verdict_complex
    assert "goldilocks_status" in verdict_complex
