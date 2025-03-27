import pickle as p
import numpy as np
from BenchmarkSteinMerge import displayResults
with open('benchmark.p', 'rb') as f:
    data = p.load(f)
displayResults(data)
 
