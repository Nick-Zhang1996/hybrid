import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pickle as p
import numpy as np

def displayResults(data):
    print('  no.cars, case 4 - collision % , case 5 - avg collision, case 5_alt, 6...')
    breakpoint()
    for records in data:
        repeats = len(records)
        records = np.array(records)
        car_count = records[0,0]
        case4_col_ratio = np.sum(records[:,1]>0)/repeats
        case4_avg_col = np.mean(records[:,1])
        case5_col_ratio = np.sum(records[:,2]>0)/repeats
        case5_avg_col = np.mean(records[:,2])
        case5_alt_col_ratio = np.sum(records[:,3]>0)/repeats
        case5_alt_avg_col = np.mean(records[:,3])
        case6_col_ratio = np.sum(records[:,4]>0)/repeats
        case6_avg_col = np.mean(records[:,4])
        print(f'{car_count}\t, {case4_col_ratio*100:.0f}%\t, {case4_avg_col:.2f}\t,{case5_col_ratio*100:.0f}%\t, {case5_avg_col:.2f}\t,{case5_alt_col_ratio*100:.0f}%\t, {case5_alt_avg_col:.2f}\t {case6_col_ratio*100:.0f}%\t, {case6_avg_col:.2f}')

with open('../benchmarks/benchmark.p', 'rb') as f:
    data = p.load(f)
displayResults(data)
