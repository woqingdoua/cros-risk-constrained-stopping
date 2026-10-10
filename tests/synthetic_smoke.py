"""Exercise shipped binomial calibration functions on synthetic event counts."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'cros_mix_validation'))
from run_ed_cds_selective_sequential_pilot import upper_cp,lower_cp
assert upper_cp(0,100,.05)<.05
assert upper_cp(100,100,.05)==1
assert lower_cp(100,100,.05)>.95
assert lower_cp(0,100,.05)==0
print('PASS: exact binomial bounds on synthetic counts; no clinical files opened')
import numpy as np
from evaluate_cros_power_v2 import apply_mix
from run_ed_cds_selective_sequential_pilot import SelectivePolicy
n=32
policies=[]
for index in range(2):
 policies.append(SelectivePolicy(name=f'synthetic-{index}',family='synthetic',errors=np.full(n,index,dtype=bool),autonomous=np.full(n,index==0),test_costs=np.full(n,index,dtype=float),total_costs=np.full(n,index,dtype=float),latencies=np.full(n,index,dtype=float),tests=np.full(n,index,dtype=float)))
weights=np.array([.4,.6]);seed=42
mixed=apply_mix(policies,np.array([0,1]),weights,seed)
expected=np.searchsorted(np.cumsum(weights),np.random.default_rng(seed).random(n),side='right')
np.testing.assert_array_equal(mixed.errors,expected.astype(bool))
np.testing.assert_array_equal(mixed.tests,expected)
np.testing.assert_array_equal(mixed.autonomous,expected==0)
assert len(set(expected))==2
print('PASS: per-episode frozen mixture realization; synthetic trajectories only')
