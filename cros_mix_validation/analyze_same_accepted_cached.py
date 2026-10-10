#!/usr/bin/env python3
"""Prepared reviewer reanalysis; run only on approved cached restricted data.
No fitting, policy selection, model calls or patient-level exports are performed.
"""
import argparse, csv, hashlib, json
from pathlib import Path
import numpy as np

EXPECTED = {'CROS-Det': (297, 52), 'CROS-Mix-realized': (288, 47)}
FIELDS = ['patient_key','policy','autonomous','error_stage0','error_stop','error_full']

def analyze(input_path, output_dir, provenance_path, seed=20261010, replicates=10000):
    if replicates != 10000 or seed != 20261010:
        raise ValueError('Use the prespecified reanalysis seed and 10,000 replicates.')
    with input_path.open(newline='') as f:
        reader=csv.DictReader(f)
        if reader.fieldnames != FIELDS:
            raise ValueError('Input schema must exactly match the protocol; exclude clinical text.')
        rows=list(reader)
    groups={}
    for row in rows:
        policy=row['policy']
        if policy not in EXPECTED: raise ValueError('Unexpected controller label.')
        if not row['patient_key']: raise ValueError('Missing opaque alignment key.')
        for col in FIELDS[2:]:
            if row[col] not in ('0','1'): raise ValueError('All indicators must be binary integers.')
        groups.setdefault(policy,[]).append(row)
    if not groups: raise ValueError('No records.')
    provenance=json.loads(provenance_path.read_text())
    hashes=provenance.get('source_sha256',{})
    if not hashes or not all(isinstance(v,str) and len(v)==64 and all(c in '0123456789abcdef' for c in v) for v in hashes.values()):
        raise ValueError('Extractor provenance must include valid canonical source hashes.')
    if not provenance.get('extraction_command') or not provenance.get('policy_provenance'):
        raise ValueError('Extractor command and frozen-controller provenance are required.')
    for policy in groups:
        if policy not in provenance['policy_provenance']:
            raise ValueError('Missing provenance for a controller.')
    if 'CROS-Mix-realized' in groups:
        mix=provenance['policy_provenance']['CROS-Mix-realized']
        if not mix.get('original_assignment_verified') or not mix.get('assignment_sha256'):
            raise ValueError('Original mixture assignment must be verified and hashed.')
        if not mix.get('assignment_source'):
            raise ValueError('Provide original assignment source or exact reconstruction provenance.')
    previous_keys=None; previous_fixed_errors=None; results=[]
    for policy in EXPECTED:  # Fixed reporting order, even if one controller is unavailable.
        if policy not in groups: continue
        data=groups[policy];keys=[r['patient_key'] for r in data]
        if len(data)!=367 or len(set(keys))!=367: raise ValueError('Expected 367 unique evaluation patients per policy.')
        if previous_keys is not None and set(keys)!=previous_keys: raise ValueError('Controller cohorts differ.')
        previous_keys=set(keys)
        fixed_errors={r['patient_key']:(int(r['error_stage0']),int(r['error_full'])) for r in data}
        if previous_fixed_errors is not None and fixed_errors!=previous_fixed_errors:
            raise ValueError('Fixed stage0/full outcomes differ across controller rows.')
        previous_fixed_errors=fixed_errors
        if tuple(sum(x[j] for x in fixed_errors.values()) for j in (0,1))!=(104,126):
            raise ValueError('Whole-cohort stage0/full errors do not reproduce canonical 104/126 counts.')
        accepted=[r for r in sorted(data,key=lambda r:r['patient_key']) if r['autonomous']=='1']
        errors=np.array([[int(r[c]) for c in FIELDS[3:]] for r in accepted],dtype=float)
        if (len(accepted),int(errors[:,1].sum()))!=EXPECTED[policy]:
            raise ValueError('Frozen accepted/stopping-error counts do not match canonical results.')
        # One sampled index vector per replicate is reused across all three stages.
        rng=np.random.default_rng(seed)
        draws=rng.integers(0,len(accepted),size=(replicates,len(accepted)))
        boot=errors[draws].mean(axis=1);point=errors.mean(axis=0)
        def interval(x): return [float(v) for v in np.quantile(x,[.025,.975])]
        result={'policy':policy,'evaluation_n':367,'accepted_n':len(accepted),'coverage':len(accepted)/367,'stage_errors':{}}
        for j,stage in enumerate(['stage0','stopping','full_workup']):
            result['stage_errors'][stage]={'errors':int(errors[:,j].sum()),'risk':float(point[j]),'paired_bootstrap_95_ci':interval(boot[:,j])}
        result['paired_differences']={name:{'difference':float(point[1]-point[j]),'paired_bootstrap_95_ci':interval(boot[:,1]-boot[:,j])} for name,j in [('stopping_minus_stage0',0),('stopping_minus_full_workup',2)]}
        results.append(result)
    manifest={'status':'EXPLORATORY_CACHED_SAME_ACCEPTED_PATIENT_ANALYSIS','input_sha256':hashlib.sha256(input_path.read_bytes()).hexdigest(),'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'extractor_provenance':provenance,'bootstrap_seed':seed,'bootstrap_replicates':replicates,'interval':'paired patient percentile bootstrap within frozen accepted subset','unavailable_controllers':[p for p in EXPECTED if p not in groups],'results':results}
    output_dir.mkdir(parents=True,exist_ok=False)
    (output_dir/'same_accepted_summary.json').write_text(json.dumps(manifest,indent=2)+'\n')
    with (output_dir/'same_accepted_summary.csv').open('w',newline='') as f:
        writer=csv.writer(f);writer.writerow(['policy','accepted_n','metric','point','ci_low','ci_high'])
        for r in results:
            for stage,x in r['stage_errors'].items():writer.writerow([r['policy'],r['accepted_n'],stage,x['risk'],*x['paired_bootstrap_95_ci']])
            for contrast,x in r['paired_differences'].items():writer.writerow([r['policy'],r['accepted_n'],contrast,x['difference'],*x['paired_bootstrap_95_ci']])
    return manifest

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',required=True,type=Path)
    parser.add_argument('--output-dir',required=True,type=Path)
    parser.add_argument('--provenance',required=True,type=Path)
    args=parser.parse_args()
    result=analyze(args.input,args.output_dir,args.provenance)
    print(json.dumps({'status':result['status'],'controllers_reported':len(result['results'])}))
