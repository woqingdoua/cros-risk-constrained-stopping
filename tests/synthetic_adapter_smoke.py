"""Optional adapter check; requires Torch from requirements-generation.txt."""
from pathlib import Path
import importlib.util,gzip,json,tempfile
root=Path(__file__).resolve().parents[1]
p=root/'integration/la_cdm/src/dataset/ed_cds_dataset.py'
spec=importlib.util.spec_from_file_location('synthetic_ed_adapter',p);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
with tempfile.TemporaryDirectory() as d:
 p=Path(d)/'synthetic.jsonl.gz'
 record={'split':'development','row_index':0,'initial_text':'SYNTHETIC DEMO ONLY','label':'synthetic_class','action_payloads':{}}
 def write(row):
  with gzip.open(p,'wt') as f:f.write(json.dumps(row)+'\n')
 write(record);dataset=module.EDCDSDataset(str(p),{'Synthetic Test':'synthetic_test'})
 assert dataset[0]['test_results']['Synthetic Test']=='not available.\n'
 write({**record,'split':'evaluation'})
 try:module.EDCDSDataset(str(p),{})
 except AssertionError:pass
 else:raise AssertionError('Adapter accepted non-development input')
print('PASS: development-only adapter and missing-result behavior; synthetic records only')
