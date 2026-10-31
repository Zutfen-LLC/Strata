from pathlib import Path
import sys,subprocess,tempfile,hashlib
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'tools'))
from generate_native_hc_pending import generate
from native_hc_pending_numeric import parse,SIZE
with tempfile.TemporaryDirectory(prefix='hc-pending-fixtures-') as d:
 for case in ['zero','cancel','sparse']:
  f=Path(d)/(case+'.bin');subprocess.run([sys.argv[1],'--synthetic',case,str(f)],check=True)
  raw=generate(case);assert len(raw)==SIZE and f.read_bytes()==raw,'independent C++/Python fixture mismatch '+case
  parse(raw);print(case,hashlib.sha256(raw).hexdigest())
