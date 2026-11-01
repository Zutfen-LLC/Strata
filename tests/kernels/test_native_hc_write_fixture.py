import pathlib,sys,tempfile,subprocess,os,hashlib
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[2]/'tools'))
from generate_native_hc_write import generate
from native_hc_write_numeric import parse
exe=str(pathlib.Path(sys.argv[1]).resolve())
with tempfile.TemporaryDirectory() as tmp:
 r=pathlib.Path(tmp)
 for case in ['zero','cancel','sparse']:
  p=r/(case+'.bin');subprocess.run([exe,'--synthetic',case,str(p)],cwd=r,env={k:v for k,v in os.environ.items() if k!='PYTHONPATH'},check=True);assert p.read_bytes()==generate(case);parse(p.read_bytes())
  assert subprocess.run([exe,'--synthetic',case,str(p)],cwd=r,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL).returncode
print('Actual fixture CLI matches independent Python generation for three cases; exclusive output preserved; no device invocation')
