"""Configure the actual W4A4 target block and inspect generated compile commands.
No GPU compiler is substituted for a claimed compile: this is a host CMake test.
"""
from pathlib import Path
import json,tempfile,subprocess,sys,shutil
source=Path(sys.argv[1]).resolve() if len(sys.argv)>1 else Path(__file__).resolve().parents[1]
cmake=sys.argv[2] if len(sys.argv)>2 else shutil.which('cmake');assert cmake and Path(cmake).is_file(),'recorded CMake executable required (optional second argument)'
s=(source/'CMakeLists.txt').read_text();start=s.index('    set(_strata_w4a4_archs ');end=s.index('    # STRATA_MMQ_TAB:',start);block=s[start:end]
with tempfile.TemporaryDirectory(prefix='hip-w4a4-cmake-') as tmp:
 root=Path(tmp);(root/'src/prefill').mkdir(parents=True);(root/'src/prefill/mmq_nvfp4_w4a4.cu').write_text('int w4a4_fixture;\n');(root/'other.cpp').write_text('int unrelated_fixture;\n')
 (root/'CMakeLists.txt').write_text('cmake_minimum_required(VERSION 3.20)\nproject(vendor_definition_test LANGUAGES CXX)\nset(CMAKE_EXPORT_COMPILE_COMMANDS ON)\nset(CMAKE_CUDA_ARCHITECTURES 86)\nset_source_files_properties(src/prefill/mmq_nvfp4_w4a4.cu PROPERTIES LANGUAGE CXX)\nadd_library(ggml-base INTERFACE)\nadd_library(strata_mmq STATIC other.cpp)\n'+block+'\n')
 for hip in (False,True):
  build=root/str(hip);subprocess.run([cmake,'-S',str(root),'-B',str(build),'-DSTRATA_HIP_GFX906='+('ON' if hip else 'OFF')],check=True,capture_output=True,text=True)
  rows=json.loads((build/'compile_commands.json').read_text());actual=next(r['command'] for r in rows if r['file'].endswith('mmq_nvfp4_w4a4.cu'));unrelated=next(r['command'] for r in rows if r['file'].endswith('other.cpp'))
  assert ('-DGGML_USE_HIP' in actual)==hip,actual
  assert 'GGML_USE_HIP' not in unrelated,'definition leaked beyond W4A4 target'
 print('PASS actual target CMake block generates HIP-only vendor definition; CUDA and unrelated target unchanged; no GPU compile')
