"""Check the pin-gated source transformation without initializing a device."""
from pathlib import Path
import subprocess
import sys
import tempfile

root=Path(__file__).resolve().parents[2]
original=(root/'src/kernels/vulkan/native_q8_stages.comp').read_text()
helper=root/'cmake/strata_native_quant_rcp.cmake'
old_scale='precise float scale=maxima[lane]/127.0;'
new_scale='precise float scale=maxima[lane]*uintBitsToFloat(0x3c010204u);'
old_ratio='precise float ratio=xi/d;'
new_ratio='precise float reciprocal=1.0/d;\n    precise float ratio=xi*reciprocal;'
assert original.count(old_scale)==1 and original.count(old_ratio)==1
expected=original.replace(old_scale,new_scale).replace(old_ratio,new_ratio)
with tempfile.TemporaryDirectory() as tmp:
    d=Path(tmp);driver=d/'transform.cmake'
    driver.write_text('include("${HELPER}")\nfile(READ "${INPUT}" source)\nstrata_native_quant_rcp_source("${source}" result)\nfile(WRITE "${OUTPUT}" "${result}")\n')
    def run(text,name):
        inp=d/(name+'.in');out=d/(name+'.out');inp.write_text(text)
        result=subprocess.run([sys.argv[1],'-DHELPER='+str(helper),'-DINPUT='+str(inp),'-DOUTPUT='+str(out),'-P',str(driver)],capture_output=True,text=True)
        return result,out
    result,out=run(original,'original');assert result.returncode==0,result.stderr
    assert out.read_text()==expected
    for i,bad in enumerate((original.replace(old_scale,'precise float scale=maxima[lane]/126.0;'),
                             original.replace(old_ratio,'precise float ratio=xi/(d+1.0);'),original+'\n// changed input\n')):
        result,out=run(bad,'bad'+str(i));assert result.returncode and not out.exists()
print('pin-gated candidate changes only scale and reciprocal association; mutated base sources refuse')
