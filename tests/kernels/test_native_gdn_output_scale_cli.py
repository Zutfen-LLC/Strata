"""Exercise the real scale wrapper and exact comparator with a fake CUDA provider.

This tests contracts and failure paths; the host provider is not a native oracle.
"""
from pathlib import Path
import hashlib, importlib.util, json, os, shutil, subprocess, sys, tempfile

root = Path(__file__).resolve().parents[2]
control, vulkan = (Path(p).resolve() for p in sys.argv[1:3])
spec = importlib.util.spec_from_file_location('provider', root / 'tests/kernels/test_native_qsa_projection_q8_native_cli.py')
provider = importlib.util.module_from_spec(spec); spec.loader.exec_module(provider)
with tempfile.TemporaryDirectory(prefix='gdn-scale-cli-') as name:
    d = Path(name); provider.build(d, replay=True)
    source = d / 'provider.cpp'; prefix = source.read_text().split('namespace strata::kernels {')[0]
    source.write_text(prefix + r'''
#include "native_q8_wo_fixture.hpp"
namespace strata::kernels {
void native_quantize_q8_1(const float* x,void* q,int n,int cols,void* stream){
 validate_shape(n,cols);validate_pointer(x);validate_pointer(q);validate_stream(stream);
 if(stream!=active || n!=6144 || cols!=1)throw std::runtime_error("quantizer shape");
 log("quant");auto bytes=native_q8_fixture::quantize(std::vector<float>(x,x+n));std::memcpy(q,bytes.data(),bytes.size());
}
void native_mmvq(int type,const void* weights,const void* x,float* y,int n,int rows,int cols,void* stream){
 if(type!=8)throw std::invalid_argument("unsupported type");
 validate_shape(n,cols);validate_pointer(weights);validate_pointer(x);validate_pointer(y);validate_stream(stream);
 if(stream!=active || n!=6144 || cols!=1 || rows!=2560)throw std::runtime_error("consumer shape");log("wo");
 using namespace native_q8_wo_fixture;native_q8_wo_fixture::Fixture f{size_t(rows),Bytes(size_t(rows)*q8_wo_row_bytes),Bytes(q8_wo_input_bytes)};
 std::memcpy(f.weights.data(),weights,f.weights.size());std::memcpy(f.input.data(),x,f.input.size());
 for(int row=0;row<rows;++row)y[row]=reference(f,size_t(row),InputScaleDomain::gdn_output_finite).value;
}}
''')
    native = d / 'native'
    subprocess.run([os.environ.get('CXX','c++'),'-std=c++17','-O2','-fno-fast-math','-ffp-contract=off',
        '-DSTRATA_GDN_OUTPUT_SCALE_SUITE=1','-I'+str(d),'-I'+str(root/'include'),'-I'+str(root/'src/core'),
        '-I'+str(root/'tests/kernels'),'-x','c++',str(root/'tests/kernels/native_qsa_wo_compose_reference.cu'),
        str(source),'-o',str(native)],check=True)
    suite=d/'suite';subprocess.run([str(control),'--scale-suite',str(suite)],check=True)
    frozen={
        'case-00.bin':'56bd646023e5fda6b049aa650bc6b969b794c00b083dfab95f48e86618e2e035',
        'case-01.bin':'beeb821930cc410fda7dc259930891d60569f1fa1f08352a05765393bc1c9b85',
        'case-02.bin':'1fd184edc129408fe83da5f95a3ea1137b296c89b8bac3fa6bf6952084d9259b',
        'case-03.bin':'7434b93e9734fd8e3ffc59083fbb68ff726d87001dd9d26ae70238fb4d94329e',
        'case-04.bin':'315e7a58e57fd1ce246804e361a33281921aef1ce2289b50314841eb044c3716',
        'case-05.bin':'db866e3289a095a1a3b4f2d60daa9701c9c71714c0399b8c2bc3265e13deb8c5',
        'case-06.bin':'905d984e99df21e867947c420628bf2f710b54815e538b44d5a636577c035b5c',
        'weights.bin':'68b08bac1aa32404b091b6b7836e57a033f53f0af812d095a2f24370399d43dd',
    }
    assert {p.name for p in suite.iterdir()} == set(frozen)
    for name,expected in frozen.items():assert hashlib.sha256((suite/name).read_bytes()).hexdigest()==expected,name
    fixture=suite/'case-00.bin';weights=suite/'weights.bin'
    for exe in (control,native,vulkan):
        out=d/(exe.name+'.json');r=subprocess.run([str(exe),'--contract',str(weights),str(fixture),str(out)],cwd='/',capture_output=True,text=True)
        assert r.returncode==0 and not r.stdout and not r.stderr,(exe,r.stderr)
        assert json.loads(out.read_text())=={'fields':{'qx.bin':6912,'out.f32le':10240,'direct-qx.bin':6912,'direct-out.f32le':10240},'qualified':False}
    tampered=d/'tampered';tampered.mkdir();changed=bytearray(fixture.read_bytes());changed[12+64*4]^=1;(tampered/fixture.name).write_bytes(changed)
    for exe in (control,native,vulkan):
        r=subprocess.run([str(exe),'--contract',str(weights),str(tampered/fixture.name),str(tampered/(exe.name+'.json'))],cwd='/',capture_output=True,text=True)
        assert r.returncode and 'fixture changed' in r.stderr,(exe,r.stderr)
    out=d/'native-out';trace=d/'trace'
    r=subprocess.run([str(native),'--device','0','--expect-device','signature-fixture GPU','--weights',str(weights),'--fixture',str(fixture),'--output',str(out)],cwd='/',env=dict(os.environ,Q8_TEST_TRACE=str(trace)),capture_output=True,text=True,timeout=120)
    assert r.returncode==0,r.stderr
    assert trace.read_text().splitlines()==['create','quant','sync','wo','sync','wo','sync','wo','sync']*1+['quant','sync','wo','sync','wo','sync','wo','sync','destroy']
    fake=d/'vulkan';shutil.copytree(out,fake);report=json.loads((fake/'report.json').read_text())
    report.pop('actual_native_quantizer_and_mmvq');report.pop('same_stream');report['actual_vulkan_quantizer_and_wo']=True;(fake/'report.json').write_text(json.dumps(report))
    spec=importlib.util.spec_from_file_location('compare',root/'tools/native_qsa_wo_compose_compare.py');compare=importlib.util.module_from_spec(spec);spec.loader.exec_module(compare)
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    args=(weights,sha(weights),fixture,sha(fixture),out,fake,control,sha(control),True)
    assert compare.compare(*args)['exact']
    altered=bytearray((fake/'direct-out.f32le').read_bytes());altered[-1]^=1;(fake/'direct-out.f32le').write_bytes(altered)
    result=compare.compare(*args);assert not result['exact'] and result['direct_different_bytes']==1
    altered=bytearray((fake/'direct-qx.bin').read_bytes());altered[0]^=1;(fake/'direct-qx.bin').write_bytes(altered)
    try:compare.compare(*args)
    except ValueError:pass
    else:raise AssertionError('changed direct input was accepted')
    if len(sys.argv)>3:
        retained=Path(sys.argv[3]).resolve();retained_out=d/'retained-host-provider'
        r=subprocess.run([str(native),'--device','0','--expect-device','signature-fixture GPU','--weights',str(weights),'--fixture',str(retained),'--output',str(retained_out)],cwd='/',env=dict(os.environ,Q8_TEST_TRACE=str(d/'retained-trace')),capture_output=True,text=True,timeout=120)
        assert r.returncode==0,r.stderr
        assert (retained_out/'fixture.bin').read_bytes()==retained.read_bytes()
print('GDN scale CLI, direct controls and complete-byte comparator pass; host provider only')
