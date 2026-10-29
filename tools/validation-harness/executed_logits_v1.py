"""Validate explicitly sparse executed-head dumps and reuse compare_logits' fixed gates."""
import argparse
import json
from pathlib import Path
import compare_logits as C


def load(directory, chunk_size):
    if type(chunk_size) is not int or chunk_size not in (0,2,3,4,5,6,7,8):
        raise ValueError('invalid expected chunk policy')
    root=Path(directory);path=root/'manifest.json'
    if path.stat().st_size>1<<20:raise ValueError('manifest exceeds bound')
    def unique(pairs):
        result={}
        for key,value in pairs:
            if key in result:raise ValueError('duplicate manifest field')
            result[key]=value
        return result
    manifest=json.loads(path.read_text(),object_pairs_hook=unique)
    if set(manifest)!={'schema','format','head_policy','chunk_policy','prompt_chunk_size','vocabulary','prompt_tokens','output_count','stop','frames'}:
        raise ValueError('unknown/missing manifest fields')
    if (type(manifest['schema']) is not int or manifest['schema']!=1 or
        manifest['format']!='executed_heads_f32le_v1' or manifest['head_policy']!='prompt_head_elision' or
        type(manifest['prompt_chunk_size']) is not int or manifest['prompt_chunk_size']!=chunk_size or
        manifest['chunk_policy']!=('layer_major_v1' if chunk_size else 'serial_t1')):
        raise ValueError('executed-head policy mismatch')
    vocabulary=manifest['vocabulary'];prompt=manifest['prompt_tokens'];count=manifest['output_count'];frames=manifest['frames']
    if (type(vocabulary) is not int or not 0<vocabulary<=C.MAX_BYTES//4 or not isinstance(prompt,list) or not prompt or
        any(type(t) is not int or not 0<=t<vocabulary for t in prompt) or type(count) is not int or count<0 or
        len(prompt)+count>256 or not isinstance(frames,list) or len(frames)!=count+1 or manifest['stop'] not in ('eos','max_new','context')):
        raise ValueError('invalid executed-head shape')
    rows=[];files={'manifest.json'}
    for index,frame in enumerate(frames):
        position=len(prompt)-1+index;name=f'pos-{position}.f32'
        role='final_prompt' if index==0 else 'terminal' if index==count else 'generated'
        if (not isinstance(frame,dict) or set(frame)!={'position','token','role','file'} or
            type(frame['position']) is not int or frame['position']!=position or frame['file']!=name or
            frame['role']!=role or type(frame['token']) is not int or not 0<=frame['token']<vocabulary or
            (index==0 and frame['token']!=prompt[-1])):
            raise ValueError('missing/extra/duplicate or malformed executed frame')
        row=C.read(root/name)
        if len(row)!=vocabulary:raise ValueError('executed vocabulary mismatch')
        if index and frame['token']!=max(range(vocabulary),key=lambda i:rows[-1][i]):
            raise ValueError('executed token is not prior greedy prediction')
        files.add(name);rows.append(row)
    if {p.name for p in root.iterdir()}!=files:raise ValueError('missing/extra executed dump file')
    return manifest,rows


def compare(baseline,candidate,baseline_chunk_size=0,candidate_chunk_size=2):
    a,arows=load(baseline,baseline_chunk_size);b,brows=load(candidate,candidate_chunk_size)
    if any(a[key]!=b[key] for key in ('vocabulary','prompt_tokens','output_count','stop','frames')):
        raise ValueError('executed positions/tokens/roles differ between policies')
    frames=[dict(frame,**C.compare(x,y)) for frame,x,y in zip(a['frames'],arows,brows)]
    return dict(limits=C.LIMITS,baseline_chunk_size=baseline_chunk_size,candidate_chunk_size=candidate_chunk_size,
                omitted_positions='not stored; never zero-filled',frames=frames,numeric_pass=all(f['numeric_pass'] for f in frames))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline',required=True);parser.add_argument('--candidate',required=True)
    parser.add_argument('--baseline-chunk-size',type=int,default=0);parser.add_argument('--candidate-chunk-size',type=int,default=2)
    args=parser.parse_args()
    result=compare(args.baseline,args.candidate,args.baseline_chunk_size,args.candidate_chunk_size)
    print(json.dumps(result,indent=2,allow_nan=False))
    return 0 if result['numeric_pass'] else 1


if __name__=='__main__':raise SystemExit(main())
