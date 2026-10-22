"""One bounded CPU-only replay over retained captures; no engine/worker modes."""
import argparse
import json
from pathlib import Path
import sys
import time

TAG="GPU-LAYER-GROUPING-V1-20261022"
SUPERVISOR="311561689bb4b1f8a109821601e3a6bda07067e6"


def main():
    entered=time.monotonic()
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--supervisor-root",required=True)
    parser.add_argument("--replay-config",required=True)
    args=parser.parse_args()
    with open(args.replay_config,"rb") as source:raw=source.read(1048577)
    if not 0<len(raw)<=1048576:raise ValueError("bounded private offline config")
    config=json.loads(raw)
    if config.get("diagnostic")!=TAG or config.get("supervisor_commit")!=SUPERVISOR or \
            config.get("offline_only") is not True or config.get("role")!="ROLE_CPU_C" or \
            config.get("approved_window_s")!=120 or not isinstance(config.get("diagnostic_revision"),str) or \
            len(config["diagnostic_revision"])!=40:
        raise ValueError("exact offline-only replay binding required")
    root=Path(args.supervisor_root).resolve(strict=True)
    sys.path[:0]=[str(root),str(root.parent),str(Path(__file__).resolve().parent)]
    import agent,policy
    from combined_diagnostic import replay
    policy.POLICIES["ROLE_REPLAY_C"]=policy.Policy("ROLE_REPLAY_C",8704*policy.MIB,8*policy.GIB,512*policy.MIB,512*policy.MIB)
    result=replay(agent,config,120,entered=entered,argv_count=7,cap=120)
    result.update(diagnostic=TAG,offline_only=True,new_GEN=0,new_tokens=0)
    print(json.dumps(result,allow_nan=False))


if __name__=="__main__":main()
