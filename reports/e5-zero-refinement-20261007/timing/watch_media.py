import json
import os
from pathlib import Path
import signal
import subprocess
import time

root=Path(__file__).resolve().parent
media_pid=94518
collector_pid=3368

def ps(pid,fields):
    p=subprocess.run(['ps','-p',str(pid),'-o',fields],capture_output=True,text=True)
    return p.stdout.strip() if p.returncode==0 else ''

def record(action,**kwargs):
    with (root/'service-control.jsonl').open('a') as f:
        f.write(json.dumps(dict(action=action,time=time.time(),media_pid=media_pid,collector_pid=collector_pid,**kwargs))+'\n')
    print(action,kwargs,flush=True)

def interrupted(signum,frame):
    raise SystemExit(128+signum)

for sig in (signal.SIGTERM,signal.SIGINT,signal.SIGHUP):
    signal.signal(sig,interrupted)

owner,comm=ps(media_pid,'uid=,comm=').split(None,1)
assert int(owner)==os.getuid()
assert comm=='/System/Library/PrivateFrameworks/MediaAnalysis.framework/Versions/A/mediaanalysisd'
assert 'T' not in ps(media_pid,'stat=')
assert 'tools/e5_refinement/timing.py collect' in ps(collector_pid,'command=')
identity=ps(media_pid,'lstart=,comm=')
collector_identity=ps(collector_pid,'lstart=,command=')
stopped=False
reason='exception'
try:
    os.kill(media_pid,signal.SIGSTOP)
    stopped=True
    record('paused',identity=identity,authorization='Explicit user approval in this conversation',watchdog_seconds=1800)
    deadline=time.monotonic()+1800
    while time.monotonic()<deadline:
        if ps(collector_pid,'lstart=,command=')!=collector_identity:
            reason='collector exited'
            break
        time.sleep(1)
    else:
        reason='30-minute safeguard reached'
finally:
    if stopped and ps(media_pid,'lstart=,comm=')==identity:
        os.kill(media_pid,signal.SIGCONT)
        record('resumed',reason=reason,state=ps(media_pid,'stat='))
    elif stopped:
        record('original process no longer exists',reason=reason)
