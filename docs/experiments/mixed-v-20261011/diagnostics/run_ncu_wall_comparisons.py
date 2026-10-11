import json, os, subprocess, time
from pathlib import Path
base=Path('/mnt/e/AI/xllm-storage/reports')
env=os.environ.copy(); env.update(PYTHONDONTWRITEBYTECODE='1',TMPDIR='/mnt/xllm-e/tmp',XDG_CACHE_HOME='/mnt/xllm-e/cache',MPLCONFIGDIR='/mnt/xllm-e/cache/matplotlib')
manifest=[]
for arm,mode,batch,split in [('int8',1,36,None),('bf16',0,37,None),('int8',1,37,None),('int8',1,37,32),('bf16',0,37,32)]:
 name=f'2026-10-11-ncu-wall-{arm}-b{batch}'+(f'-split{split}' if split else '')
 stem=base/name
 argv=['/usr/local/cuda-13.2/bin/ncu','--clock-control','none','--cache-control','all']
 for section in ['SpeedOfLight','LaunchStats','Occupancy','MemoryWorkloadAnalysis','ComputeWorkloadAnalysis','WarpStateStats','SchedulerStats']: argv+=['--section',section]
 argv+=['--kernel-name-base','mangled','--kernel-name',f'regex:.*BatchPrefillWithPagedKVCacheKernel.*PrefillParamsILi{mode}.*','--launch-skip','20','--launch-count','1','--export',str(stem),'/home/mingasa/venvs/vllm/bin/python','/mnt/e/AI/xllm-storage/control/diagnose_mixed_batch_wall.py','--batches',str(batch),'--contexts','12288','--repetitions','3','--output',str(stem)+'-app.json']
 if split: argv+=['--fixed-split-pages',str(split)]
 print('START',name,flush=True)
 with open(str(stem)+'.log','w') as log: code=subprocess.run(argv,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=600).returncode
 manifest.append(dict(name=name,argv=argv,exit_code=code,completed=time.time()))
 (base/'2026-10-11-ncu-wall-manifest.json').write_text(json.dumps(manifest,indent=2))
 if code: raise RuntimeError(name+' failed')
 with open(str(stem)+'.csv','w') as out: subprocess.run(['/usr/local/cuda-13.2/bin/ncu','--import',str(stem)+'.ncu-rep','--page','raw','--csv'],stdout=out,check=True)
 print('DONE',name,flush=True)
