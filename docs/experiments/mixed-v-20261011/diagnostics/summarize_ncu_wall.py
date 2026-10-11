import csv,json,subprocess
from pathlib import Path
base=Path('/mnt/e/AI/xllm-storage/reports')
metrics=['gpu__time_duration.sum','dram__bytes.sum.per_second','gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed','sm__throughput.avg.pct_of_peak_sustained_elapsed','sm__warps_active.avg.pct_of_peak_sustained_active','sm__cycles_active.avg','sm__cycles_active.min','sm__cycles_active.max','gpc__cycles_elapsed.avg','gpc__cycles_elapsed.avg.per_second','smsp__warps_eligible.avg.per_cycle_active','smsp__issue_active.avg.pct_of_peak_sustained_active','smsp__average_warps_active_per_inst_executed.ratio','smsp__average_warps_issue_stalled_barrier_per_issue_active.ratio','smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio','smsp__average_warps_issue_stalled_short_scoreboard_per_issue_active.ratio','launch__waves_per_multiprocessor','launch__grid_size','launch__registers_per_thread','launch__occupancy_limit_shared_mem','launch__occupancy_limit_registers','launch__shared_mem_per_block']
rows=[]
for p in sorted(base.glob('2026-10-11-ncu-wall-*.csv')):
 with p.open() as f:
  reader=csv.DictReader(f); raw=list(reader)
  assert len(raw)==2,(p,len(raw))
  units,data=raw
 values={k:float(data[k].replace(',','')) for k in metrics}
 duration_ms=values['gpu__time_duration.sum']*{'ms':1,'us':.001,'ns':.000001}[units['gpu__time_duration.sum']]
 rows.append(dict(name=p.stem,metrics={k:dict(value=values[k],unit=units[k]) for k in metrics},derived=dict(duration_ms=duration_ms,dram_transferred_MB_estimate=values['dram__bytes.sum.per_second']*duration_ms,sm_active_mean_over_max=values['sm__cycles_active.avg']/values['sm__cycles_active.max'],sm_active_mean_over_gpc_elapsed_approx=values['sm__cycles_active.avg']/values['gpc__cycles_elapsed.avg'])))
 detail=p.with_name(p.stem+'-details.txt')
 if not detail.exists():
  with detail.open('w') as out:subprocess.run(['/usr/local/cuda-13.2/bin/ncu','--import',str(p.with_suffix('.ncu-rep')),'--page','details'],stdout=out,check=True)
 print(p.stem,round(duration_ms,3),round(rows[-1]['derived']['dram_transferred_MB_estimate'],1),round(rows[-1]['derived']['sm_active_mean_over_max']*100,1))
(base/'2026-10-11-ncu-wall-compact.json').write_text(json.dumps(dict(scope='six individual diagnostic attention kernels; not formal service timings; split cases exclude merge kernel',clock_control='none',cache_control='all',rows=rows),indent=2))
