# Copyright 2026 The xLLM Authors. Licensed under the Apache License, Version 2.0.
"""Query static attributes/occupancy of extracted existing kernels; no new kernel."""
import ctypes as c
import hashlib
import json
from pathlib import Path
import re

root=Path('/mnt/e/AI/xllm-storage/reports')
lib=c.CDLL('/usr/lib/wsl/lib/libcuda.so.1')
def check(result):
    if result: raise RuntimeError('CUDA driver error '+str(result))
check(lib.cuInit(0))
ctx=c.c_void_p();check(lib.cuDevicePrimaryCtxRetain(c.byref(ctx),0));check(lib.cuCtxSetCurrent(ctx))
module=c.c_void_p();file=root/'mixed_prefill.2.sm_120.cubin';check(lib.cuModuleLoad(c.byref(module),str(file).encode()))
resources=(root/'2026-10-11-mixed-wall-resources.txt').read_text()
rows=[]
for name,attrs in re.findall(r'Function (\S+):\n\s+([^\n]+)',resources):
    if 'BatchPrefillWithPagedKVCacheKernel' not in name: continue
    tile=int(re.search(r'MaskModeE1ELj(\d+)',name).group(1));mode=int(re.search(r'PrefillParamsILi(\d+)',name).group(1))
    function=c.c_void_p();check(lib.cuModuleGetFunction(c.byref(function),module,name.encode()))
    values={}
    for label,attribute in [('registers',4),('static_shared_bytes',1),('local_bytes',3)]:
        v=c.c_int();check(lib.cuFuncGetAttribute(c.byref(v),attribute,function));values[label]=v.value
    # Existing traits/head128/BF16: union size is max(Q+K+V, warp-sync storage).
    dynamic_shared=36864 if tile==16 else 49152
    check(lib.cuFuncSetAttribute(function,8,dynamic_shared))
    blocks=c.c_int();check(lib.cuOccupancyMaxActiveBlocksPerMultiprocessor(c.byref(blocks),function,128,c.c_size_t(dynamic_shared)))
    rows.append(dict(tile_q=tile,mode=mode,threads_per_block=128,dynamic_shared_bytes=dynamic_shared,theoretical_active_blocks_per_sm=blocks.value,**values))
result={'scope':'static resource-based theoretical occupancy, not achieved occupancy or hardware utilization', 'cubin_sha256':hashlib.sha256(file.read_bytes()).hexdigest(),'dynamic_shared_method':'sizeof existing BF16 traits union: tile16 36864; tile64/128 49152; source-derived, unchanged all modes','kernels':rows}
(root/'2026-10-11-mixed-wall-occupancy.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
check(lib.cuModuleUnload(module));check(lib.cuDevicePrimaryCtxRelease(0))
