# Mixed-V大batch性能拐点：受控诊断（2026-10-11）

## 结论

已经观察到同shape纯decode量化attention在batch36→37附近的真实性能拐点，并通过KV拆分干预减轻。证据更支持的解释是：默认工作划分与GPU有限的线程块驻留能力相互作用，放大量化路径的长执行/尾部成本；具体执行机制仍需更细的计数器或消融确认。还不能宣称已证明某种硬件带宽、算力或同步指标饱和。

正式GovReport20/20 cells已全部完成且exit0，最终CPU评分仍在进行；诊断没有混入任何正式测量，未修改正式scheduler或仓库kernel。

## 相同shape测量

同checkpoint架构Q12/KV2/head128，query=1，均匀context12288，随机分页相同。复用现有`benchmark_flashinfer_mixed_v.py`的case/codec/kernel和BF16原版等值检查，只替换为eager CUDA-event计时与保存planner。每arm30次预热、每段20calls、随机arm顺序5轮；事件区间包含CPU发射间隙。所有arm native artifact SHA与正式完全一致。

| Batch | BF16默认 ms | INT8默认 ms | INT8相对BF16 | BF16拆分 ms | INT8拆分 ms |
|---|---:|---:|---:|---:|---:|
| 34 | 1.021 | 0.899 | -12.0% | 1.028 | 0.917 |
| 35 | 1.054 | 0.905 | -14.2% | 1.059 | 0.936 |
| 36 | 1.094 | 0.963 | -12.0% | 1.099 | 0.952 |
| 37 | 1.239 | 1.510 | +21.9% | 1.149 | 1.139 |
| 38 | 1.247 | 1.507 | +20.8% | 1.167 | 1.148 |
| 39 | 1.262 | 1.511 | +19.8% | 1.184 | 1.152 |
| 40 | 1.268 | 1.520 | +19.9% | 1.208 | 1.157 |
| 48 | 1.444 | 1.564 | +8.4% | 1.453 | 1.265 |

INT8在默认batch36→37执行间隔增加约57%，BF16增加约13%；同shape量化从约快12%变成慢22%。context8192也出现相近拐点，故不能只用工作量增加解释。batch64量化又快约11%，说明不是超过36后永久陷入容量/算力上限。

## 工作划分干预提供的证据

设备CUDA属性确认36个SM。默认q1采用tile_q16；`mixed_prefill.cu`grid为(padded_batch_size,1,2)，batch36/37不拆分时是72/74个CTA。source-derived dynamic shared memory36KiB；现有cubin CUDA driver occupancy查询四格式理论均2CTA/SM，总共72个同时驻留槽位。因此batch37越过此几何边界，与计时突变位置一致。这是静态资源上界与时间相关证据，不是已测实际调度波次/SM occupancy。

独立干预使用既有planner的fixed_split_size=32页，把12288-token上下文拆成3段4096token并merge，各arm相同。batch37 INT8约1.510→1.139ms（−24.6%），BF16约1.239→1.149ms（−7.3%），INT8对BF16从慢22%缩至约快0.8%。batch38–40也减轻。拐点未完全消失，只是明显缓和；这支持任务划分/尾部效率问题的实际贡献，但没有独立分离merge、额外launch、L2或转换成本。32页是定位用的对照参数，不是正式服务建议，未部署。

## 排除与尚未排除

现有cubin tile16 BF16/INT8/FP8/INT4每线程registers为128/168/168/168，driver local bytes均0。但理论active blocks全部2；共享内存已限制驻留，因此不能说量化寄存器增加导致此处独有的occupancy下降或register spill。实际warp active、barrier等待、DRAM throughput、L2 hit rate均没有硬件计数器数据。

Nsight Compute2026.1.1实际失败为ERR_NVGPUCTRPERM。用户选择先继续无计数器诊断，未修改Windows权限、时钟或重启。失败日志保留。WSL硬件计数器权限依赖Windows设置，见[NVIDIA官方说明](https://docs.nvidia.com/nsight-compute/ReleaseNotes/topics/system-requirements.html)。

## 与正式服务的关系

c64实际纯decode BF16 batch22–32、INT8 batch34–41，恰好两边主要处于上述边界不同侧。正式独立诊断attention/step23.94→41.92ms；均匀context12288的每层BF16 batch29约.88ms、INT8 batch38约1.51ms，乘28层约24.6/42.2ms，与事件量级接近。这是强的一致性证据，但不是精确因果闭合：真实context/page布局异质、各层CPU间隙及batch变化，不能拿两个均匀点代替正式全程。

因此前文“batch更大所以每轮更长”需要补充：在当前kernel/planner下，量化服务跨越了一个可复现、可被工作划分干预缓和的性能拐点。它不是显存放不下，更不是必须为量化收益接受的固定成本；是否通过改进planner获得保质服务加速，仍需正确性和完整服务协议重新验证。prefill/mixed造成的停顿是另一项成本，不因此消失。

## 原始材料

`2026-10-11-mixed-wall-eager-sweep.json`、`2026-10-11-mixed-wall-knee-default.json`、`2026-10-11-mixed-wall-knee-split32.json`保存全部5轮原始值、planner、时钟快照、artifact及script SHA；source副本均在同目录。`2026-10-11-mixed-wall-resources.txt`、`2026-10-11-mixed-wall-occupancy.json`保存静态资源和理论occupancy，cubin提取未改原artifact。图为`2026-10-11-mixed-wall-knee.png`/PDF。


## 2026-10-11：实际硬件计数器补充

NVIDIA App权限开启后，六个单kernel Nsight Compute测量全部成功。INT8 b36→37时DRAM利用率83%→55%，跨SM活跃周期均值/最大值98%→62%，没有spill或驻留块数改变；固定KV分片主kernel减少24%时间。进一步支持CTA工作划分尾部是主要拐点因素。详情与边界见[硬件归因报告](2026-10-11-nsight-compute-mixed-v-wall-analysis.md)，正式服务结果保持不变。
