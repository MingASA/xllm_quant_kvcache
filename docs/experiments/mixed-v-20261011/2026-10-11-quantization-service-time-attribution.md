# 量化服务变慢：时间归因（阶段性，2026-10-11）

## 已验证的证据

GovReport c64：BF16→V-INT8，generated tok/s 367.9→332.8（−9.5%），p95 TPOT 89.0→130.8 ms，E2E 146.71→178.15 s；平均输出长度662.3→664.4，生成长度不足以解释差异。运行请求峰值37→46，waiting平均32.2→23.9。服务释放了排队压力，但每步执行变慢。

已完成服务计数窗口的模型执行累计691.0/774.4秒；除以eager次数得到粗略73.5/105.7 ms每步。调度0.325/0.299秒，分配0.076/0.074秒。主要差异位于模型执行路径。窗口不完全重合，且engine/worker/model区间嵌套，不能相加，也不能将粗略均值当每token精确耗时。

## 已有同形状GPU算子实测，而非解量化猜测

复核 `results/flashinfer-mixed-v2-20261010/kernel_calibration_pdl_off.json`。与正式实验同build和artifact；固定batch、context、query长度，随机arm计时顺序，热身后CUDA event测attention。GPU计时采用graph replay减少CPU发射噪声；正式服务仍为eager，因此这些数值只能解释算子，不能代替服务TPOT。

| Decode形状（batch/context） | BF16 μs | V-INT8 μs | V-FP8 μs | V-INT4 μs |
|---|---:|---:|---:|---:|
| 8/2048 | 16.03 | 29.15 | 28.34 | 28.69 |
| 32/8192 | 637.30 | 577.70 | 573.55 | 551.99 |
| 128/8192 | 2540.69 | 2294.87 | 2278.97 | 2186.26 |

结论：量化attention并非普遍更慢。长上下文batch32时INT8约快9.4%、INT4约快13.4%；小工作量时量化路径可以明显更慢。因此不能直接用“服务变慢=解量化kernel慢”解释GovReport。

同一份计时也测到prefill明显变慢：batch1/query4096/context8192，BF16为3694.5 μs，INT8为4395.8 μs（+19.0%），FP8为4369.0 μs（+18.3%），INT4为4277.8 μs（+15.8%）；batch8/query128/context2048，量化约+23%～26%。这提供了prefill成本增加的直接同形状证据，但真实GovReport中prefill占总GPU时间多少仍待时间线确认。

## 需要区分的因果链

1. 压缩允许同时执行更多长上下文请求；实际服务batch和所读上下文总量增大，QK及其他模型计算也增大，K仍BF16。相同shape单次attention加速，不保证扩大batch后每步仍快，或每步完成的decode token增长足以补偿耗时。
2. 长prompt的prefill与decode混批。即便输出更长，完整请求仍包含prefill；需要按步分类，测其时间份额，不能只看平均输出长度。
3. Cache writer的scale计算/编码、planner与Python/CUDA发射、同步等待也是成本。GPU时间线应分别汇总attention、writer、GEMM、其他kernel，区分CPU空洞与GPU忙；不得把CUDA同步API阻塞时间额外加在GPU时间之上。
4. 当前后端所有arm都使用FA2 BatchPrefill planner/mixed_prefill入口，decode也走这一入口；是公平的自定义FlashInfer实现，不等同于已使用FlashInfer最优独立decode策略。它如何影响小query性能需独立检验，正式配置暂不改。

这些机制目前只有第1项的服务batch变化和同形状算子测量有直接证据；prefill份额、writer占比、memory-bound与融合解量化代价尚未完成归因。

## 当前profiling有效性

首份BF16 Nsight仅有CPU/CUDA API和嵌套Python NVTX，无GPU kernel表，因此不能用于GPU耗时占比。SQLite诊断显示使用hardware tracing，并提示driver13.4超出collector13.2支持范围；尚不能唯一确定缺数据原因。已切换本地支持的cuda-sw并开启周期flush；只有实际GPU kernel事件数>0才接收采集成功。下一次在完整正式cell之后独立配对BF16/INT8，正式计分和吞吐不使用诊断请求。

官方说明：虚拟化环境或不受支持GPU/driver可强制软件CUDA trace：https://docs.nvidia.com/nsight-systems/UserGuide/index.html 。

最终归因必须给出“相同shape量化开销”和“真实服务工作量变化”两层证据；融合attention中的解量化不能从单个kernel总时长独立分离，若需要进一步拆分，使用独立消融/指令指标，而不编造时间。

## 服务计数补充：更大的每步工作量有直接证据

c64 BF16/INT8已保存telemetry窗口：prompt阶段token工作4,160,000/4,161,460，generated阶段264,477/265,345，工作量接近；eager调用9,403/7,326（−22.1%）。generated token工作除以调用数为28.1/36.2，模型时间除以调用数为73.5/105.7 ms。说明INT8平均每次调用承担更多token工作，不能直接将每步+44%全归因于量化算子。两个计数包含混批且采样边界不同，28.1/36.2不是严格纯decode batch均值。现有model时间无prefill/decode标签，阶段GPU份额仍待采集。详细口径见同目录 `2026-10-11-govreport-c64-prefill-decode-counters.md`。

## 代码层面的明确额外工作（尚非耗时占比）

`prefill.cuh:868` 的 `decode_mixed_v_shared_tile`：每8个V元素读scale，INT8转换/FP8转换或INT4拆nibble，乘scale，再舍入为BF16，写回shared-memory tile，并执行额外`__syncthreads()`。MMA仍BF16，shared tile仍为BF16大小，launch的shared storage也是共同BF16 Traits。真实压缩常驻VRAM成立，但没有因此获得低精度MMA吞吐或更小BF16工作tile；收益主要来自全局V读取减少，同时付出转换、shared写和同步。这能解释为什么不同shape存在盈亏分界，但不能直接证明这些指令占服务时间多少。量化prefill实测变慢是证据，具体由转换、同步或访存哪项主导仍需硬件指标/消融。

## 避免错误归因

不能仅凭“GSM8K上下文短”断定attention量化更慢：已有batch128/context512同形状decode计时反而INT8快13%、INT4快36%，而batch32/context512量化慢约81%～88%。性能同时依赖batch、query长度、context、tile/分区策略与实际token工作量。分析GSM8K同样需要看服务实际shape及attention在模型时间中的占比。

INT4还改变了生成长度与质量：GovReport c32平均输出558.4（BF16为662.1），ROUGE-L30.77（BF16为32.04）。其generated tok/s受输出变短影响，不能直接用其下降16.4%作为kernel变慢幅度；但E2E和TPOT仍变差，需要保留该服务结论。

## 为什么decode更快仍可能整体更慢

仅用两个已测校准点作示意：INT8长上下文decode的时间比为0.906，长query prefill为1.190。若忽略writer和其他工作，并假定这两个时间比能代表各阶段，则attention总时间比为 `0.906*(1-f)+1.190*f`，其中f是BF16 attention时间中prefill份额；f超过约33%即不再有attention总时间收益。这不是GovReport的实际份额或最终归因，只说明“长输出”并不足以保证整体收益。真实服务还必须纳入工作量扩大、GEMM/writer及调度等待；下一步直接测f及各部分，不能靠输入/输出token数量推导时间份额。

CUDA-event替代诊断已准备：固定种子Bernoulli采样约10% model调用（避免固定步距与调度周期发生别名），同时记录batch/query lengths、纯decode/prefill/mixed标签，以及attention/writer/linear/runner区间。GPU event测的是流上时间区间，可能包含CPU发射空洞；同步及event本身有插桩开销，嵌套model/runner不可相加。它用于同协议配对定位，正式吞吐和评分仍只取未插桩数据。原仓库和正式server参数未改，所有hook/executor来源SHA保存。

## 可以量化到哪一步：服务退化与模型路径的数值闭合

由同一对telemetry窗口，generated-work/model-time为264477/690.980≈382.8和265345/774.409≈342.6 token/model-second，约−10.5%，与正式服务tok/s −9.5%接近。等价地，每调用generated-work从28.1增至36.2（+28.8%），但每调用model耗时从73.5升至105.7 ms（+43.8%）；产出增长赶不上执行时间增长，因此模型路径token效率下降。这是直接计数关系，不是容量推演，也不是证明所有43.8%来自量化转换。窗口和混批口径差异使它只能近似解释服务退化，不能当严格逐请求时间分解。

### 诊断状态更新

首个CUDA-event启动因Python导入优先级加载了原executor，hook-ready验收失败，没有产生有效event数据，正式已恢复FP8 c128。已改诊断cwd/PYTHONPATH；纯CPU `find_spec`确认选中了隔离executor。下一尝试先以2个正常EOS请求检查实际event非空、attention区间为正，再进入长诊断。失败原始manifest/log保留，不将启动失败或空采集写作GPU时间结果。

## 已记录的硬件活动变化（c64正式窗口）

| Arm | GPU utilization均值 | memory utilization均值 | power均值W |
|---|---:|---:|---:|
| BF16 | 92.21% | 49.03% | 119.43 |
| V-INT8 | 94.20% | 38.98% | 111.48 |
| V-FP8 | 94.03% | 40.08% | 111.87 |
| V-INT4 | 94.71% | 36.34% | 116.47 |

来源是各正式c64 `telemetry.jsonl` 的nvidia-smi样本；GPU利用率较高，显存读写活动比例反而下降，但tok/s更低。说明降低显存活动并不自动转化为更高服务吞吐；与计算/转换/同步成本及真实batch变化抵消收益的解释相容。这些指标不能证明具体kernel是compute-bound或memory-bound。GPU utilization也是有kernel执行的时间比例，不是SM occupancy或达到峰值算力的百分比。memory utilization是采样期内显存读写活跃时间比例，不是GB/s或相对于峰值带宽的利用率（[NVIDIA NVML定义](https://docs.nvidia.com/deploy/archive/R550/nvml-api/structnvmlUtilization__t.html)）。没有记录SM/memory clocks，不能排除DVFS对同形状计时/服务差异的影响；诊断会记录时钟快照，不锁时钟或修改正式参数。

## c128 FP8交叉验证与有效事件采集

FP8 c128完成400请求，BF16→FP8 tok/s 371.9→333.3（−10.4%），TPOT p95 89.3→130.3ms（+45.8%），E2E p95 264.77→303.27s（+14.5%）；输出667.7→679.3（+1.7%）。输出增长不足以解释整体变慢。running均值28.58→37.49，waiting均值84.47→77.66，但TTFT p95仍192.48→204.53s：队列请求数量降低不能等同等待时间降低。来源：`2026-10-11-govreport-c128-fp8-vs-reused-bf16.md`，复用baseline配置差异仍保留。

修正Python overlay路径后，诊断1791723038416109691实际hook ready指向E盘overlay，2请求正常EOS检查取得正model、attention、writer CUDA event时间。检查数据只证明采集链路有效，不作为高负载耗时比例。正在组间空档做BF16/INT8 c64正常预热及配对采集；CUDA event是流上的时间间隔，含CPU发射间隙，不等于kernel busy-time。正式实验不包含这些采样和同步。

## 实际配对CUDA-event诊断完成

诊断1791723038416109691两arm通过，正式INT4 c128自动恢复。纯decode BF/INT8每步model37.67/53.02ms、attention23.94/41.92ms、writer1.888/.754ms；batch27.43/37.50且范围完全不重叠。model/产出token1.3734/1.4139ms（约+3%），attention/token.8728/1.1179（约+28%）。writer不是本次采样中的主因。混批仅3/8样本，model数百至千ms，有长暂停但不能据此估计正式阶段份额或完整解释吞吐−9.5%。完整独立分析见`2026-10-11-govreport-c64-gpu-time-attribution.md`。区间含CPU发射间隙，不是kernel busy时间，融合解量化耗时尚未独立测出。

## 后续同shape诊断找到拐点（不能再只用batch变大解释）

正式20/20组完成后做独立eager attention扫描，同artifact、相同context12288/query1。INT8 batch36→37为.963→1.510ms，BF16为1.094→1.239ms。GPU36SM、现有kernel理论2CTA/SM、grid2B，对应72→74个CTA的驻留边界。强制既有planner把context拆3段时，batch37 INT8变1.139ms、BF16 1.149ms，量化劣势明显缓和。证据支持默认工作划分/尾部执行效率对退化的贡献；不是显存放不下，也不能解释成普遍量化更慢。寄存器128→168但各mode理论驻留均2，无INT8独有occupancy下降。硬件计数器访问被驱动拒绝，用户选择不改权限，因此仍不能给出实测DRAM/SM/barrier饱和结论。完整数据、图与限制见`2026-10-11-mixed-batch-wall-analysis.md`。正式参数和原始结果未改。
