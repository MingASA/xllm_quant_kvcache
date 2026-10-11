# GovReport混合轮次与调度判断（2026-10-11，阶段性）

正式参数保持不变，INT4 c128继续。结论：当前观察支持默认调度策略与当前算子实现之间的吞吐/延迟取舍，尚未证明调度错误，也不能称量化退化为不可改善的必然结果。

## 实际发生了什么

独立CUDA-event诊断BF16/INT8 c64纯decode每轮37.7/53.0ms。mixed仅3/8个样本，每轮476.7–864.7 / 671.3–1146.5ms；每轮BF16含25–30个decode序列及1个prefill，INT8含33–38个decode及1–2个prefill。prefill token分别6035–10215 / 7741–10205。少量长prompt工作可以让同批数十个decode请求等整轮执行。

mixed与纯decode的阶段均值差（非同shape配对，单位ms）：

| 阶段差值 | BF16 | INT8 |
|---|---:|---:|
| model | 653.1 | 849.2 |
| linear | 456.3 | 471.8 |
| attention | 150.5 | 326.5 |
| writer | 0.28 | 1.52 |

观测到的长轮次包含大量线性层工作，并非单独attention或V写入造成。线性层处理整批query rows，新增数千prefill token的投影与MLP工作不因V Cache压缩而减少；这属于真实新增计算。INT8混批attention间隔也更大，但batch、context、query布局未配对，不能将额外176ms称为codec净成本。

CUDA event区间含CPU发射空洞；n=3/8不足以估计正式prefill时间份额。计时和样本来源见`2026-10-11-govreport-c64-gpu-time-attribution.md`。正式全窗口prompt工作约416万token、generated工作约26.5万token；纯decode局部约3%单位token退化尚不能完整解释正式tok/s−9.5%。

## 当前实际策略（源码链）

- `scheduler_policy.cpp:1050`工厂：enable_mix_batch且普通fcfs时选择DecodeFirstPolicy。
- `decode_first_policy.cpp:54–89`：先安排decode，再安排未完成prefill，再安排新prefill，最后将剩余token预算重新分配给prefill。decode优先保护入批资格，不保证输出先返回。
- `continuous_scheduler.cpp:507–529`：默认schedule_overlap=false，engine step之后才统一处理batch output。
- `scheduler_config.cpp`默认mix=true、chunked_prefill=true、max_tokens_per_batch=10240；`xllm.cpp:476`把未指定的chunk上限设置为batch预算，故有效chunk上限10240。
- `profile_config.cpp`默认enable_latency_aware_schedule=false；预算依据token数，不依据本轮实际执行时间。因此相同10240 token不意味着不同query/context/batch/V格式执行时间相同。
- `decode_first_policy.cpp:70–89`、`scheduler_policy.cpp:279–300`新prefill admission用1.1×已用块+完整prompt footprint；同时存在95%内存利用率短路。1.1预留与完整prompt检查可解释约90%时停止接纳新请求、free blocks仍非零、preemption仍为零。没有拒绝原因计数，不能逐次确认唯一触发条件。

## 问题、取舍与必然成本怎么区分

**必需计算**：正常完整prompt的prefill和生成所需decode都必须执行。V压缩不会减少Q/K计算或MLP工作，也不会让融合反量化成本消失。

**策略选择**：将最多近10240的剩余query token集中加入同一轮，减少轮次数、提高大矩阵效率，同时延长decode请求等待；这是吞吐倾向的策略。分更小轮次或时间预算可以改变停顿分布，但增加发射/调度次数，未必提高吞吐。不能只看TPOT变好就宣称整体优化。

**当前实现的适配限制**：token预算未启用dtype/shape执行时间预测；压缩后接纳更多请求，批次扩大，其收益是否抵过步时增加取决于算子效率。FlashInfer路径decode也走FA2 prefill planner，不是独立最优decode路径；这是独立算子/规划问题，不能都算在scheduler头上。

**尚未证明的bug**：没有观察到漏调decode、超序列限制、抢占震荡或明显调度耗时。源码里remaining-budget redistribution可把prefill allocation扩大到剩余总预算，没有再次套compute_prefill_tokens中的chunk cap；默认chunk=total时不构成这次超预算证据。如果以后独立测试更小chunk，这一点必须先核对，不能以为改一个chunk flag就已经限制了轮次。

## 如何补齐因果证据

先完成下一组四arm正式对比和质量，保留所有负面结果。继续从已有ITL滚动时序检查长暂停是否真实出现在未插桩服务；rolling平均/p99不是全程分位数，request TPOT也是整段平均后的请求分位数，不能混用。

下一次独立诊断需在相同query/context/page形状、相同batch下测eager attention，再用按phase累计轮次数与执行时间估计mixed时间份额。要认定调度改善，需要另立调度消融：同trace/格式/预算之外配置受控，测总完成时间、tok/s及尾延迟，并验证chunk限制真正生效。正式现有实验不改参数，不用调度改进掩盖量化负面结论。

## 未插桩ITL补充

bvar LatencyRecorder使用默认10秒滚动窗口；约1.05秒采样，窗口高度重叠。正式BF16/INT8窗口最大token gap分别1.822/1.917秒，长间隔是真实服务现象。时序图为`2026-10-11-govreport-c64-rolling-itl.png`（另存PDF），源摘要为`2026-10-11-govreport-c64-inter-token-bvar-timeseries.json`。rolling latency是平均值而非p50；不能把窗口的p99再当整个实验p99，也不能按重复窗口max计数停顿次数。窗口时序只能支持长停顿存在，未记录逐step phase，不能直接证明每次由prefill造成。

### 窗口审计修正

两正式c64 cell实际各预热64请求（run_config确认），telemetry从measurement_start开始抓取，但首10秒bvar窗含预热；图已排除两边首10个跨界样本与各2个估计结束后的边界样本。源审计为`2026-10-11-govreport-c64-telemetry-window-audit.json`。末端wall-clock与monotonic elapsed有约0.5–1.4秒转换偏差，因此尾端仅按估计边界筛选，不做毫秒级phase关联。吞吐及请求级延迟依旧采用原始monotonic计时，未重算/改写正式summary。

## 下一组c128正式结果验证

四arm各400请求完整。INT8/FP8 tok/s333.6/333.3（BF16 371.9），TPOT p95 133.0/130.3ms（BF16 89.3），从c32到c128没有吞吐平台上移。c32较低TTFT收益在高负载消失，说明更多running、较少waiting不足以补偿服务速率下降。INT4请求完成总时间700.0 vs718.1秒、median E2E210.17 vs224.16秒改善，但输出短15.4%、p95 TPOT164.2ms、p95 E2E288.29秒更差，质量待补。不能说所有延迟都变差，也不能说已经有保质加速。

观测running peak最高54，复用BF16 cap128并未直接约束本组；仍保留配置差异，不用它武断解释退化。源码/计时支持“默认token预算的大混批停顿与实际算子效率”机制，尚无漏调/超限/抢占震荡证据。c128独立判断见`2026-10-11-govreport-c128-four-arm-service-analysis.md`。正式继续c8/c16。
