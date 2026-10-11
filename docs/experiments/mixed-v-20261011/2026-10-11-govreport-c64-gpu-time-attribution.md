# GovReport c64：真实服务时间归因（2026-10-11）

正式服务的INT8退化主要在模型执行路径；新采集进一步定位到attention和混合prefill/decode批次，而不是V writer。但尚不能将退化全部归因于解量化。

## 协议与证据来源

组间独立诊断，BF16/INT8均seq256、默认10240/10240、同FlashInfer artifact、c64、正常EOS、同200题×2 trace、cap1024。每arm先2请求验证、64请求预热，重放60秒后采60秒，固定seed Bernoulli p=.1采model步骤。CUDA event记录attention、writer、linear，RAM写入后归档E。未插桩正式结果与诊断严格分开；诊断不评分，不作为正式吞吐。

原始数据：`results/flashinfer-service-profile-20261011/1791723038416109691/{auto,bf16_int8}/cuda_event_steps.jsonl`，manifest保存命令、trace SHA、hook SHA、build/artifact SHA。artifact为`7928ba9bdc69c8e439b41f45fac5093712ea5db98e9eddc145c4456e816fedcc`。BF16 SM时钟2722→2715MHz，INT8 2700→2707MHz，显存时钟均13801MHz；快照差异约1%，不能排除采样期间波动。

CUDA event是同stream区间，含CPU发射间隙；额外event及同步有开销。model与runner嵌套，不能相加；下列不是kernel busy-time或SM利用率。

## 纯decode：attention变重，writer不是主要瓶颈

| 指标 | BF16 | V-INT8 |
|---|---:|---:|
| 纯decode采样步数 | 70 | 36 |
| batch均值（范围） | 27.43（22–32） | 37.50（34–41） |
| 每步平均上下文长度 | 12260 | 11900 |
| model间隔/步 ms | 37.67 | 53.02 |
| attention间隔/步 ms | 23.94 | 41.92 |
| writer间隔/步 ms | 1.888 | 0.754 |
| linear间隔/步 ms | 7.824 | 7.547 |
| model间隔/产出token ms | 1.3734 | 1.4139 |
| attention间隔/产出token ms | 0.8728 | 1.1179 |
| writer间隔/产出token ms | 0.0688 | 0.0201 |
| linear间隔/产出token ms | 0.2852 | 0.2012 |

归一化用各区间总和/纯decode query_tokens总和，BF16/INT8分别1920/1350，不是逐步比值平均。INT8 batch增加36.7%、model每步增加40.8%，故model/token只增加约3.0%；attention/token增加28.1%，linear/token下降29.4%，writer/token下降70.8%。这定位了真实执行形状下attention效率下降，但两个batch范围没有交集，不能得到同形状codec因果开销，也不能将writer下降归因于特定kernel指令。上下文均值近似也不能保证每个请求长度分布、page布局和planner划分相同。

这也解释为何更少waiting不保证用户更快：更多请求共享每步GPU时间，每个请求拿到下一个token的间隔仍会变长。吞吐取决于batch产出能否抵消步时增长，延迟则直接承受步时与混批暂停。

## 混合批次：长暂停确实存在，时间份额仍不确定

仅采到BF16 3、INT8 8个mixed步骤，无纯prefill。BF16 query_tokens6065–10240，model476.7–864.7ms；INT8 query_tokens7775–10240，model671.3–1146.5ms。两边decode步骤几十ms，混批步骤数百至千ms，会暂停同批decode请求，影响token间隔。混批平均model/attention/linear分别为BF16 690.8/174.4/464.1ms、INT8 902.2/368.4/479.4ms；不同步query分布及样本太少，不是配对同shape对照。

不能将3/73与8/44或其累计时间直接当正式阶段份额：60秒窗口稀疏Bernoulli采样遇到长步骤时方差很大，诊断还改变发射/同步行为。更不能把约3%的纯decode model/token差异当成已经完整解释正式tok/s约−9.5%。需要更长低开销阶段计数或同shape受控算子诊断补齐剩余归因。

## 与正式全窗口和算子校准交叉验证

正式c64模型执行691.0→774.4秒，generated-work近似相同；generated-work/model-time下降约10.5%，与正式tok/s−9.5%接近。调度仅0.325/0.299秒、block分配0.076/0.075秒，不能解释这一级差异。模型调用9403→7326，平均generated-work/调用28.1→36.2（包含混批）；更多每步工作没有抵消更长的执行时间。

同artifact固定shape attention校准：batch32/context8192纯decode，INT8比BF16快9.4%；batch1/query4096/context8192 prefill，INT8慢19.0%。因此量化本身可以有decode收益，服务差异是实际shape、prefill/mixed、额外转换和其他模型成本的组合。校准用graph计时，正式/诊断为eager，不能直接替代服务结果。

代码显示V加载后转为BF16 shared tile，增加scale/转换/shared写及同步，MMA仍BF16。它是额外工作证据，不是具体耗时份额。当前数据不能独立测出融合解量化时间，不能判断哪个kernel严格bandwidth-bound，也不能证明不同shape为何与graph校准趋势不同。

## 可执行的下一步

保持正式实验配置和负面结果。下一组间空档用实际context/page布局、固定相同batch/query的eager算子对照，拆出纯codec增量；若增加低开销按phase累计计数，必须单列诊断并记录插桩开销。只有定位到同shape退化或planner策略问题后才考虑实现改进，不为追求收益改变正式配置。
