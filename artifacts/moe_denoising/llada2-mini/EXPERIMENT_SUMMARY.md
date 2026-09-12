# LLaDA2.0-mini 单 block MoE 动机实验汇总

更新日期：2026-09-12。本文汇总 Mac 上 GSM8K/HumanEval × batch 8/16 的四组原始轨迹及离线复核，是当前结论的统一入口。

## 核心结果

- Batch 从 8 增至 16，两个任务的 active experts 都增加约 20，effective experts 仅增加约 2。支持此范围内有效工作集增长缓慢，尚不能确定严格饱和点。
- 相邻步完整排序的 Oracle 收益恢复率：GSM8K 97.36% → 98.42%，HumanEval 96.60% → 97.70%。
- 后续路由比第零步的等 assignment 数下采样更分散，同时负载排序仍高度可预测。不能表述为“越去噪越集中”。
- 证据范围是单个 32-token block 内未解析 mask Query 的路由，尚未证明多 block 稳定性或真实加速。
- 下一项是单层专家 replay，检验具体 MoE 执行实现能否从排序中获得收益。

## 实验设置与统计流程

| 配置 | 取值 |
|---|---|
| 模型 | LLaDA2.0-mini，256 experts，Top-8，记录 19 个 MoE 层 |
| 数据 | GSM8K / HumanEval，各配置 32 prompts |
| Batch | 8：4 groups；16：2 groups |
| Prompt | chat template 后过滤长度超过 128 的样本，并非截断 |
| 生成 | 1 个 32-token block，最多 32 次 denoising forward |
| 解码 | confidence > 0.95 或最低 transfer quota；temperature 0 |
| 执行 | HF Accelerate 双卡层切分，use_cache=False，未启用 FOCUS |
| 统计对象 | 当前生成 block 在 forward 开始时仍为 mask 的位置 |

每组在 prompt 后追加全 mask block。每一步完整执行模型主体，取得各层 Top-8 expert IDs，再筛选未解析位置，累计每层 256 维 assignment 直方图。LM head 对生成 block 计算候选 token，按置信度和最低接收量更新 mask，直至全部解析。

初始 Q=B×32，每层统计 assignment 数为 8Q。Q 下降代表统计对象减少；实际模型主体仍计算完整 prompt、padding 和整个生成 block。这些直方图并非 Vanilla 全部 MoE 工作量。双卡层切分也不是 Expert Parallel。

完整性检查通过：group 步号连续、相邻步剩余 Q 一致、末步 Q=0、每步 19 层、每层直方图之和为 Q×8。

| 配置 | 每组实际 forward 次数 |
|---|---|
| GSM8K batch 8 | 23、23、22、22 |
| GSM8K batch 16 | 25、22 |
| HumanEval batch 8 | 21、21、27、30 |
| HumanEval batch 16 | 24、30 |

## 四组路由对照

Active、effective、Top-10 share 为 step 0–12 均值；相邻 Jaccard 为目标 step 1–12 均值。区间内全部 groups 均存活，各 group、各层等权。该窗口仅用于描述性比较，相同步号不代表相同 Q 或生成进度。

Effective=(Σc)²/Σc²，表示集中程度，不能解释为可保留的实际专家个数。Top-10 share 是 assignment 占比，不是耗时或 router 权重占比。

| 数据集 | Batch | Active | Effective | Top-10 share | 相邻 Jaccard |
|---|---:|---:|---:|---:|---:|
| GSM8K | 8 | 98.64 | 33.16 | 47.18% | 79.14% |
| GSM8K | 16 | 118.88 | 35.21 | 45.78% | 83.48% |
| HumanEval | 8 | 94.86 | 37.07 | 44.89% | 74.62% |
| HumanEval | 16 | 115.18 | 39.31 | 43.71% | 78.29% |

GSM8K 的 active 增加 20.25、effective 增加 2.05；HumanEval 分别增加 20.31、2.24。支持有效工作集相对物理激活集合增长缓慢，而非固定的“30–40 个专家”定律：HumanEval batch 16 的 step 14 effective 约为 48.9。

HumanEval 从全 mask 初始化到第一次更新后的 Jaccard 仅约 53%–54%，随后较高；初始化转换值得单独分析，不宜把第零步专家表作为全程固定集合。

## 完整排序跨步预测

对当前步负载，分别按当前负载降序（Oracle）、先前步负载降序和随机排序期望计算累计覆盖曲线面积 AUC。收益恢复率=(AUC_previous−AUC_random)/(AUC_oracle−AUC_random)。并列负载按专家 ID 排序。

| 数据集 | Batch | lag 1 收益恢复率 | lag 8 收益恢复率 | lag 1 Oracle AUC | lag 1 Previous AUC |
|---|---:|---:|---:|---:|---:|
| GSM8K | 8 | 97.36% | 88.84% | 0.92565 | 0.91448 |
| GSM8K | 16 | 98.42% | 91.48% | 0.91841 | 0.91181 |
| HumanEval | 8 | 96.60% | 87.66% | 0.92317 | 0.90889 |
| HumanEval | 16 | 97.70% | 90.28% | 0.91638 | 0.90686 |

目标 step≤12；lag 8 只覆盖 step 8–12，不同 lag 并非完全相同目标样本上的比较。AUC 是负载覆盖指标，不是逐专家排名准确率或加速比。Oracle 对覆盖曲线最优，不保证对真实 kernel latency 最优。

结果支持复用上一步逐层排序作为执行准备的候选信号。仍需与固定逐层先验比较，区分长期专家偏好和额外时间连续性。

## Query-matched null

每个 group、每层固定 step-0 直方图，无放回抽取与目标步相同数量的 assignment，重复 256 次，seed=0。下表为 step 1–12 均值，区间与路由表不同。

| 数据集 | Batch | Active 真实 / Null | Effective 真实 / Null | Top-10 share 真实 / Null |
|---|---:|---:|---:|---:|
| GSM8K | 8 | 99.94 / 75.17 | 33.67 / 26.69 | 46.61% / 54.36% |
| GSM8K | 16 | 120.81 / 88.15 | 35.82 / 27.71 | 45.16% / 53.36% |
| HumanEval | 8 | 96.17 / 69.71 | 37.92 / 26.42 | 44.01% / 55.75% |
| HumanEval | 16 | 117.26 / 81.51 | 40.31 / 27.11 | 42.78% / 55.11% |

四组真实后续路由均比此基线更分散，支持“不能仅以第零步分布下采样解释后续变化”。这是 assignment-level 近似，未保持同一 token 的 Top-8 联合结构，不能声称排除了所有有限样本影响。其 95% 区间是条件抽样区间，不是跨数据集置信区间。

## 动机与证据边界

可支持的动机：两个任务、batch 8/16、单 block 内，未解析 Query 的专家负载非均匀且跨步可预测，有效工作集增长相对缓慢。候选方法是复用上一 step 的逐层负载顺序，准备专家调度或数据移动，保持当前 router 的全部 assignment。

尚未证明严格饱和、固定专家数、无损裁剪长尾、真实长文本任务质量、跨 block 稳定性、与 FOCUS 的互补加速或实际 latency 收益。Batch 16 仅两个 groups，层和相邻 step 并非独立样本。Trace 未记录 prompt ID/hash，不能仅凭直方图验证各配置请求完全相同或重建 token 路由。

尾部 group 完成后，仅对存活 group 汇总，平均 Q 可能回升，例如 HumanEval batch 16 step 23→24。这不表示同一 group 重新增加 mask。

## 下一步：单层专家 replay

四组轨迹和离线复核已完成；后续顺序为 replay → 多 block → Vanilla/FOCUS → 跨模型 → Expert Parallel。

先实现按真实直方图负载形状构造的合成单层 replay，明确使用的 MoE kernel 和模型专家矩阵尺寸。比较现有实现顺序、上一 step 顺序、当前负载降序；同一观测保持相同输入、权重、assignment 数和计算量。测量 kernel 时间、含重排/调度的总时间，加入 warm-up、重复测量和输出数值误差校验。

合成 replay 仅验证执行形状与顺序。真实 replay 需采集该层输入激活、Top-8 ID 和 router 权重。若评测 Vanilla 全部 MoE 工作量，还需包含 prompt 和已解析 token；mask-only trace 不足以重建它。

若当前负载排序都无收益，需定位 kernel/访存瓶颈后调整方向；若有收益，再检验前一步排序能否保留收益并覆盖维护成本。不能仅在串行逐专家循环中制造调度差异，就声称生产 grouped GEMM 获得加速。

已实现合成单层 replay：入口为 benchmark/replay_moe_layer.py，运行命令与测量口径见 [Replay 实验说明](../../../benchmark/MOE_REPLAY.md)。本地 CPU 验证通过，CUDA 编译、数值与性能尚待服务器验证。以下是本次离线复核的复现命令，已执行完成，无需重跑：

```bash
cd /Users/keduck/Documents/nju/Code/FOCUS
for dataset in gsm8k humaneval; do
  python benchmark/analyze_moe_offline_hypotheses.py \
    "artifacts/moe_denoising/llada2-mini/${dataset}/bs16/routes_bs16.jsonl" \
    --output-dir "artifacts/moe_denoising/llada2-mini/${dataset}/bs16/offline_hypotheses" \
    --main-max-step 12 --max-lag 8 --null-repeats 256 --seed 0
done
```

Python 需有 NumPy；本次已用 Mac 上带 NumPy 的 runtime 完成。

## 结果索引

各目录包含原始 JSONL、逐层/逐步 CSV、轨迹 SVG；offline_hypotheses 子目录包含排序和 null 的明细、汇总与 SVG。

- [GSM8K batch 8](gsm8k/bs8/)
- [GSM8K batch 16](gsm8k/bs16/)
- [HumanEval batch 8](humaneval/bs8/)
- [HumanEval batch 16](humaneval/bs16/)
- [早期 batch-8 动机报告](gsm8k/bs8/MOTIVATION_EXPERIMENT.md)
