# 同一专家内部的跨请求冗余：纯观察实验

## 研究问题

此前 B1–B32 轨迹说明多个请求共同访问一些专家，但访问共享不等于计算冗余；之前的旁路观察也没有支持直接把不同专家的输出互相替代。

本实验转而问：**固定同一个专家、同一个 layer/step/state，它处理不同请求的 token 时，输出是否存在可观测的低维或近邻结构？这种结构是输入本来就有的，还是专家变换之后更加明显？**

这不是方法评测，也不是干预实验。只读取保存的原生层输入和真实 Top-k，旁路计算原本被选中的专家分支。不会重新生成、不替换输出、不改路由、不缓存到解码过程、不运行未被选中的替代专家。

## 输入与资源

- 入口：`benchmark/analyze_intra_expert_batch_redundancy.py`。
- 输入目录为 `observe_expert_lifecycle.py` 生成的观测目录，必须保留 `metadata.json`、`generation_complete.json`、`group*.json`、`inputs_group*_step*.pt`。仅有分析 CSV 或少量 `functional_outputs` 样本不够。
- 默认使用已有 `results/expert_lifecycle_observation/b32_l10_full_v1` 的四个 B32 group，L10、step 0/4/7、32 个生成位置。step 从 0 开始，0 是第一次 forward。
- 需要采集时的同一份未量化 LLaDA2.0-mini checkpoint，且保留全部 safetensors 分片和索引。
- 模型权重读取与 SwiGLU 计算复用已有 `analyze_expert_output_redundancy.py`。环境需要 PyTorch、Safetensors；不需要 LMDeploy、OpenCompass、Transformers 或重新加载全模型。
- **单卡足够进行这种旁路重放**：每次只在 GPU 放一个专家的 gate/up/down 权重，以及小块 token 输入。不会把 16B 模型全部放到一张 A800 40G 上。GPU 1 不参与本实验。
- `--inventory-only` 仅需 CPU PyTorch，不读取专家权重，不生成输出指标。

## 采集与比较流程

1. 固定真实 B32 的一个 group、layer、step；读取目标 MoE MLP 的实际输入，不使用末层 hidden state。
2. 用该次 forward **输入时**的 token ID 区分 MASK 与 decoded。接受当步仍以 MASK 输入的位置属于 MASK；首次以实际 token 输入的下一步才属于 decoded。只分析生成位置，不含 prompt/padding。
3. 从真实 Top-k 分配建立“专家 → 请求 → token 位置”索引。对于同一专家，同一 token 只出现一次。分析未乘 router weight 的单专家输出，不含 shared expert，也不是完整 MoE 加权输出。
4. 按真实 checkpoint 的 `down_proj(silu(gate_proj(h)) * up_proj(h))` 旁路重放。采用 BF16，输出转 FP32 在 CPU 统计；chunk/GEMM 实现可能与原生融合算子有数值差异，不宣称 bitwise 一致。
5. 独立进行下面两类比较，并记录各自覆盖情况。

### A. 固定 token 总数，改变请求组成

始终固定 **N=16 个不同 token**，从 K=1/2/4/8 个不同请求取样，每条请求严格贡献 N/K 个 token：16/8/4/2。不能把不同 K 的 token 总数也一起增加。

同一个 expert/group/step/state 必须同时满足四种 K 的取样条件，才能进入这条曲线，避免每个 K 换一批专家。K 增加时请求集合嵌套；原有请求的 token 顺序固定，只截取更短前缀。token 不放回抽样，默认三个种子重复。

记录四种有效秩：输入、输出、去均值输入、去均值输出，以及输入→输出和 K→K1 的配对变化。

有效秩采用奇异值平方能量的熵指数：`exp(-sum(p_i log p_i))`，`p_i = sigma_i² / sum(sigma_j²)`。它是连续的能量集中指标，不是整数矩阵秩，也不是可删除 token 的数量。去均值是在每个 16-token 样本内部进行；可区分公共均值方向与剩余差异。

### B. 请求内与跨请求的近邻比较

对实际访问该专家的 token，选取同时具有两类候选的 anchor：

- 请求内：同请求的其他 token，排除 anchor 自身。
- 跨请求：其他请求的 token。

每侧随机取 **4 个不同候选**，每个 expert/state 最多取 64 个 anchor，三次重复。输入与输出四个空间使用完全相同的 anchor/candidate 样本。不能直接比较“所有同请求候选”和“所有跨请求候选”的最小值，因为候选池大小不同会机械降低最小距离。

统计最小 cosine distance 与最小 relative L2 distance。relative L2 的分母是 anchor 范数；两个指标的最优候选可能不同。去均值近邻使用该 expert/state 的全部实际访问 token 的均值，因此其中心化口径与 A 的固定 16-token 秩不同。

四个空间任意相关向量为零范数时，联合排除该 anchor，并报告数量；不能把无定义的 cosine 当作相似。跨请求候选按 token 均匀采样，**不是按请求均匀**；高访问量请求会贡献更多候选。

## 运行命令

在服务器 `/root/lkd/FOCUS`，确认新增脚本及其依赖已同步后：

```bash
cd /root/lkd/FOCUS
conda activate focus-moe

# 可选：先检查样本覆盖，不需要 GPU 或模型权重。
python benchmark/analyze_intra_expert_batch_redundancy.py \
  results/expert_lifecycle_observation/b32_l10_full_v1 \
  --output-dir results/intra_expert_batch_redundancy/coverage_v1 \
  --inventory-only --steps 0 4 7
```

冒烟运行：只用 group 0、三个 step、一次抽样重复；保持 N=16 和四种请求组成不变。

```bash
CUDA_VISIBLE_DEVICES=0 python benchmark/analyze_intra_expert_batch_redundancy.py \
  results/expert_lifecycle_observation/b32_l10_full_v1 \
  /root/lkd/Models/LLaDA2.0-mini \
  --output-dir results/intra_expert_batch_redundancy/b32_l10_smoke_v1 \
  --layer 10 --groups 0 --steps 0 4 7 \
  --states mask decoded --tokens 16 --request-sizes 1 2 4 8 \
  --repeats 1 --device cuda:0
```

正式运行：四个 group，共 128 条请求，三次抽样重复。

```bash
CUDA_VISIBLE_DEVICES=0 python benchmark/analyze_intra_expert_batch_redundancy.py \
  results/expert_lifecycle_observation/b32_l10_full_v1 \
  /root/lkd/Models/LLaDA2.0-mini \
  --output-dir results/intra_expert_batch_redundancy/b32_l10_full_v1 \
  --layer 10 --steps 0 4 7 \
  --states mask decoded --tokens 16 --request-sizes 1 2 4 8 \
  --repeats 3 --device cuda:0
```

如果实际观测目录不同，替换第一个路径即可。每次使用新的输出目录，程序拒绝覆盖非空目录。步骤必须已经保存，并且其所有请求仍活跃；缺文件、缺原生 observer 对照 audit、请求已结束都会明确报错。不能随意加入晚期 step 12/16：现有四个 group 中部分请求届时已结束。

## 输出文件与上传

| 文件 | 内容 |
|---|---|
| `metadata.json` | 参数、原始采集配置、输入文件哈希、checkpoint config 哈希及数值口径 |
| `coverage.csv` | 每专家和每状态的真实访问量、各 K 的可采样请求数、共同合格专家数、这些专家覆盖的分配比例 |
| `ranks.csv` | 配对有效秩原始记录，包含请求 ID、token 位置和相对 K1 的变化 |
| `rank_group_summary.csv` / `rank_summary.csv` | 先重复平均、再专家等权平均，最后 group 等权平均 |
| `neighbors.csv` | 同/跨请求、输入/输出、raw/centered 的距离及有效 anchor 数 |
| `neighbor_group_summary.csv` / `neighbor_summary.csv` | 排除没有有效 anchor 的记录后，按同样层级汇总 |
| `complete.json` | 完成标记、有效记录数和解释边界 |

用于后续本地绘图，上传整个输出目录即可，不需要再上传权重或原始大体积 PT。仅有 `complete.json` 的 `complete=true` 才表示指定范围全部跑完；中途退出仍会保留此前完成的 group/step CSV，但没有断点续跑功能。

## 已核查的样本覆盖与解释边界

本地对现有四个 B32 group 的 L10、step 0/4/7 做了仅覆盖检查：MASK 的共同合格专家每个 group/step 为 7–18 个；decoded 在 step 0 不存在、step 4 为 0–2 个、step 7 每个 group 为 2 个。**decoded 固定 N/K 曲线只代表极小的高负载子集，不能概括全部 decoded 专家。**近邻实验有独立资格条件，覆盖范围较广；两类结果不能假定来自同一专家总体。

共同秩资格偏向热专家，必须同时查看 `common_rank_assignment_share`，不能只呈现好看的低秩曲线。step 0 没有 decoded 会输出覆盖为零，而不是“decoded 有效秩为零”。

K 表示**从同一次真实 B32 forward 的保存数据中构造的离线请求组成**，不是重新运行了真实 B1/B2/B4/B8。不同 K 的 token 集合并不相同，不能把差异解释成 batch 改变了专家函数。抽样重复、同 group 内专家和跨 step 也不是独立实验样本，当前脚本不输出显著性检验。

可能值得继续的现象：跨请求距离在控制候选数后较小、中心化后的输出仍低秩、输入→输出出现一致结构变化，且覆盖率足够。但这些都**不能单独证明安全合并、直接复制输出、缓存正确性或真实加速**。若仅 raw 输出低秩而中心化后不低秩，更可能是公共方向占主导。

## 本地代码测试

```bash
python -B tests/test_lmdeploy/test_intra_expert_batch_redundancy.py
```

12 项测试覆盖固定 N/K、嵌套请求、无放回、共同专家资格、候选池配平、零范数处理、聚合权重、坏输入拒绝、覆盖检查及模拟权重的 CPU 端到端重放。没有 PyTorch 时数值测试会跳过。CPU 测试不代替真实 checkpoint/CUDA 验证。
