# 原生路由竞争、生命周期持续性、跨请求协调空间

本研究包实现三个问题，但严格区分两类证据：

1. **原生候选竞争**：私有专家的排名、Top-8/9 边界、合法共享候选覆盖。
2. **原生生命周期**：专家进入/退出、原生排名持续性、候选下一步自然进入实际 Top-8。
3. **离线规划潜力**：相同逐 token 分支预算下，独立 token、请求内联合、整个 batch 联合的专家并集。

前两项是纯观察；第三项是保存路由上的统计规划，不是新的模型运行结果。**没有删除/替换分支的模型干预，没有重新生成，没有改变任何接受决策，也没有专家输出相似性或输出缓存。**

## A. 候选竞争与生命周期（同一个入口，CPU）

入口：`benchmark/observe_route_competition.py`。

### 输入

复用既有 `results/expert_lifecycle_observation/b32_l10_full_v1`，需要完整保留：

- `metadata.json`、`generation_complete.json`、`group*.json`。
- `inputs_group*_step*.pt`：原生 router logits、Top-8、权重、输入 token ID、live-request 标记。
- 同一 checkpoint 的 `config.json` 和 `model.layers.10.mlp.gate.expert_bias`。只读取这一条偏置向量；**不读取 router 权重，不加载专家，不加载整个 16B 模型**。

环境仅需 CPU PyTorch、Safetensors。服务器已有 `focus-moe` 环境即可，CUDA_VISIBLE_DEVICES 不需要设置。没有 LMDeploy、OpenCompass、Transformer 模型加载或 NCCL 初始化。

### 官方规则与审计

公式对应仓库所带 HF 源码 `opencompass-0.5.1.post1/opencompass/models/LLaDA2.0-mini/modeling_llada2_moe.py` 中的 `LLaDA2MoeGate`，不是 LMDeploy 重实现：

1. 读取保存的 FP32 logits，计算 sigmoid。
2. 排序分数加 `expert_bias`；混合权重不加偏置。
3. 每个 group 的两个最高分相加，选择原配置的 `topk_group` 个 group。
4. **仅在这批合法 group 内**得到 Top-8 和 Top-16；扩大候选数不改变 group 选择数量。
5. 每一个活跃 token 都核对记录的原生 Top-8 成员和混合权重。权重按官方公式归一化并保留 scaling factor。

真正的精确分数并列允许不同 topk 调用选出不同成员，但保留原始 Top-8、标记并列，不把并列排名解释为严格偏好。其他成员/分组/权重不一致会明确报错；不能假设偏置为零、改用未受分组限制的候选，或静默略过错配。

审计容差仅用于核对浮点权重，不是方法阈值。`native_gate_audited=true` 只在整个指定范围成功后写入。

### 观察口径

- 每个 layer 分别处理；不同层的同编号专家不是同一参数，不跨层合并 ID。
- 当前“私有”严格定义为该 step 只被 **一条活跃请求**访问。另存所有请求访问次数和同 token 状态的访问次数，不设“≤4 就是私有”的阈值。
- 排名是带偏置的原生选择分数排名，权重是实际混合权重，两者不能混淆。
- 本步候选是额外的 rank 9–16，不包括该 token 的实际 Top-8；分别观察它们是否被本请求其他位置、其他请求真实 Top-8 使用。
- 热门对照从**其他 group**的同一步请求访问频率建立，先限定该 token 的合法专家，再匹配其他请求实际专家池的合法集合大小。没有其他可用 group/request 时对照为缺失，而不是零。
- 对照只控制池大小和 held-out 热门排序，不保持二部图的所有边际，不能替代此前的 degree-preserving 重连。
- 接受时间来自原始结果，仅作事后生命周期标签。接受当步的输入仍是 MASK，下一次实际 token 输入才是 decoded。

### 跨步与删失

只配对相同请求、相同位置、同一层的**相邻原生 step**。记录：

- Top-8、Top-16 Jaccard。
- 新进入专家有多少来自上一步 rank 9–16。
- 上一步的共享候选有多少在下一步自然被选中。
- 私有专家的下一步持续率，按上一排名分别汇总。

同一输入身份下，记录每个实际专家的连续选择 episode。真实退出 Top-8、输入身份替换、请求完成、观测间隔、轨迹结束使用不同结束原因。请求/窗口结束为右删失，不把“没有后续观测”当作永远稳定。若从中途开始观察，已有选择可能左删失。

默认读取每组保存的完整自然轨迹。完成请求会从支持集和观测位置中排除，后期 live request 数下降；比较“大 batch”时应限制共同完整活跃窗口（此数据为 step 0–7），而完整生命周期结果另按实际活跃请求数解释。

### 冒烟运行

代码同步到服务器后：

```bash
cd /root/lkd/FOCUS
conda activate focus-moe

python benchmark/observe_route_competition.py \
  results/expert_lifecycle_observation/b32_l10_full_v1 \
  /root/lkd/Models/LLaDA2.0-mini \
  --output-dir results/route_competition/l10_smoke_v1 \
  --layer 10 --groups 0 --steps 0 1 2 3 4 5 6 7 \
  --candidates 16
```

使用完整目录，即使只处理 group 0，也会读取其他 group 的同一步记录来建立 held-out 热门对照。

### 正式运行：四组、完整自然轨迹

```bash
python benchmark/observe_route_competition.py \
  results/expert_lifecycle_observation/b32_l10_full_v1 \
  /root/lkd/Models/LLaDA2.0-mini \
  --output-dir results/route_competition/l10_full_v1 \
  --layer 10 --candidates 16
```

不写 `--steps`，程序自动读取每组所有保存步骤，不额外追加 forward。若只想看完整 B32 窗口，显式加 `--steps 0 1 2 3 4 5 6 7` 并用不同输出目录。

现有输入只保存 L10 的完整 logits。因此该入口现在先做 L10；仅改成 L2/L18 不会凭空生成缺少的层输入。B 部分的原始 Top-8 规划可以直接分析 L2/L10/L18。

首次运行导出 `router.json`。之后可在本地或服务器用这个快照，不需要 checkpoint/Safetensors：

```bash
python benchmark/observe_route_competition.py \
  results/expert_lifecycle_observation/b32_l10_full_v1 \
  --router-snapshot results/route_competition/l10_full_v1/router.json \
  --output-dir results/route_competition/l10_reanalysis_v1
```

### 输出

| 文件 | 含义 |
|---|---|
| `tokens.jsonl` | 每活跃位置的实际 Top-8、合法 Top-16、两种分数、实际权重、身份、排名边界和候选共享统计 |
| `branches.csv` | 实际选中专家的排名、请求访问次数、私有标记、与最佳合法共享额外候选的分数差 |
| `token_group_summary.csv` / `token_summary.csv` | 候选竞争与匹配热门对照，先 group 内位置均值，再 group 等权 |
| `rank_group_summary.csv` / `rank_summary.csv` | 按原生排名分析私有比例、权重和分数差；条件指标保留有效计数 |
| `transitions.csv` / `branch_transitions.csv` | 真实相邻 step 的自然更替及按原排名的持续性 |
| `transition_group_summary.csv` / `transition_summary.csv` | 分 MASK→MASK、MASK→decoded、decoded→decoded 的汇总 |
| `rank_survival_group_summary.csv` | 上一步私有/非私有 × 排名的自然持续率 |
| `episodes.jsonl` / `episode_duration_histogram.csv` | 同输入身份专家选择的连续观测长度及左右删失 |
| `router.json` / `metadata.json` / `complete.json` | 路由偏置快照、输入哈希、协议和完成标记 |

所有 CSV 分母缺失时保留空值；每个汇总指标同时保存有效数/有效 group 数。没有显著性检验，不把 token、专家、step 当作独立实验重复。

## B. 合格选择空间与请求内/跨请求潜力（CPU、无需模型）

入口：`benchmark/analyze_route_choice_space.py`，仅需 Python 标准库。

输入是既有实际 B1/B2/B4/B8/B16/B32 的完整 Top-8 轨迹，不是从 B32 抽样冒充真实不同 B。该入口与 A 使用不同的既有原生采集目录，各自遵循自己的 token/接受历史，不假设两次解码的每一位置完全相同。

固定一个路由权重容忍量 `epsilon=0.1`，不做多参数搜索：

1. 将原 Top-8 的记录权重归一化，得到每个 token 覆盖 90% 原路由质量所需的**最小分支数**。
2. 精确枚举该分支数下所有合格原 Top-8 子集。这个“质量”仅指 probability mass，不是生成质量。
3. 用完全相同的逐 token 分支数、质量约束和原候选集合，做 token 独立、请求内联合、batch 联合三种**离线**规划。
4. 各方案每个 token 的预算与可行性均核对，不能用“平均分配数相同”替代逐 token 相同。

规划使用确定性的专家消除启发式，多种可行起点，并非全局最优。batch 方案把独立及请求内方案作为可行起点，保证不劣于它们；这种不劣性质是构造，不是新现象。重要的是收益大小。

请求内优化可能破坏跨请求重叠，因此**请求内方案的全 batch 并集可能比 token 独立更大**。不强制将该负面结果改为单调曲线。三种差值均报告：

- `within_request_gain = token_union - request_union`，允许负值。
- `cross_request_gain = request_union - batch_union`。
- `total_joint_gain = token_union - batch_union`，防止把恢复请求内规划破坏的共享当成新增压缩。

默认对 MASK 和 decoded 均分析选择空间。若复现旧的仅压缩 MASK 基线，加 `--planning-states mask`；decoded 会保留全部原生分支，各范围使用同一个全局固定支持集。必须对比相同协议，不能将这两种设置混用。

### 冒烟运行

```bash
python benchmark/analyze_route_choice_space.py \
  results/expert_trajectory/sharing_b1_to_b32_v1 \
  --output-dir results/route_competition/planning_smoke_v1 \
  --batches 32 --layers 10 --steps 4 --groups 0 --epsilon 0.1
```

### 正式运行

```bash
python benchmark/analyze_route_choice_space.py \
  results/expert_trajectory/sharing_b1_to_b32_v1 \
  --output-dir results/route_competition/planning_b1_to_b32_v1 \
  --batches 1 2 4 8 16 32 --layers 2 10 18 --steps 0 4 7 \
  --epsilon 0.1
```

已在本地完整跑过这个协议：221,184 个 token-layer-step 记录、2,268 个规划单元。结果位于本地 `artifacts/route_competition/planning_b1_to_b32_v2/`；可直接分析，不必在服务器重复跑 B。

主要输出：

- `choices.csv`、`choice_group_summary.csv`、`choice_summary.csv`：最少分支数、合格组合数、有选择空间的位置比例、合格组合全部使用其他请求已激活专家的比例。
- `planning_groups.csv`、`planning_summary.csv`：三个范围的专家并集、相同分配数、三个差值及逐 token 最低保留 mass。
- `--save-plans` 可额外输出各 token 的离线专家 ID；这些 ID 不会被执行。
- 原轨迹权重可能有序列化舍入。“精确相同约束”是相对于记录值，不是所有后端的 bitwise 计算保证。

## 上传和结论边界

两项服务器入口完成后上传各自完整输出目录，后续本地绘图不需要模型权重。原始 PT/JSONL 留在原目录，不改动、不重复上传。仅 `complete.json` 的 `complete=true` 表示全部指定范围完成；中途中止无完成标记，没有断点续跑功能，每次用新目录。

本实验用于选择研究路线：是明确个体偏好，还是跨请求共同候选与暂时性竞争；联合优势来自单请求内部还是跨请求。不能据分数相近宣布专家功能等价，也不能将规划并集降幅称为加速或无损生成。不会安排单分支删替干预。

## 测试

```bash
python -B tests/test_lmdeploy/test_routing_competition.py
```

22 项测试涵盖精确子集枚举、逐 token 预算、真实跨请求 toy、请求内优化破坏全局共享的负例、固定支持、B1 对照、held-out 池大小、时间间隔和删失，以及直接执行仓库官方 HF Gate 类的 CPU 数值对照（包含 256 experts / Top-8 / Top-16）。CPU observer 使用合成原生 gate 输出完成端到端测试；本机没有真实 checkpoint 偏置，真实候选竞争审计仍需服务器运行。无 PyTorch 时数值测试跳过。
