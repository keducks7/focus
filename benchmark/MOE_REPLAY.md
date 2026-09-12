# 单层 MoE 专家顺序 replay

这是基于实测直方图的合成 replay。只读取本地模型 config.json 来确定专家矩阵尺寸；使用随机 BF16 权重和激活，不加载完整模型。只需一张 A800，GPU 1 可空闲。

## 验证的问题

上一 denoising step 的负载排序，能否在实际 Triton fused MoE kernel 中降低运行时间，并覆盖维护该排序的成本？

复用 lmdeploy/pytorch/kernels/cuda/fused_moe.py，增加可选 expert_order，将网格第二维映射到真实专家 ID。权重不移动，所有真实 assignment 都执行。默认 None 保持原来映射。GPU 不保证按网格 ID 串行启动，因此这里检验网格排列的性能敏感性，不保证实现了“重专家严格先执行”。

四个对照：

- native：原有无映射路径。
- identity：恒等专家映射，识别指针映射及其编译路径开销。
- previous：上一 step 负载排序。
- oracle：当前 step 负载降序，仅为负载排序对照，不是硬件最优上界。

排序不是新方法中的多参数函数。layer/steps/repeats 是实验采样和测量设置。

## 输入与数值验证

Trace 只有各专家计数，无法恢复原始 token-to-expert 联合路由。脚本用二部图度数构造生成合成 Top-8 路由，严格保持每个专家 assignment 数及每个 token 的 8 个不同专家。不同顺序共用同一组输入、权重、合成路由、均匀路由权重。

每个观测先检查所有顺序的 MoE 输出及 prepared 路径与 baseline 数值一致；另对两个 token 用 PyTorch linear/SwiGLU 做独立参考。失败即停止，不输出成功结论。默认选 layer 10、steps 1/4/8/12，覆盖每个 group，使用对应上一 step 的完整直方图。

不包含 Attention、router 网络、共享专家、跨卡通信或原始语义。当前 trace 是 mask-only，因此不能将结果直接当成 Vanilla 全部 MoE 或端到端加速。

## 测量口径

- prepared_experts：两个 grouped GEMM 与 SwiGLU，预先准备索引和缓冲区，不含 dispatch、reduce 和排序。
- moe_total：现有 fused_moe 完整调用，含 dispatch、缓冲分配与归并；previous 额外计入当前计数/排序维护以供下一步使用，oracle 计入当前计数/排序后用于本步。
- CUDA Event 与同步 wall time 同时记录；每个路径先编译/自动调优/warm-up，再随机交错四种顺序和两种口径，重复多个 rounds。
- 权重保持驻留，同一观测重复执行，未强制清空缓存。结果反映重复热工作负载，不代表跨层、多 block 或冷缓存访问。
- 查看 identity 对 native 的差异以及每个 group/step 的测量范围，不能仅从某一个快样本宣称加速。

## 服务器命令

代码同步后，在服务器已有 focus 环境中确认依赖：

```bash
cd /root/lkd/FOCUS
conda activate focus
python -c "import torch, triton, numpy; print(torch.__version__, triton.__version__, torch.cuda.is_available())"
```

先冒烟，包含数值验证和两个 group，首次 Triton 编译/调优可能需要时间：

```bash
CUDA_VISIBLE_DEVICES=0 python benchmark/replay_moe_layer.py \
  artifacts/moe_denoising/llada2-mini/gsm8k/bs16/routes_bs16.jsonl \
  /root/lkd/Models/LLaDA2.0-mini \
  --output-dir results/moe_replay/gsm8k_bs16_smoke \
  --layer 10 --steps 4 --warmup 2 --iterations 3 --rounds 2
```

通过后执行两任务正式实验：

```bash
for dataset in gsm8k humaneval; do
  CUDA_VISIBLE_DEVICES=0 python benchmark/replay_moe_layer.py \
    "artifacts/moe_denoising/llada2-mini/${dataset}/bs16/routes_bs16.jsonl" \
    /root/lkd/Models/LLaDA2.0-mini \
    --output-dir "results/moe_replay/${dataset}_bs16" \
    --layer 10 --steps 1 4 8 12 --warmup 5 --iterations 10 --rounds 5 || break
done
```

输出 metadata.json、timings.csv、summary.csv。已有 timings.csv 时拒绝覆盖，请为重复实验指定新目录。

## 如何判断下一步

先检查数值验证、identity 成本和重复测量波动，再看 previous 是否在 moe_total 稳定优于 native。只在 prepared 路径变快而总耗时无改善，说明排序维护/dispatch 等成本抵消收益。若 oracle 也不改善，当前实现缺少“只改专家网格顺序”的加速证据；这不能否定其他权重预取或 EP 优化。

若有稳定收益，应扩展到其他层及真实激活 replay，然后验证多 block 和真实评测长度；不可由单层合成结果直接推算端到端收益。

本地验证：路由重建单元测试、四组真实 trace 选定观测的精确计数验证、Python 语法检查。Mac 无 CUDA，Triton 编译、GPU 数值和性能尚待服务器执行。

