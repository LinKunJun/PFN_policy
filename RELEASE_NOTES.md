# PFN Sequential / Static optimized release

本包以 2026-10-13 的八卡 MTT S5000 优化项目源码为基准，固化经过 MUSA 数值检查及
Sequential 单 Update 实验的优化路径。解压后无需任何 PFN_FAST_* 或其他优化环境变量。

## 启动与控制（保留原 Bash 入口业务语义）

项目目录：`/home/mccxadmin/PFN_network_interference_musa_optimized/`。

```bash
cd /home/mccxadmin/PFN_network_interference_musa_optimized
bash PFN_network_interference_musa_start.sh             # 正常全量训练：static + sequential
bash PFN_network_interference_musa_start.sh status      # 最新任务状态
bash PFN_network_interference_musa_start.sh logs        # 追踪过滤后的日志
bash PFN_network_interference_musa_start.sh raw         # 原始日志
bash PFN_network_interference_musa_start.sh check       # 8 卡通信/环境检查
bash PFN_network_interference_musa_start.sh dry-run     # 仅输出将启动的命令
```

正常 `start` 在后台启动，仍使用 `logs/pretrain_latest.txt` 跟踪 PID、输出和日志，
并保留原有锁、`RESUME`、`CONFIG`、`OBJECTIVE`、`PYTHON`、`NPROC_PER_NODE` 等接口。
`OBJECTIVE` 默认 `both`；单独 Sequential 训练用 `OBJECTIVE=sequential`，
不是为了触发性能优化，所有目标均使用同一套优化实现。

### CPU / MUSA 并行

默认 8 个 DDP rank、每 rank 16 个持久化环境子进程（`ENV_WORKERS=16`）、
每 rank 4 个 CPU 计算线程（`CPU_THREADS=4`），环境 worker 内 BLAS 固定为 1 线程。
约 160 个并发执行槽位，合理使用 128 物理核与 256 个 SMT 逻辑线程，同时保留
MUSA 驱动、通信与 Python 主线程调度余量。**这是硬件拓扑驱动的保守默认值，
不是经八卡实测证明最快的配置。** 完整 Update 的 77.89 秒基准是在
`OMP/MKL/OPENBLAS_NUM_THREADS=2`、`--env-workers=2` 的专用测量入口下得到。
并行度可以通过 `CPU_THREADS` 和 `ENV_WORKERS` 覆盖，算法优化本身不可关闭。

## 固化的性能改动

- GNN inference 使用一个连续的 GPU 状态 Bank；只在冻结权重范围内复用编码。
- 不变观测内容键复用、同大小图节点聚合、增量历史索引与反馈打包成为默认路径。
- PPO 无碰撞的 1D `(record_id, episode)` 去重和有限状态 ID 标记映射成为默认路径。
- PPO 使用 `Adam(foreach=True, fused=False)`、`clip_grad_norm_(foreach=True)`；
  PPO `Categorical` 禁用防御性参数验证，前向 logits 和优化目标不变。
- Catalog 和 Dense 图使用**按实际张量尺寸判断的动态零 Padding**：
  已经满足尺寸时不调用 `pad`，需要补齐时始终执行原样的 `pad`。
- 对异尺寸图、超出容量的 GPU Bank、超出 int64 安全范围的键保留必要的
  **数据有效性保护**；这些是正确性边界，并非旧版可切换实现。
- 不使用 Minibatch Plan 预取，不做 Transformer 历史长度分组。

## 性能、正确性、边界

最后一次目标机单 Update 实测：**77.89 秒（8 卡 Sequential）**，当时使用
Unique 两处优化和 `PFN_SKIP_ZERO_PAD=1`、已验证的其他加速开关。
这不是 `OBJECTIVE=both` 全量训练的总耗时，也不能保证不同 worker/thread
数量下的每次 Update 时间逐秒一致。新包清除了实验开关并固定同一计算路径，
需要在目标服务器上再次跑完整单 Update 后才能称为发布版验收。

之前各轮 MUSA 实验已分别验证数值正确性，动态零 Padding 的 Edge PPO 梯度出现过
最大绝对偏差约 1.49e-8 的 FP32 舍入差异；此处不声称按 bit 精确一致。
保留 GCN/Transformer/PPO 结构、随机种子生成、MCCL/DDP 全同步、梯度累积
和完整训练超参数。没有更换框架或要求新的加速依赖。

## 发布内容

包内保留 `configs/`、原有必要的 `scripts/`、`tests/`、`pfn_network/`、
依赖文件、README 和论文说明；不包含训练 `runs/`、`profiles/`、`logs/`、
旧源码备份、pycache、一次性 A/B 安装器或大型 Trace 文件。
解压到新的独立目录，避免覆盖正在运行的旧工程。


## GNN inference content IDs — 2026-10-14

Frozen inference now interns exact graph/state/global content keys to integer IDs. Content equality, stable deduplication and LRU order are preserved; writable/non-owning inputs still re-read bytes. IDs remain scope-local and monotonic, metadata is bounded, and model/PPO/checkpoint mathematics are unchanged.

Validation: original pytest and exact CPU fingerprint; MUSA actions, rollout outputs, PPO metrics, RNG and boundary cases exact. Only empirically noisy floating fields use min(4 × five-original-run noise, 2e-5); stable fields remain exact. Candidate pytest and single-GPU/one-CPU-thread update passed.

Eight-rank sequential A/B: three full updates per source, 4 CPU threads/rank, 16 environment workers/rank; median 87.9647s → 81.4110s (7.45% faster), all three paired wins. Each update has 245,760 interactions and 480 PPO optimizer steps; measured training time excludes final validation/checkpoint work.

Experiment and backups: `runs/patch_gnn_keycache_resume_20261014_195826_483431/`. Model source SHA256: `49a47378af268c8ce4e0496d5fa862d22596b9215b6dc4ca960687347370ec1d`.


## GNN cache-hit CPU key plan - 2026-10-14

The normal inference bank checks its existing identity memo directly for plain owning read-only NumPy arrays, verifying both weakrefs and re-reading flags on each lookup. Misses, writable/non-owning inputs and subclasses use the existing content-key routine. Local bindings and dictionary setdefault preserve first occurrence order while reducing Python calls. GPU operations, model/PPO definitions, LRU eviction, capacity guards and content-ID semantics are unchanged.

Validation: 28,736-state key/representative/gather checks, mutation/view/subclass/weakref checks, exact CPU rollout/PPO/Adam/RNG/restore fingerprint, pytest, and one full 8-rank sequential update.

One uninstrumented update: 79.1137s; previously confirmed best 81.2873s, training time reduced 2.67%. Same 8 ranks, 4 CPU threads/rank, 16 environment workers/rank, 245,760 interactions and 480 PPO steps. This is a single new measurement, compared against the recorded prior best.

Experiment/backups: `runs/patch_bank_hit_20261014_214851_510368/`. Model SHA256: `90c2ce45a71202ac859ed949d63e120fc30448ba3b418f6a0506bcb4fe235192`.


## History index arrays - 2026-10-14

Keep before/after history IDs as contiguous int64 NumPy arrays. Concatenate per-task slices instead of boxing every index into Python lists and converting those lists back to arrays/tensors in the model. Stable first-occurrence order, ring windows, padding, state registration, GPU gather values and model/PPO calculations are unchanged.

Validation: exact full-sized 24,576-record packing and downstream index bytes; empty/mixed histories, ring wrap/rebuild, irregular nodes and fallback; exact CPU rollout/PPO/Adam/RNG/restore fingerprint; MUSA index conversion/gather; pytest; one full eight-rank sequential update.

One uninstrumented update: 76.1171s; previously confirmed best 79.1137s, training time reduced 3.79%. Same 8 ranks, 4 CPU threads/rank, 16 environment workers/rank, 245,760 interactions and 480 PPO steps. This is a single new measurement compared against the recorded prior best.

Experiment/backups: `runs/patch_history_indices_20261014_234824_514874/`. History source SHA256: `56938e63faf6ce702cbc2ccaad34ca01d7ec07341131635fa62dd295d4f528c8`.
Model source SHA256: `90c2ce45a71202ac859ed949d63e120fc30448ba3b418f6a0506bcb4fe235192`.


## Incremental rollout content-key planning - 2026-10-15

Resolve content IDs once for new collector-owned, owning, read-only snapshots. Gather prior IDs by history-registry indices and preserve first occurrence representatives with NumPy. Generic mutable/view inputs retain the original checks. Scope changes and content-table pruning invalidate plans. GNN bank allocation, LRU, GPU operations, PPO and CPU action RNG are unchanged.

Validation: actual-source CPU plan/pack/slot equivalence, full 64x384 histories, pruning/capacity/shape/fallback/scope tests; exact CPU rollout/PPO/Adam/RNG/restore fingerprint with confirmed incremental-path coverage; pytest; one full 8-rank update.

One update: 60.1049s vs recorded 76.1171s; time reduction 21.04%. 8 ranks, 4 CPU threads/rank, 16 environment workers/rank, 245760 interactions and 480 PPO steps. One candidate measurement.

Experiment/backups: `runs/patch_incremental_keys_20261015_112602_551247/`. Source hashes: {'pfn': {'path': 'pfn_network/models/pfn.py', 'old': '90c2ce45a71202ac859ed949d63e120fc30448ba3b418f6a0506bcb4fe235192', 'new': 'a3a27c94f20a495c3c5f0274688a9a0f425b00ac5cd9955ce9d9406e78b39157'}, 'history': {'path': 'pfn_network/models/incremental_history.py', 'old': '56938e63faf6ce702cbc2ccaad34ca01d7ec07341131635fa62dd295d4f528c8', 'new': '216a1b31ced72bc82392e2f7df8e2ac92fd3d99bfb88b9034a4b760a24541073'}, 'train': {'path': 'pfn_network/train_eval/train.py', 'old': 'b8900440535820aa1e03ed66a35fca95fcab9b7a429e8dc2e4f31bc06afbc6fd', 'new': '5c985a9e9b75fbefbe7ffde33d299fff988e1f688522a78f514c9d4e29c50a9a'}}
