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
