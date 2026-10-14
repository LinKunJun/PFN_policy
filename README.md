已去掉标题中的“摩尔线程版”，并把“用户”相关表述改为作者/项目表述。完整 README 如下：

```markdown
# 网络干涉因果决策 PFN

目标为 Linux、Python 3.10、PyTorch 2.7.1 和 torch_musa 2.7.1。提供单卡及 DDP + MCCL 的八卡同步元训练。算法说明如下。

发布包不包含 `runs/`、`profiles/`、`logs/` 或已训练权重。

## 标准启动与管理

在已安装 PyTorch/torch_musa 2.7.1 的主机上，进入项目：

```bash
cd /home/mccxadmin/PFN_network_interference_musa_optimized
bash PFN_network_interference_musa_start.sh          # 后台启动 static + sequential 全量训练
bash PFN_network_interference_musa_start.sh status   # 查看 PID、输出与进度
bash PFN_network_interference_musa_start.sh logs     # 持续查看日志
bash PFN_network_interference_musa_start.sh raw      # 查看未过滤的日志
bash PFN_network_interference_musa_start.sh check    # 8 卡通讯自检
bash PFN_network_interference_musa_start.sh dry-run  # 打印启动命令但不启动
```

默认 `NPROC_PER_NODE=8`、`CPU_THREADS=4`、`ENV_WORKERS=16`；均可通过同名环境变量调整。

## 其他命令行运行方式

在已有摩尔线程基础镜像中、项目根目录运行：

```bash
bash scripts/install_musa.sh
source .venv-musa/bin/activate
python -m pfn_network.train_eval.run --config configs/smoke.json --stage all --device musa:0 --output runs/smoke
# 手动命令行训练（也可直接使用上面的标准启动脚本）
bash scripts/train_ddp.sh --config configs/default.json --output runs/ddp_train
```

CPU 诊断显式使用 `--device cpu`。多卡的全局任务数和 minibatch 必须能被卡数整除；八卡短测试使用 `configs/smoke_ddp.json`。依赖版本以 `package_versions.txt` 为准。

## 2. 源码结构

| 路径 | 内容 |
|---|---|
| `pfn_network/env/core.py` | 公共任务、状态、FC/CD 转移、合法掩码、环境接口 |
| `pfn_network/env/dgp.py` | ER/SBM 任务先验、候选表、独立封装的真实响应机制 |
| `pfn_network/env/score.py` | 事实 Ridge 示例估计器、静态评分、规划评分模型 |
| `pfn_network/models/pfn.py` | 残差 GCN、因果 Transformer、联合候选头、价值头 |
| `pfn_network/baselines/offline.py` | 候选表内 Greedy/Local Swap/GRASP，另有全空间版本 |
| `pfn_network/baselines/bandit.py` | Random、UCB、ε-greedy、冻结评分 |
| `pfn_network/baselines/planning.py` | 可达状态反向归纳 Exact DP、滚动规划 |
| `pfn_network/baselines/tabular.py` | Q-learning、事实频次模型 Dyna-Q、检查点 |
| `pfn_network/train_eval/train.py` | GAE/PPO、跨任务训练、验证选择、训练辅助函数与检查点恢复 |
| `pfn_network/train_eval/evaluate.py` | 真实模型评估、阈值校准、四组比较表、曲线绘图 |
| `pfn_network/train_eval/run.py` | 统一命令行入口：训练、校准、评估、绘图 |
| `tests/test_mechanisms.py` | 手算机制、信息边界、GAE、掩码及置换检查 |

## 3. 与文档保持一致的机制

### 暴露与三种价值

固定无向无自环图。对每个完整动作 `z` 计算 `E=Az/d`，以及严格多数 `S=(Az > floor(d/2))`。平票为 0；孤立节点 `E=S=0`，其 `mu[:, :, 1]` 无支持并以 NaN 表示。决策中只查询实际可达的类别。

明确区分：

- `predicted_value`：给定 `mu_hat` 的多数暴露评分。
- `V_mu`：评估器根据 Bernoulli(1/2) 设计积分得到的真实多数分组均值评分。
- `V_env`：实际精确暴露 `E` 下的预期环境奖励。

模拟 DGP 是本实现显式指定的实验先验，主文档未给定唯一的真实函数。五组潜在系数独立从 `N(0,0.5²)` 抽取；节点特征为标准正态。令 `x=[1,X]`，使用

\[
m_i(a,e)=0.2\tanh(x_i\beta_0)
+[0.5+\tanh(x_i\beta_1)]a
+[0.5+\tanh(x_i\beta_2)]e
+\tanh(x_i\beta_3)ae+\tanh(x_i\beta_4)e^2.
\]

独立节点噪声为 `N(0,sigma²)`，环境直接采样等价的标量均值噪声 `N(0,sigma²/N)`。任务内真实系数、图和候选表始终固定；reset 仅重置物理状态和按协议设置噪声流，不重抽真实任务。

### FC / CD / 无动态

- FC：`q=(L,f)`，`f_next=z`，本期奖励 `base - c_f * sum(f)/N`。不是 `f*z` 重复处理惩罚，不改变干预效应，也不限制动作。末期后不再收费。
- CD：`q=(L,h)`，处理节点下期 `h=2`，未处理节点减一至 0。处理后两个完整时段不可再处理，第三个时段恢复。没有疲劳成本。
- P2：`dynamics=NONE`，物理记忆始终全零；`L` 只是本次 bandit 运行的计时。
- 每期预算重置。默认所有主表用 `at_most`，保证 FC/CD/PFN 共用含全零动作的候选表。公共任务与全空间静态函数另支持 exact-count；`B>0` 时 exact-count 候选表不能同时包含全零，因此不用于默认跨机制比较。

### 缓存与信息边界

`reset()` 一次向量化生成整个 `M×N` 暴露缓存与 `M` 个基础均值的私有缓存。`step(action_id)` 对奖励只查一个标量，再施加 FC 成本和噪声，不重算候选暴露，也不返回真值。

奖励/暴露**按动作索引取值相对 M 为 O(1)**；生成完整合法掩码仍为 O(MN)，复制节点状态为 O(N)，因此不宣称整个 step 是常数复杂度。

环境采用 Gymnasium 风格的五元组，但无需安装 Gymnasium：

```python
obs, info = env.reset(seed=seed)
next_obs, reward, done, truncated, info = env.step(action_id)
```

`reward` 是 Python `float`；动态/无动态真实环境含观测噪声。`info={}`，`truncated=False`。非法动作在抽噪声和推进状态前被拒绝，并计数。

观测包括 `pfn_state`（NumPy 节点矩阵）、`tabular_state_key=(L, tuple(memory))`、长度为 M 的布尔 `valid_action_mask`，以及公开的图、候选表、候选暴露、动态类型和时域。候选暴露只由已知图和动作计算，不是未观测收益。终止状态 mask 全 False；不会再送入策略采样。

`EnvModel` 仅由模拟器、实验事实数据生成器、评估器以及显式标记的 Oracle Exact DP 使用。神经网络和查表学习器不接收 task bundle、环境对象、潜在系数或均值缓存；训练收集器负责环境交互，把可见数据传给学习器。Python 私有属性用于结构隔离，不作为恶意代码的安全沙箱。

## 4. 候选表与 P1 公平性

默认 N=30，M=48。候选表由独立于真实结果和预测评分的随机节点排列前缀生成：包含全零，并覆盖不同处理人数与组合。多个前缀链方便静态单点添加搜索；节点顺序、候选顺序和动作编号在整次测试内不变。

**表 1 的 Greedy/Local Swap/GRASP 都限制在同一个固定候选表。** 添加、删除、交换必须落到已有候选行。Greedy 从全零开始，最大可用单点增益非正即停止；Local Swap 允许候选表内单点添加/删除和一换一；GRASP 使用 RCL 随机构造、随机候选起点和局部改进，并保留 greedy incumbent。诱导邻域可能不连通，GRASP 随机起点允许访问其他分量。

另提供 `greedy_full`、`local_swap_full`、`grasp_full`，在全部预算可行分配上实现文档中的标准组合搜索，含可选两点添加/删除。这些函数直接返回分配向量，不混入表 1，也不冒充与固定候选 PFN 具有相同动作空间。全空间搜索采用清晰的完整评分重算，没有实现增量邻域加速或 MILP。`Catalog exact` 枚举固定 M 行，是候选表内的精确静态参照，不能称为全空间最优。

## 5. 静态预测表与两类 PFN

附件没有提供已有静态 PFN 模型或权重。因此默认先在每个任务执行 `estimation_rounds=64` 次 iid Bernoulli(1/2) 随机实验，每次只生成该次执行分配的事实节点结果，再训练跨节点共享的 Ridge 四类别回归器。它是可替换的上游示例估计器，**不是静态 PFN**，不读取真实四臂表。对应实验成本单独记录，不提供给 P4 学习器。

若已有 `mu_hat`，设置 `prediction_dir`。目录下 `task_0.npz` 等文件须包含：

```python
np.savez("task_0.npz", mu_hat=mu_hat, adjacency=A, features=X)
# mu_hat.shape == (N, 2, 2); 轴顺序 [node, own treatment, majority exposure]
```

图、特征和节点顺序必须与生成任务精确匹配。测试任务可以通过 `sample_task(prior, seed, "test", index)` 预先重建；全部节点顺序都按数组行对应。

- **静态 Amortized PFN**：独立训练 checkpoint；T=1，所有方法看到同一个 `mu_hat`，训练 reward 是已知预测目标 `V_hat`。这是文档静态优化特例，确定性目标不加人为噪声。部署决策后仍调用真实 env.step，并由评估器计算 `V_env`。
- **序贯 Amortized PFN**：在 FC/CD 训练任务混合先验上，仅用有噪声的事实标量反馈训练；`mu_hat` 输入槽全零且标记不可用。测试时权重冻结，只增加事实上下文。
- **PPO From Scratch**：同一模型架构与 PPO 超参数，从随机权重开始，只使用当前测试任务的预算内交互更新；不读取静态评分。

外部估计器分布可能与默认 Ridge 元训练分布不同，必须作为分布变化报告；若要完全匹配，应在元训练中替换 `prediction_for` 为同一个上游预测流程。

## 6. 神经策略与 PPO

节点输入为 `X, degree/(N-1), f, h/2, availability, B/N, L/T`，另有静态实验专用预测槽和可用标记。动力学类别、预算类型、成本、时域为公共全局输入。先 MLP，再残差 GCN；节点不使用 ID 位置嵌入。

每条事实历史 token 包含执行前/后图状态、按完整分配聚合的节点表征、暴露表征、标量奖励、done、episode 内时间和相对 episode 编号。Transformer 使用严格的上三角布尔因果 mask。当前状态作为最后 query token，不包含当期未观测 reward。时间位置嵌入仅作用于轨迹序列。不同任务清空 context；同一任务可保留前几个 episode 的事实记录，默认配置最多 96 条（ModelConfig 类默认值为 24）。

候选得分严格使用完整联合动作：

\[
\ell_a=C_a\log\sigma(v)+(1-C_a)\log\sigma(-v)+g_\theta(I,C_a).
\]

`logsigmoid` 避免概率饱和后的 log(0)。联合头使用处理/未处理节点池化、精确/多数暴露池化、被共同处理的边比例和全局历史。合法性处理为 `logits.masked_fill(~mask, -1e9)` 后创建 `Categorical`。不先 softmax 再乘 mask。整个候选分配是一次环境动作，不按节点累计多个“交互”。

PPO 用冻结旧策略采样完整 episode，保留旧联合 log probability、当时的 mask、原始可见状态与历史、已执行 action id 和完整分配。更新时重新编码这些原始输入，不替换旧 log probability，不重新采样动作。关闭 dropout，以保持概率比的定义。变长样本采用 padding 和 batched Transformer，历史记录及图状态打包复用。实现使用基础张量图聚合和显式注意力；DDP 在每次全局 minibatch 上同步梯度，未开启 AMP。

GAE 按 episode 倒序计算，终止 `L=0` 的 **bootstrap value=0**。最后一步训练 target 是实际末步奖励 `R_{T-1}`，不是强行设为 0；把末步奖励 target 设为 0 会丢失真实收益。返回、优势与 value target 均不跨 episode 传播。代码只收完整 episode，因此没有 rollout 中途截断。policy/value/entropy 损失均保留 `gamma**t` 权重。

## 7. 规划和传统 RL

Exact DP 先枚举初始状态可达的各时域层，再从 L=0 反向归纳。FC 只需上一期分配；CD 在本设定中可达记忆由最近两次执行动作决定，固定候选表可以显著减少相对 `3^N` 的状态数，但大 M 时搜索仍昂贵。默认限制 500,000 状态与 120 秒；超限输出状态和空价值，不改用近似解冒充 Exact DP。

滚动规划深度 `min(d,L)`，截断边界 value=0，每次执行首动作后重规划。表 3 的 Planned Value 累计完整执行时域内模型奖励，**不是**局部深度 d 搜索值。普通规划只访问预测评分，Oracle Exact DP 另行标记额外真实模型信息。

FC 的理论核对：exact-count 时短视已最优；at-most 时 d=2 已足够。代码不会预设复杂方法必须更优。CD 手算例 `b=[0,3,3,5]`，T=3，短视为 5，精确最优为 6。

Q-learning 使用包含 L 的不可变 tuple key；探索、bootstrap 都限定合法动作，terminal target 仅含即时奖励。Dyna-Q 仅记录实际访问的 `(q,a)`：访问次数、奖励和、联合 `(q_next,done)` 频次及条件奖励均值。虚拟更新先均匀抽已访问键，再按事实频率抽后继；不查询模拟器补全未知项。动作与模型采样 RNG 分离，K_D=0 与 Q-learning 同流程。评估不会建立新 Q 行或改变训练 RNG；未见状态按全零 Q，并使用独立随机流在合法并列动作中选择。

## 8. 评估和输出解释

图、响应机制和候选表在每个测试任务内固定。训练、验证、测试用按 latent task 分离的 SeedSequence 命名空间；重复种子仅改变策略采样或观测噪声。同任务的不同动态设置始终属于同一个 split。所有方法使用相同的每轮环境噪声种子；因动作分化不强求轨迹相同。

| 表 | 方法与关键输出 |
|---|---|
| 1 | Greedy、Local Swap、GRASP、T=1 PFN、Catalog exact；V_hat / V_env / V_mu、预测 gap、决策时间 |
| 2 | Random、UCB、ε-greedy、Frozen score；累计 pseudo-regret、累计有噪声 reward、每轮决策时间 |
| 3 | 分 FC/CD：Myopic、d=2、Exact DP、Oracle Exact DP；计划模型价值、实际有噪声回报、实际期望回报、规划时间和状态 |
| 4 | 分 FC/CD：Q-learning、Dyna-Q、同架构从零 PPO、冻结 PFN；评估长期回报、达标交互次数、未达标状态、无效动作数、额外模型/梯度更新与耗时 |

P2 UCB 先按固定动作顺序访问未见臂，之后使用文档中的 `c*sigma/sqrt(N)*sqrt(2*log(max(2,t+1))/n_a)`。ε-greedy 未见臂均值初始化为 0、并列选最小索引。Pseudo-regret 只在 evaluator 使用真实候选均值计算，不使用最大观测 reward。

P4 包括 0 次适应交互的评价；此时评价 episode 自身仍允许观察已执行动作的奖励。横轴只计评价前的适应交互，评价期间的全部事实交互另列 `evaluation_interactions`，不回流到训练/适应 context。每个评价 episode 从同一份适应 context 的独立副本开始。所有方法用相同评价 seeds 和完整有限时域；PFN context-only 适应也逐次计入真实预算。

目标是验证任务上 `mean(Random) + fraction*(mean(Oracle optimum)-mean(Random))` 的**绝对回报**，默认 fraction=0.8，测试前固定。这个全局阈值对某些异质测试任务可能不可达，照实报告 not reached；绝不用测试最优值另设更容易达到的阈值。首次超过阈值是固定评价网格上的估计，不是插值精确样本数。

`interactions_to_target` 对未达标记录为 NaN，汇总均值仅在达标样本中计算；必须同时看 `target_reached` 达标比例、原始 `target_status` 和有限交互上限，不能单独据条件均值排名。未做基于测试结果的早停；最后回报固定取预算终点，不取测试曲线最大值。

主汇总先按任务内独立种子平均，再跨独立任务给 mean/std/count；只有一个任务时 std 为空，不能据重复噪声种子声称跨任务泛化。配对回报差另存 CSV。

典型输出：

- `table{1..4}_raw.csv`、`table{1..4}_summary.csv`；异常 Exact DP 的失败原因在 raw 表。
- `bandit_traces.csv`、`learning_curves.csv`、`evaluation_episodes.csv`、`paired_return_differences.csv`。
- `task_*.npz` 保存图、X、节点顺序、固定候选表和预测输入；真实机制由固定配置及 task seed 重建。
- `preparation_costs.csv` 分列任务/候选生成、上游预测、评分与环境缓存构建；决策计时不把这些共享前处理隐藏在算法收益里。
- `*_training_cost.json` 单列累计 meta 训练交互、训练时间、静态实验成本；`checkpoint_provenance.json` 记录选中 checkpoint 的训练成本和监督条件。
- `config.json`、`runtime.json` 记录完整配置与实际运行的 Python、PyTorch、torch_musa、GPU 信息；`checkpoints/` 保存新任务学习结果。

MUSA 决策计时在前后 synchronize，CPU 基线计时不包含无关 GPU 等待。P4 training_seconds 包含收集事实数据与更新/上下文处理，不含独立评价。神经推理时间包含图、历史编码和 CPU→设备张量转换；没有声称 O(1) 推理。当前预训练成本计时包括任务生成/数据采集/优化，不含验证、写盘及总进程启动开销。

可选绘图和检查：

```bash
python -m pfn_network.train_eval.run --stage plot --results runs/evaluation
python -m pytest tests -q
```

图中阴影为先种子平均、再跨任务的标准差。测试覆盖 FC/CD 手算值、非法动作不消耗噪声、step 不重算暴露、终止 GAE、Dyna 频次、K=0 一致性、评估只读、掩码有限梯度和节点置换一致性。

## 9. 单任务恢复与范围

`train_eval/train.py` 中的 `continue_tabular(checkpoint, bundle, additional_episodes, destination)` 和 `continue_scratch(checkpoint, bundle, config, additional_intervals, destination, device)` 可从完整 episode / PPO 更新边界继续。`bundle` 应通过原配置、root seed、test task index 与 dynamics 重建。检查点绑定图、特征、节点顺序、候选表、预算、时域与动态规则；环境噪声按原 episode 序号重建。PPO 的 `.context.pkl` 是配套事实历史文件，必须与 `.pt` 一起保留。仅加载可信来源的 pickle/checkpoint。

本实现覆盖项目指定的固定候选主方法。主文档中的 autoregressive prefix、全空间 PFN decoder、MILP、行为克隆专家预训练，以及全套 OOD/架构消融属于额外实验，不混入默认四表。已提供 `graph_layers=0`、`history=false`、`joint_head=false`、context 长度和图类型参数用于独立配置消融；跨任务预训练预算必须另行报告。

默认表 2 不包含历史 PFN bandit 的额外训练对照；P4 是混合 FC/CD 预训练，不能直接把这些权重称为已训练好的无动态 bandit 策略。所有结果都应保留其目标与信息条件。

End_to_End_PFN_Policy.tex 由本文档作者编写，已原样附带。模拟中的静态随机化识别，也不自动证明观察性日志下的动态策略可识别；本实现的 P4 依赖实际执行动作后获得的事实反馈。
```
