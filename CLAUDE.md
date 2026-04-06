# 项目规则

## 机器人部署原则

### 严禁在部署脚本中添加硬编码动作触发器

部署脚本（如 `deploy_ur5.py`）必须严格遵守以下原则：

**允许的内容：**
- 从相机和传感器读取观测数据
- 将观测数据传入训练好的模型进行推理
- 执行模型输出的动作（TCP 位置、夹爪指令）
- 日志记录和安全急停逻辑

**严禁的内容：**
- 基于 TCP 位置、关节角度等条件强制覆盖夹爪状态
- 伪造或修改传入模型的观测值（如手动设置 gripper.pos）
- 添加任何 if/else 位置触发器来补偿模型行为
- 在模型推理之外人工干预动作序列

**原因：**
所有动作决策（包括夹爪开合时机）必须完全由训练模型控制。
如果模型行为不正确，解决方案是修复训练数据或模型配置，
而不是在部署脚本中打补丁。

---

### 环境重置（允许）与模型干预（严禁）的区别

**允许 — 环境重置（policy 循环开始之前）：**
- `moveJ(HOME_JOINTS)`：将机器人移动到训练起始位置
- 物理开合夹爪：确保初始状态与训练数据一致
- `policy.reset()`：清空 temporal ensembling 队列

这些操作等同于数据采集时"将机器人放到起始位置"，在 policy 循环外执行，不干预模型决策。

**严禁 — 在 policy 循环内干预：**
- 根据 TCP 位置/关节角度在循环内强制执行夹爪动作
- 修改传入模型的观测值
- 任何补偿模型行为的 if/else 逻辑

---

### 部署参数必须从训练数据集中自动提取，严禁硬编码

**受影响的参数：**
- `HOME_JOINTS`：机器人起始关节角度
- `MAX_STEPS`：单次任务的最大执行步数

**规则：**
这些参数依赖于训练数据集（每次重新采集数据后都可能变化），
因此**不得**在脚本中写死数值。必须在 `main()` 启动时从数据集动态加载：

```python
HOME_JOINTS, MAX_STEPS = load_deploy_params(DATASET_REPO_ID, DATASET_ROOT)
```

`load_deploy_params()` 从数据集中提取：
- `HOME_JOINTS` = 所有 episode 第 0 帧的关节角度均值（std < 0.0001 rad）
- `MAX_STEPS`   = 数据集中最长 episode 的帧数 + 5（防止循环的安全上限）

**原因：**
- `HOME_JOINTS` 错误会导致机器人从 OOD 位置启动，模型行为异常（案例：y轴5.7cm偏差导致动作循环）
- `MAX_STEPS` 过大会导致任务完成后模型继续运行并重复执行动作序列
- 硬编码数值在重新采集数据或换任务后必然失效，自动提取则始终与训练数据同步

---

## 数据采集与训练的标准工作流程

每次采集数据或训练后，使用以下技能（slash commands）代替手动脚本：

| 步骤 | 时机 | 命令 |
|------|------|------|
| 采集完成后 | `lerobot-record` 结束后 | `/after-recording` |
| 训练完成后 | `lerobot-train` 结束后 | `/after-train` |

**`/after-recording`** 自动执行：
- 检查每个 episode 的夹爪开合位置（close_x/z、open_x/z）
- 标记并删除偏差超过 2σ 的 outlier episode
- 重新打印清理后的数据集摘要

**`/after-train`** 自动执行：
- 验证 checkpoint 配置（state/action shape、chunk_size）
- 检查夹爪归一化（min=0, max=2，两种状态均存在）
- 从数据集计算 HOME_JOINTS 和 MAX_STEPS
- 自动更新 `deploy_ur5.py` 中的 `CHECKPOINT_PATH`
- 打印部署就绪摘要和 deploy 命令
