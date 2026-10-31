> 本文件是开发阶段审查记录。文内 work/ 路径属于原开发目录；本仓库入口和命令以根目录 README 为准。

# T-RECS 运行准备检查

检查日期：2026-10-31。结论：在锁定的兼容环境下，已通过两次 MF 和一次 Random 的完整配置运行及所列记录一致性检查。此结论不等于研究假设成立，也不代表升级到最新依赖后可用。

## 运行设置和结果

每次使用 500 用户、2000 内容、20 维固定真实属性、10 条推荐组成的列表，20 轮随机启动后再运行 100 轮。禁止重复消费，关闭兴趣漂移和新增内容。真实用户和物品属性使用 Dirichlet(0.5) 初始化。

MF 使用 10 个潜在因子，每次 ALS 拟合 5 次迭代；启动后初次训练，随后每轮训练，合计 101 次拟合。通过 reset_interactions=False 显式保留累积交互。没有修改 T-RECS 算法源码。

| 运行 | 用时 | 消费记录 | 训练次数 |
|---|---:|---:|---:|
| MF seed 123 | 39.036 秒 | 60000 | 101 |
| MF seed 124 | 35.432 秒 | 60000 | 101 |
| Random seed 123 | 42.420 秒 | 60000 | 不适用 |

时间包含模型初始化、内部断言及压缩日志保存，不含 Python 导入；不能把这组时间直接推广到不同的迭代数、规模或漂移设置。此 Random 运行用于程序检查，不是严格共享启动记录的配对效果实验。

## 已通过的检查

- 每步 500 个选择，每人每轮一个；所有编号有效，选择来自本轮推荐列表。
- 推荐列表不包含该用户已经消费过的内容；每人 120 次选择均不重复。
- MF 历史交互在初次训练时为 10000 条，结束时为 60000 条；与逐轮导出的消费记录逐项相同。
- 每次拟合后的用户和物品因子都与 LensKit 返回的编号对应。
- 每次拟合后预测分数均有限；每次训练预测均有变化。
- 真实内容属性和用户偏好保持不变，与本次 drift=0 的设计一致。
- 每个用户结束时仍有 1880 条未消费内容，没有耗尽整个内容池。

## 需要保留的限制

1. 无 GPU 要求；当前测试只覆盖这台电脑上的固定环境。Python 3.10.4，trecs 0.2.1，numpy 1.23.5，scipy 1.10.1，pandas 1.5.3，LensKit 0.14.4，numba 0.57.1。T-RECS 提交为 0de83a3b5039d768c86bc5f3a0cf58b2d4072920。
2. 五次 ALS 迭代是运行检查配置；未验证优化收敛或预测准确性。正式案例需检查分数是否有实际含义。
3. 作者实现会让尚未被交互覆盖的内容保留初始潜在向量。本次首次拟合覆盖 1971/2000 和 1981/2000 条内容；结束时覆盖 1991 和 1996 条。这是需要披露和进一步检查的冷启动行为，不能默认为完全解决。
4. 未消费内容充足不等于每位用户最喜欢的内容始终充足；正式分析仍需检查偏好内容消耗情况。
5. 没有验证 10 个以上种子、兴趣漂移网格、最新版本依赖、收缩检测结果或可视化界面。

## 可重复运行入口

脚本：work/check_simulator_readiness.py。

使用现有 work/trecs-venv 环境运行；设置 NUMBA_CACHE_DIR 为工作区内的缓存目录、OPENBLAS_NUM_THREADS=1。

结果目录：work/readiness_2026_10_31。包含环境摘要 summary.json、每次运行的结果 JSON，以及逐轮消费和推荐列表的 NPZ 文件。

## Update 中可以使用的表述

We verified T-RECS in a pinned compatible environment using two MF runs and one Random run, each with 500 users, 2,000 items, 20 startup steps and 100 evaluation steps. Checks confirmed retained interaction histories, valid non-repeated choices, aligned model indices and finite, changing predictions after retraining. Each run took approximately 35–43 seconds in the tested configuration. This establishes simulator readiness, not detection performance or model convergence; unseen-item initialization remains a documented limitation.
