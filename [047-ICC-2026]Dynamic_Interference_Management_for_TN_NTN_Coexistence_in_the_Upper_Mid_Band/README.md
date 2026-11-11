# ICC 2026 TN–NTN 动态干扰管理复现

复现论文 Fig. 4–6 的干扰分布和基站活动率，并用独立场景比较 PPO 训练前后的表现。相对原文 18 条曲线，主模型的中位数平均绝对误差为 **6.28 dB**，补充模型为 **5.86 dB**。两套模型均未达到平均误差 ≤3 dB 的目标。

论文为 Pradyumna Kumar Bishoyi、Chia Chia Lee、Navid Keshtiarast 和 Marina Petrova 的 *Dynamic Interference Management for TN–NTN Coexistence in the Upper Mid-Band*，IEEE ICC 2026。复现以正式发表版为准。[正式发表版（DOI）](https://doi.org/10.1109/ICC59461.2026.11587308) · [arXiv 预印本](https://arxiv.org/abs/2602.10813)

## 运行

已在 Python 3.13 下验证。运行脚本需要从正式发表版 PDF 提取参考曲线和原图。仓库不附带 PDF，请先通过上方 DOI 获取正式发表版并放入项目目录，再执行以下命令。

```powershell
python -m pip install -r requirements.txt
python run_reproduction.py
```

- 默认加载两套随包模型，重新评估并生成图表、结果数据和核验记录。单独评估可加 `--profile bessel` 或 `--profile ascent-envelope`。
- 重新训练用 `python run_reproduction.py --train --episodes 800`。每套模型用种子 11、22、33，各训练 800 次完整过境，每次 160 步。训练均匀抽样五类任务。
- 确定性推理可加 `--inference deterministic --output <独立输出目录>`。如需复用模型，先把两套模型的 `models` 和 `training` 子目录按原结构复制到新目录。缺少权重时会启动训练。
- `python verify.py` 执行行为检查。`python official_ascent.py --check-saved` 核验已保存的官方单快照和三快照证据。`python official_ascent.py --help` 查看实际重跑入口。

## 复现结果与适用范围

| 模型 | 18 条曲线中位数的平均绝对误差 | 误差 ≤3 dB | 误差 ≤5 dB |
|---|---:|---:|---:|
| 主模型 · 圆孔径 UT | 6.28 dB | 7/18 | 7/18 |
| 补充模型 · ETSI 包络 UT 与 16×16 阵列 | 5.86 dB | 9/18 | 9/18 |

- 在相同输入下，加入 NLOS 分支使中位 INR 降低约 38–40 dB，加入 7 dB 噪声系数使 INR 降低 7 dB。[预算诊断](output/diagnostics/PHYSICS_DIAGNOSIS.md) · [同输入消融](output/diagnostics/frozen_ablation.csv)
- 主模型的 PPO 在三个密度场景中较无协调降低中位 INR 约 2.8 dB，基站活动率约 93%。中密度全 T1 场景的中位 INR 分别为无协调 −2.54 dB、ASCENT-derived −8.68 dB、PPO −5.30 dB。高密度下 PPO 仍有 55.99% 的用户快照超过聚合保护门限。当前结果与原文所述约 87% 活动率和 6–8 dB 改善仍有差距。[逐场景结果](output/results.json) · [补充模型结果](output/profiles/ascent_envelope/results.json)
- 三个独立中密度全 T1 场景使用相同动作随机数，PPO 的采样奖励从训练前的 0.7385、0.8699、0.7297 分别升至 0.8118、0.9434、0.7898。初始化接近最大功率和 6° 下倾。主模型三个种子的更新末价值解释方差为 0.959、0.977、0.979。[训练日志](output/training/) · [学习归因](output/verification.json) · [确定性评估](output/deterministic_policy_evaluation.json)
- 参考来自发表版 PDF 第 6 页的矢量曲线，读数精度约 ±0.2 dB。CDF 误差仅比较 −20 至 20 dB 图窗，图窗外分位点不作外推。Fig. 5 原图的 ASCENT 中位数约 −7.6 dB、Proposed 约 0 dB，与正文的优越性表述有出入，本项目沿用原图图例。[图 7 中位数对照](output/fig07_reference_medians.png) · [逐曲线差值](output/paper_comparison.csv) · [补充模型逐曲线差值](output/profiles/ascent_envelope/paper_comparison.csv)

以上数值记录随包实验。重新运行后的数值见各模型输出目录中的 `results.json`、`verification.json` 和 `paper_comparison.csv`。

## 论文参数与重建假设

Table I 给出 12 GHz、200 MHz、106 个 gNB、每小区 12–20 个 UE，卫星发射功率和增益为 48 dBm、38.3 dBi，T1/T2 峰值增益为 33/17 dBi，gNB/UE 最大发射功率为 33/23 dBm。20×20 km² 场景分别部署 24、40、120 个 NTN 用户。Fig. 4 使用全 T1。Fig. 5 的固定密度未明确，本项目选用中密度 $10^{-7}\,\mathrm{m}^{-2}$。

原始 106 站坐标、建筑与 LOS 选择、UT 完整接收方向图、噪声配置、奖励权重及 PPO 超参数均未公开。本次未找到 ICC 实验的官方代码附件。配置值及其来源见 [config.py](config.py)。

| 项目 | 采用的设置 |
|---|---|
| 场景与调度 | 默认合成站址，可通过 `bs_coordinates_csv` 接入含 106 行 `x_m,y_m` 的坐标 CSV。NTN 用户在固定人数下均匀落点。TN UE 最近站关联，每站形成 12–20 用户的服务池，每快照调度一条 UL 流。每站一个控制单元。Fig. 2 展示原文站点地图。 |
| 过境与传播 | 600 km 高度，160 个扫描角（10°–169°）。论文另有用 500 km 和 53° 说明快照时间的段落。主实验采用 Eq. (4) 的全 LOS 参考场景和无阴影设置。地面链路超过 5 km 使用对数距离外推，近地 UE–NTN 链路使用 UMa 高度锚定。NLOS 和阴影作为敏感性选项。 |
| 卫星链路与噪声 | 两套模型的卫星发射图均按 Eq. (3) Bessel。Eq. (6) 的 UT 接收图需另行假设。Eq. (7) 与 Eq. (9) 的单位口径按线性干扰功率求和及总热噪声 $kTB$ 解释。额外 NF 为 0，卫星额外损耗为 3 dB，上下行占空比为 0.75/0.25。零干扰样本以 −300 dB 标记。 |
| 主模型 | 噪声温度 290 K。UT 采用圆孔径假设，效率 0.65，背向相对峰值衰减 55 dB。gNB 采用 Eq. (5) 形式的抛物线方向图，峰值 15 dBi 与波束宽度见配置。[背瓣与传播敏感性](output/diagnostics/profile_sensitivity.csv) |
| 补充模型 | 噪声温度 200 K，取自官方 ASCENT 案例。UT 用 ETSI Class B 的 6° 以上接收包络（$40-25\log_{10}\theta$，48° 后取 −2 dBi），6° 以下按该式延拓并按 T1/T2 的 33/17 dBi 封顶。gNB 用 16×16 阵列和实际链路角度，与官方波束函数同输入最大差约 $4.2\times10^{-12}$ dB。[ETSI §4.2.8.2.2](https://www.etsi.org/deliver/etsi_en/303900_303999/303981/01.02.01_60/en_303981v010201p.pdf) · [同输入验证](output/diagnostics/official_function_comparison.json) |
| PPO | 共享 actor 与集中 critic，逐站控制上下行功率、下倾和活动状态。使用联合 clipped 目标、GAE、独立优化器和按 100 缩放的价值预测及目标。critic 使用折扣蒙特卡洛目标，截断时保留 bootstrap。固定保护门限 −6 dB，吞吐、活动和超阈惩罚权重分别为 1、0.5、1，吞吐按瞬时无干扰上界归一化。[PPO 原文](https://arxiv.org/abs/1707.06347) |

评估使用独立几何种子 91001、91002、91003 和动作种子 791001、791002、791003。各方法在同场景共享几何和传播随机量。误差条为场景与模型配对的样本标准差。原图参考值和目标活动率仅用于事后对比。活动率衡量基站在网比例，原文未给 TN 最小服务质量或吞吐约束。

## 官方 ASCENT 与目标基线

使用作者实验室官方仓库的固定提交 `356c09f4caabfed772c66e2498a1aa8c2223a8c1`，执行原数值函数的 Blacksburg 晴天静态案例（33 站，8644 条建筑记录构造出 8177 个对象）。适配器按 SHA 校验所需文件，离线执行原函数。缺失角度缓存由上游原优化函数补齐。上游 DSA 子目录提供 MIT 许可，根目录模拟器未发现明确许可，固定版本适配器可从官方仓库获取所需文件，核验证据保存在输出目录。[实验室项目页](https://nextgwirelesslab.org/project/swift-context-aware-spectrum-coexistence-design-and-implementation-in-satellite-bands-ascent/) · [固定提交](https://github.com/NextG-Wireless-Lab-Mason/Context-aware-Spectrum-Coexistence-Analyzer/tree/356c09f4caabfed772c66e2498a1aa8c2223a8c1) · [ASCENT 原文](https://arxiv.org/html/2402.05273v3)

| 官方执行口径 | INR |
|---|---:|
| 首快照 33 站干扰线性聚合 | 10.290772 dB |
| 三快照的聚合干扰线性均值 | 7.923468 dB |
| 原代码跨站与快照的线性均值 | −7.261671 dB |

原控制器用固定 33 项 INR 覆盖物理反馈。完整三快照产生 99 项反馈，移除覆盖后会被原接口拒绝。记录同时保留原样执行和按站取线性均值的显式适配。上游的角度换算、断点单位、遮挡循环和信道重叠检查存在实现疑点。[单快照](output/official_ascent/upstream_exact.json) · [三快照](output/official_ascent/full_case_3snapshots.json) · [控制器检查](output/official_ascent/checked_controller.json) · [文件来源与校验](output/official_ascent/provenance/)

目标实验的 ASCENT-derived 以逐站门限为基础。每站取对所有 NTN 接收端的最大贡献，超阈静默，降回门限后恢复。逐站门限和 PPO 的软惩罚都可能留下聚合超阈。逐场景超阈率与吞吐见 [results.json](output/results.json)。

## 文件与核验

- [run_reproduction.py](run_reproduction.py) 是主入口。[tn_ntn_model.py](tn_ntn_model.py) 构建环境，[baselines.py](baselines.py) 实现基线，[ppo.py](ppo.py) 训练，[simulation.py](simulation.py) 执行同输入评估。
- [paper_assets.py](paper_assets.py) 从发表版 PDF 提取两张场景图和参考曲线，并计算原图误差。[plotting.py](plotting.py) 生成图 3–7。
- [official_ascent.py](official_ascent.py) 执行固定上游案例，[diagnostics.py](diagnostics.py) 执行链路预算与敏感性诊断。
- [主模型输出](output/) 与 [补充模型输出](output/profiles/ascent_envelope/) 各包含图表、逐用户样本、控制轨迹、模型权重、训练日志与配置清单。完整资料包含 9 项行为检查、独立重跑的 32 项数据和图表对比、两套模型的权重重载检查，以及 ZIP 的逐文件校验。[独立复跑记录](output/reproducibility_check.json) · [文件核验](output/release_validation.json) · [打包清单](output/delivery_manifest.json)
