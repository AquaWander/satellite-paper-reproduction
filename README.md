# 📡 Satellite Paper Reproduction

> 卫星通信领域论文复现代码合集
>
> 来自公众号 **「静水卫星漫步」**

本仓库整理了卫星通信相关论文的复现代码，每篇论文对应一个独立目录，包含完整的代码和说明文档。

## 写在前面

如今学术论文数量井喷，质量却参差不齐。对于刚踏入研究生阶段的同学而言，面对浩如烟海的文献，往往不知从何读起，好不容易读完一篇却发现收获寥寥——这种情况，相信很多同学都深有体会。

AI 模型想要性能出色，尚且需要高质量的训练数据；人的大脑又何尝不是如此？输入决定输出，读什么样的论文，很大程度上塑造着你的研究品味和方向。

基于这样的想法，本号致力于**甄选卫星通信领域的优质论文**，免去大家辨别论文好坏的困扰。每一篇都配有：

- **复现代码** — 作为一个引子，帮助你从代码层面理解论文核心思路（不保证完全复现，但足以引导你深入探索）
- **公众号学术笔记** — 梳理论文的关键脉络与思考

精选论文、动手复现、反复研读——假以时日，这门"武功"自然水到渠成。

---

本仓库所有内容均为**免费开源**，都是我利用业余休息时间维护的。做这件事的初衷，是希望能帮助到和我一样、在研究生初期迷茫焦虑、看论文看到晕头转向的同学们——哪怕只带来一点微光，也算值得。

个人精力有限，更新节奏全凭热情驱动。**您的 Star、关注和转发，是支撑这件事继续走下去的最大动力。** 谢谢大家！

## 已复现论文

| # | 论文 | 期刊 | 关键词 | 目录 |
|---|------|------|--------|------|
| 001 | Avoiding Self-Interference in Megaconstellations Through Cooperative Satellite Routing and Frequency Assignment | JSAC 2024 | 星座自干扰避免、协作路由、频率分配 | [`[001-JSAC-2024]Avoiding_Self-Interference...`](./[001-JSAC-2024]Avoiding_Self-Interference_in_Megaconstellations_Through_Cooperative_Satellite_Routing_and_Frequency_Assignment) |
| 002 | Joint Power Allocation and Beam Scheduling in Beam-Hopping Satellites: A Two-Stage Framework With a Probabilistic Perspective | TWC 2024 | 波束跳变、功率分配、两阶段框架 | [`[002-TWC-2024]Joint_Power_Allocation...`](./[002-TWC-2024]Joint_Power_Allocation_and_Beam_Scheduling_in_Beam-Hopping_Satellites_A_Two-Stage_Framework_With_a_Probabilistic_Perspective) |
| 003 | Beam Footprint Design, Scheduling, and Spectrum Assignment in Low Earth Orbit Mega-Constellations | TVT 2025 | 波束足迹设计、用户调度、频谱分配 | [`[003-TVT-2025]Beam_Footprint_Design...`](./[003-TVT-2025]Beam_Footprint_Design_Scheduling_and_Spectrum_Assignment_in_Low_Earth_Orbit_Mega-Constellations) |
| 004 | Modeling Interference From Millimeter Wave and Terahertz Bands Cross-Links in Low Earth Orbit Satellite Networks for 6G and Beyond | JSAC 2024 | mmWave/THz星间链路、干扰建模、6G | [`[004-JSAC-2024]Modeling_Interference...`](./[004-JSAC-2024]Modeling_Interference_From_Millimeter_Wave_and_Terahertz_Bands_Cross-Links_in_Low_Earth_Orbit_Satellite_Networks_for_6G_and_Beyond) |
| 005 | Analysis Method for Downlink Co-Frequency Interference in NGSO Satellite Communication Systems With Information Geometry | TWC 2025 | 信息几何、共频干扰分析、矩阵流形 | [`[005-TWC-2025]Analysis_Method...`](./[005-TWC-2025]Analysis_Method_for_Downlink_Co-Frequency_Interference_in_NGSO_Satellite_Communication_Systems_With_Information_Geometry) |
| 006 | Feasibility Analysis of In-Band Coexistence in Dense LEO Satellite Communication Systems | TWC 2025 | 同频共存、干扰分析、卫星选择策略 | [`[006-TWC-2025]Feasibility_Analysis...`](./[006-TWC-2025]Feasibility_Analysis_of_In-Band_Coexistence_in_Dense_LEO_Satellite_Communication_Systems) |
| 007 | Resource Allocation and Load Balancing for Beam Hopping Scheduling in Satellite-Terrestrial Communications: A Cooperative Satellite Approach | TWC 2025 | 多星协作、跳波束调度、DRL资源分配、ISL负载均衡 | [`[007-TWC-2025]Resource_Allocation...`](./[007-TWC-2025]Resource_Allocation_and_Load_Balancing_for_Beam_Hopping_Scheduling_in_Satellite-Terrestrial_Communications_A_Cooperative_Satellite_Approach) |
| 008 | Semantic Satellite Communications Based on Generative Foundation Model | JSAC 2025 | 语义通信、生成式基础模型、知识图谱 | [`[008-JSAC-2025]Semantic_Satellite...`](./[008-JSAC-2025]Semantic_Satellite_Communications_Based_on_Generative_Foundation_Model) |
| 009 | Cooperative Multi-Satellite and Multi-RIS Beamforming: Enhancing LEO SatCom and Mitigating LEO-GEO Intersystem Interference | JSAC 2025 | 多星协作、RIS波束赋形、LEO-GEO干扰抑制、RZF预编码 | [`[009-JSAC-2025]Cooperative_Multi-Satellite...`](./[009-JSAC-2025]Cooperative_Multi-Satellite_and_Multi-RIS_Beamforming_Enhancing_LEO_SatCom_and_Mitigating_LEO-GEO_Intersystem_Interference) |
| 010 | LLM-Aided Spectrum-Sharing LEO Satellite Communications | JSAC 2026 | LLM辅助频谱共享、中断概率、资源分配 | [`[010-JSAC-2026]LLM-Aided...`](./[010-JSAC-2026]LLM-Aided_Spectrum-Sharing_LEO_Satellite_Communications) |
| 011 | Achieving Covert Communications in Ultra-Dense LEO Satellite Systems by Exploiting Interference and Directional Uncertainty | JSAC 2026 | 隐蔽通信、干扰利用、方向不确定性 | [`[011-JSAC-2026]Achieving_Covert...`](./[011-JSAC-2026]Achieving_Covert_Communications_in_Ultra-Dense_LEO_Satellite_Systems_by_Exploiting_Interference_and_Directional_Uncertainty) |
| 012 | Space-Time Beamforming for LEO Satellite Communications: Enabling Extremely Narrow Beams | TWC 2026 | 空时波束赋形、超窄波束、LEO卫星通信 | [`[012-TWC-2026]Space-Time_Beamforming...`](./[012-TWC-2026]Space-Time_Beamforming_for_LEO_Satellite_Communications_Enabling_Extremely_Narrow_Beams) |
| 013 | Satellite Selection for In-Band Coexistence of Dense LEO Networks | TWC 2026 | 卫星选择、同频共存、密集LEO网络 | [`[013-TWC-2026]Satellite_Selection...`](./[013-TWC-2026]Satellite_Selection_for_In-Band_Coexistence_of_Dense_LEO_Networks) |
| 014 | Direct-to-Device Non-Terrestrial Communications: Ensuring Interference-Free GSO Coexistence | TCOM 2026 | 星地直连、GSO共存、动态用户关联、禁区切换 | [`[014-TCOM-2026]Direct-to-Device...`](./[014-TCOM-2026]Direct-to-Device_Non-Terrestrial_Communications_Ensuring_Interference-Free_GSO_Coexistence) |
| 015 | Beamforming Design and Satellite Selection for Realizing the Integrated Communication and Navigation in LEO Satellite Networks | TWC 2026 | 通导一体化、波束赋形、卫星选择、LEO网络 | [`[015-TWC-2026]Beamforming_Design...`](./[015-TWC-2026]Beamforming_Design_and_Satellite_Selection_for_Realizing_the_Integrated_Communication_and_Navigation_in_LEO_Satellite_Networks) |
| 016 | Coverage Diversity in Mega Satellite Constellations: A Stochastic Geometry Approach | TWC 2025 | 覆盖多样性、巨型星座、随机几何 | [`[016-TWC-2025]Coverage_Diversity...`](./[016-TWC-2025]Coverage_Diversity_in_Mega_Satellite_Constellations_A_Stochastic_Geometry_Approach) |
| 017 | Joint Resource Management and Load Balancing in Multi-Satellite Beam Hopping With Interference Suppression: An Energy Minimization Perspective | TWC 2026 | 多星跳波束、Lyapunov资源管理、能量最小化、负载均衡 | [`[017-TWC-2026]Joint_Resource...`](./[017-TWC-2026]Joint_Resource_Management_and_Load_Balancing_in_Multi-Satellite_Beam_Hopping_With_Interference_Suppression_An_Energy_Minimization_Perspective) |
| 018 | A Dynamic Co-Frequency Interference Analysis Model Based on Time-Elevation Interference Spectrum for NGSO Mega-Constellations | TWC 2026 | 共频干扰分析、时-仰角干扰谱、NGSO巨型星座 | [`[018-TWC-2026]A_Dynamic...`](./[018-TWC-2026]A_Dynamic_Co-Frequency_Interference_Analysis_Model_Based_on_Time-Elevation_Interference_Spectrum_for_NGSO_Mega-Constellations) |
| 019 | Semantic Communication Enabled 6G-NTN Framework: A Novel Denoising and Gateway Hop Integration Mechanism | TWC 2025 | 语义通信、6G非地面网络、去噪机制、网关跳集成 | [`[019-TWC-2025]Semantic_Communication...`](./[019-TWC-2025]Semantic_Communication_Enabled_6G-NTN_Framework_A_Novel_Denoising_and_Gateway_Hop_Integration_Mechanism) |
| 020 | Time-Division Spectrum Sharing and Coordination Between Beam-Hopping NGSO Satellites and Terrestrial Networks | WCL 2026 | 时分频谱共享、跳波束、星地协同 | [`[020-WCL-2026]Time-Division...`](./[020-WCL-2026]Time-Division_Spectrum_Sharing_and_Coordination_Between_Beam-Hopping_NGSO_Satellites_and_Terrestrial_Networks) |
| 021 | Joint Illumination Power and Band Allocation for Multi-Beam LEO Satellites With Beam-Hopping Using Mixed-Integer Linear Programming | TWC 2026 | MILP跳波束、功率与带宽联合分配 | [`[021-TWC-2026]Joint_Illumination...`](./[021-TWC-2026]Joint_Illumination_Power_and_Band_Allocation_for_Multi-Beam_LEO_Satellites_With_Beam-Hopping_Using_Mixed-Integer_Linear_Programming) |
| 022 | A Comprehensive Survey of Large AI Models for Future Communications: Foundations, Applications and Challenges | COMST 2026 | 大模型综述、AI通信、6G | [`[022-COMST-2026]A_Comprehensive...`](./[022-COMST-2026]A_Comprehensive_Survey_of_Large_AI_Models_for_Future_Communications_Foundations_Applications_and_Challenges) |
| 023 | Multi-Satellite Coordinated Beam Hopping for Interference Mitigation Under Tilted Beam Effects: A Graph-Theoretic Approach | WCL 2026 | 多星协调跳波束、干扰抑制、图论、倾斜波束 | [`[023-WCL-2026]Multi-Satellite...`](./[023-WCL-2026]Multi-Satellite_Coordinated_Beam_Hopping_for_Interference_Mitigation_Under_Tilted_Beam_Effects_A_Graph-Theoretic_Approach) |
| 024 | Large Language Models for Optimization in Next-Generation Wireless Network Management: A Survey | COMST 2026 | LLM优化、无线网络管理、综述 | [`[024-COMST-2026]Large_Language...`](./[024-COMST-2026]Large_Language_Models_for_Optimization_in_Next-Generation_Wireless_Network_Management_A_Survey) |
| 025 | Artificial Intelligence for Satellite Communication: A Survey | COMST 2026 | AI卫星通信、综述 | [`[025-COMST-2026]Artificial_Intelligence...`](./[025-COMST-2026]Artificial_Intelligence_for_Satellite_Communication_A_Survey) |
| 026 | Multi-Satellite Cooperative Communications for 6G: Fundamentals, System Design, and Applications | COMST 2026 | 多星协同通信、6G、综述 | [`[026-COMST-2026]Multi-Satellite...`](./[026-COMST-2026]Multi-Satellite_Cooperative_Communications_for_6G_Fundamentals_System_Design_and_Applications) |
| 027 | Downlink Performance of Cell-Free Massive MIMO for LEO Satellite Mega-Constellation | TMC 2026 | 无蜂窝大规模MIMO、LEO巨型星座、下行性能 | [`[027-TMC-2026]Downlink...`](./[027-TMC-2026]Downlink_Performance_of_Cell-Free_Massive_MIMO_for_LEO_Satellite_Mega-Constellation) |
| 028 | Multi-Satellite Collaborations: Conception, Merits, Mechanisms, and Prospects | COMST 2026 | 多星协作、天地一体化、综述 | [`[028-COMST-2026]Multi_Satellite...`](./[028-COMST-2026]Multi_Satellite_Collaborations_Conception_Merits_Mechanisms_and_Prospects) |
| 029 | Antenna Technologies for 6G: Advances and Challenges | TAP 2025 | 6G天线技术、综述 | [`[029-TAP-2025]Antenna...`](./[029-TAP-2025]Antenna_Technologies_for_6G_Advances_and_Challenges) |
| 030 | An Integrated Shared-Aperture Active Phased Array Enabling STAR LEO Satellite Communication | JSAC 2026 | 共口径相控阵、同时收发、LEO卫星 | [`[030-JSAC-2026]An_Integrated...`](./[030-JSAC-2026]An_Integrated_Shared-Aperture_Active_Phased_Array_Enabling_STAR_LEO_Satellite_Communication) |
| 031 | Secure Multi-Satellite Collaborations With ISAC | TWC 2026 | ISAC多星协同、安全通信、联合波束赋形 | [`[031-TWC-2026]Secure...`](./[031-TWC-2026]Secure_Multi-Satellite_Collaborations_With_ISAC) |
| 032 | Coverage and Rate Analysis of Follower-Based LEO Satellite Networks: A Stochastic Geometry Approach | TWC 2026 | 随机几何、Follower簇、覆盖与速率分析 | [`[032-TWC-2026]Coverage...`](./[032-TWC-2026]Coverage_and_Rate_Analysis_of_Follower-Based_LEO_Satellite_Networks_A_Stochastic_Geometry_Approach) |
| 033 | ISAC-Empowered LEO Constellation: Networked Sensing, Challenges and Opportunities | WCM 2026 | ISAC、LEO星座、网络化感知、综述 | [`[033-WCM-2026]ISAC...`](./[033-WCM-2026]ISAC-Empowered_LEO_Constellation_Networked_Sensing_Challenges_and_Opportunities) |
| 034 | Co-Existence Analysis of Terrestrial and Non-Terrestrial Networks in S-Band Using Stochastic Geometry | TCOM 2026 | 星地共存、随机几何、S波段 | [`[034-TCOM-2026]Co-Existence...`](./[034-TCOM-2026]Co-Existence_Analysis_of_Terrestrial_and_Non-Terrestrial_Networks_in_S-Band_Using_Stochastic_Geometry) |
| 035 | RFI Detection and Geolocation From LEO | TAES 2026 | 射频干扰检测、地理定位、LEO | [`[035-TAES-2026]RFI...`](./[035-TAES-2026]RFI_Detection_and_Geolocation_From_LEO) |
| 036 | Space Computing Constellation: System Architecture, Implementations, and Challenges | IOTJ 2026 | 空间计算星座、系统架构、星载计算 | [`[036-IOTJ-2026]Space...`](./[036-IOTJ-2026]Space_Computing_Constellation_System_Architecture_Implementations_and_Challenges) |
| 037 | Satellite Computing Network Construction: Optimal Computing Node Deployment in Multi-Layer LEO Mega-Constellations | TCOM 2026 | 卫星计算网络、计算节点部署、多层LEO | [`[037-TCOM-2026]Satellite...`](./[037-TCOM-2026]Satellite_Computing_Network_Construction_Optimal_Computing_Node_Deployment_in_Multi-Layer_LEO_Mega-Constellations) |
| 038 | Survey: Spectrum Sharing and Interference Management for 6G LEO Satellite-Terrestrial Network Integration | COMST 2025 | 频谱共享、干扰管理、星地融合、综述 | [`[038-COMST-2025]Survey...`](./[038-COMST-2025]Survey_Spectrum_Sharing_and_Interference_Management_for_6G_LEO_Satellite-Terrestrial_Network_Integration) |
| 039 | Split Learning-Based Channel Prediction for 6G-Enabled LEO Satellite Systems | TGCN 2026 | 拆分学习、信道预测、6G LEO | [`[039-TGCN-2026]Split...`](./[039-TGCN-2026]Split_Learning-Based_Channel_Prediction_for_6G-Enabled_LEO_Satellite_Systems) |

## 快速使用

```bash
# 克隆仓库
git clone https://github.com/AquaWander/satellite-paper-reproduction.git

# 进入某篇论文的目录，按 README 说明运行
cd "[001-JSAC-2024]Avoiding_Self-Interference_in_Megaconstellations_Through_Cooperative_Satellite_Routing_and_Frequency_Assignment"
python run_reproduction.py
```

## 依赖安装

各论文所需依赖略有不同，主要包括：

```bash
pip install numpy matplotlib scipy scikit-learn
```

部分论文可能还需要 `torch`、`seaborn` 等，详见各目录下的 README。

## 关于

**「静水卫星漫步」** 专注于通信与信号处理领域的技术分享，包括：

- 卫星通信前沿论文解读
- 关键算法代码复现
- 仿真实验与性能分析

如果觉得有收获，欢迎 **Star** ⭐ 本仓库，后续更新第一时间收到！

## 关注我们

- **公众号**：静水卫星漫步
- **GitHub**：https://github.com/AquaWander

---

> 感谢关注「静水卫星漫步」！如果觉得有收获，欢迎点赞、在看、转发三连！

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=AquaWander/satellite-paper-reproduction&type=Date)](https://www.star-history.com/#AquaWander/satellite-paper-reproduction&Date)
