# AD–NC 静息 EEG 分类与外部验证代码

本仓库整理已经完成外部测试的最终实现，仅包含自编代码、配置模板、非身份性特征字典及纯随机合成演示。真实 EEG、临床表、被试 ID、特征、逐人预测和研究模型均不公开。

完整 EEG、普通 PCA、cPCA 是三个并列的表征分支，均使用正则化、类别平衡的 Logistic Regression；各有加入年龄/性别的版本，并保留人口学基线。cPCA 的 target 为 AD、background 为 NC，不是完整 EEG 分类器的事后解释工具。最终选择包含 **alpha=0**，不能据此证明 NC 背景惩罚产生额外作用，也不能宣称 cPCA 优于普通 PCA。

## 安装与快速演示

在仓库根目录运行 PowerShell：

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-tested.txt
.\.venv\Scripts\python.exe -m pip install --no-deps -e .
.\.venv\Scripts\cpca-ad-nc.exe --help
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe examples\synthetic_demo.py
```

演示从随机数生成 96 个训练及 32 个独立测试样本，ID 均以 `SYN_` 开头，使用固定的完整 752 列。演示的小网格仅用于验证训练、冻结、重载和预测，不代表论文结果。不指定输出目录时自动使用临时目录。

## 真实数据入口

复制 `configs/config.example.toml` 为被 Git 忽略的 `configs/config.local.toml`，填写自有合法数据路径。所有相对路径以配置文件所在目录为基准，支持中文及空格。按顺序运行：

```powershell
.\.venv\Scripts\cpca-ad-nc.exe prepare --config configs\config.local.toml
.\.venv\Scripts\cpca-ad-nc.exe internal --config configs\config.local.toml
.\.venv\Scripts\cpca-ad-nc.exe freeze --config configs\config.local.toml
.\.venv\Scripts\cpca-ad-nc.exe external --config configs\config.local.toml
.\.venv\Scripts\cpca-ad-nc.exe interpret --config configs\config.local.toml
.\.venv\Scripts\cpca-ad-nc.exe report --config configs\config.local.toml
```

`all` 依次执行以上六阶段。仅外部预测使用 `predict`，需要外部年龄/性别和特征及私有冻结包，不需要诊断标签或开发集，不重新拟合任何参数。

支持两条输入路径：直接提供符合 schema 的被试级特征，或从已经预处理的 EEGLAB v7.3 HDF5 `.set`/配对 `.fdt` 提取。后者要求 500 Hz、REST、确认的微伏单位；在 metadata 中提供 `set_path`，运行 `features --cohort development`（或 `external`），再将该组配置中的 `*_features` 置为空字符串以使用生成的特征与 QC 表。不支持直接从任意 EDF 开始自动完成预处理。MATLAB 文件保留的是 EEGLAB 读取核验与数组导出，并非新写的特征提取器。

元数据按唯一 `subject_id` 联结，训练/评价要求 `group` 为 AD/NC、年龄完整、性别为女=0/男=1。特征名称、数目及顺序严格检查；MMSE 缺失不删除分类样本。`cognition` 为可选原始 MMSE 分析，`stability` 为固定参数条件下的稳定性分析。

## 数据与解释范围

自有临床/EEG 数据有访问限制。外部来源为 OpenNeuro [ds004504 v1.0.8](https://doi.org/10.18112/openneuro.ds004504.v1.0.8)，本地元数据固定版本与 CC0 信息已核实；不打包数据，也不自动替换为新版本。原始公共数据仍需兼容、可追溯的预处理。

16 导只解释电极或粗略头皮区域，不能直接称为 DMN 等皮层网络。不同模型使用各自的权重与 Haufe 模式；置换重要性不是可相加的贡献百分比。AD+NC 合并 MMSE 相关不能直接当作 AD 内严重程度证据，需单独查看 AD 组内及年龄/性别、教育敏感性结果。

本次已实际完成安装/导入、帮助信息、8 项测试、完整 schema 演示、合成数据六阶段运行；350 个已有外层模型与 7 个外部预测分支的私有对照通过，AUC 差为 0。没有为整理仓库重跑全部 EEG 或全部论文嵌套验证。详细方法、输入格式、参数和核验边界见 [复现说明](docs/reproducibility.md)。

## cPCA 方法与实现

本项目采用 Abid 等人（2018）提出的对比主成分分析（cPCA）方法，并将作者的官方仓库 [abidlabs/contrastive](https://github.com/abidlabs/contrastive) 列为参考实现。

本项目的 `cpca_fit` 函数位于 `src/cpca_ad_nc/modeling.py`，通过基于 QR 的子空间特征分解，求解目标组协方差减去 α 倍背景组协方差所对应的投影方向，并接入本项目的训练、交叉验证与冻结外部验证流程。方法引用和源码许可分别记录，具体说明见 [THIRD_PARTY_NOTICES](docs/THIRD_PARTY_NOTICES.md#cpca-method-and-implementation)。

参考文献：Abid A, Zhang MJ, Bagaria VK, Zou J. Exploring patterns enriched in a dataset with contrastive principal component analysis. *Nature Communications*. 2018;9:2134. [doi:10.1038/s41467-018-04608-8](https://doi.org/10.1038/s41467-018-04608-8)。

## 许可与归属

Peichen Jiang 有权许可的原创代码和文档采用标准 [MIT License](LICENSE)，版权署名为 **Copyright (c) 2026 Peichen Jiang**，覆盖本项目原创的流程组织、接口、测试、示例和文档。使用 PCA、cPCA、逻辑回归、Haufe 变换等既有方法不代表本项目提出了这些算法。

`cpca_fit` 因来源尚未核实而暂不纳入本次 MIT 授权的具体范围、上游许可历史及未修改的 [MIT–NSFG 许可副本](docs/contrastive_LICENSE.md)，统一见 [THIRD_PARTY_NOTICES](docs/THIRD_PARTY_NOTICES.md#source-code-licensing-and-evidence)。方法引用不能代替源码许可证据，本仓库不作无例外的“全部仅 MIT”声明。

依赖及第三方材料保留各自版权和许可。本软件许可不授权使用或再分发私有 EEG、临床资料、外部数据集或研究模型，其访问和许可条件单独适用。
