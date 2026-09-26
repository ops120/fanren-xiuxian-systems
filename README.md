# Fanren Xiuxian Systems (凡人修仙传系统合集)

> 凡人修仙传相关的小系统 / 工具合集，每个子目录为一个独立项目，自带 README 与依赖清单。

## 子项目

| 目录 | 项目 | 说明 |
|---|---|---|
| `凡人修仙传赞助花名册提取系统/` | [Fanren Zanzhu Huamingce Tiqu Xitong](凡人修仙传赞助花名册提取系统/README.md) | 视频片尾滚动赞助名单 OCR 提取：33,783 条用户名 / 时间 / 档位 → 4 列 CSV，全本地 CPU 运行 |

## 目录结构

```
凡人修仙传系统合集/                 # 总仓库（fanren-xiuxian-systems）
└── 凡人修仙传赞助花名册提取系统/       # 子项目：赞助花名册提取器
    ├── src/extract.py            # 提取脚本（smoke / full）
    ├── samples/                  # 试跑产物与效果图
    ├── output/                   # 交付数据（names.csv 等）
    ├── requirements.txt / LICENSE
    └── README.md
```

## 约束

- 各子项目独立安装与运行，进入对应目录按各自 README 操作。
- 新增子项目：根目录建文件夹，自带 `README.md`（含环境/使用/许可证），可选择性自带独立 LICENSE。

## 许可证

本项目基于 [MIT License](LICENSE) 开源（子项目可自带独立许可证）。

## 社区

本项目在 [LINUX DO](https://linux.do) 社区进行开源推广，感谢社区佬友的交流、反馈与建议。
