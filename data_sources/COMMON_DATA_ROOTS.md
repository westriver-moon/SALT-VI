# 公共数据目录

本项目不复制数据集到 `SALT-VI/`。服务器公共数据根目录为：

```text
/home/cgv841/datasets/
```

当前已知数据/派生数据入口：

| 数据或派生数据 | 公共路径 | 备注 |
| --- | --- | --- |
| SYSU-MM01 | `/home/cgv841/datasets/SYSU-MM01` | 原始 SYSU 数据 |
| RegDB | `/home/cgv841/datasets/RegDB` | 需以服务器实际目录名核验 |
| LLCM | `/home/cgv841/datasets/LLCM` | 需以服务器实际目录名核验 |
| SYSU SWinIR x2 + PMT 256 处理结果 | `/home/cgv841/datasets/derived/SYSU-MM01-swinir-x2-pmt256-v1` | RGB 与 IR 均为离线派生数据 |

正式实验配置应记录绝对路径、数据版本、文件清单哈希和生成配置；不把公共数据目录复制到项目中。
