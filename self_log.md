# NAS 日志
>  本文件是NAS父级项目的学习日志 是本人的工作记录日志

## 文件目录树
- nas
    - data
        - cifar-10-... CIFAR10 数据集
    - runs
      - architect 网络架构文件 用于定制alpha(DARTS) 和beta(PC-DARTS)的优化过程
      - arg   参数配置文件
      - genotypes 搜索空间配置文件
      - main 主函数 训练
      - model_build 网络模型的搭建 gamma的优化过程
      - operations 实现搜索空间中的操作
      - utils 独立函数
## 2022.01.30

今天除夕前一天了 在这里把这个NAS代码(暂且叫Continue Mask NAS CM-NAS)重新归档整理一下


