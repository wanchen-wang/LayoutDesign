import numpy as np

# 使用完全相同的随机种子
rng = np.random.default_rng(42)

# 在 0 到 99 之间无放回抽取 30 个数字
train_pick = rng.choice(100, size=30, replace=False)

# 打印结果（按从小到大排序，方便你和文件夹对应）
print("被抽中的 30 个数据组索引为：")
print(sorted(train_pick))