from pathlib import Path
import torch
import pandas as pd

data_dir = Path(__file__).resolve().parent.parent / 'data'
data_dir.mkdir(parents=True, exist_ok=True)
data_file = data_dir / 'house_tiny.csv'
with data_file.open('w') as f:
    f.write('NumRooms,Alley,Price\n')
    f.write('NA,Pave,127500\n')
    f.write('2,NA,106000\n')
    f.write('4,NA,178100\n')
    f.write('NA,NA,140000\n')


# pd.read_csv：读取 CSV 文件，并把内容转换成 pandas 的 DataFrame 表格。
# data 的每一列都有一个列名，每一行代表一条房屋数据。
data = pd.read_csv(data_file)
print(data)

# iloc：按照“行号、列号”选取数据。
# : 表示选择全部行；0:2 表示选择第 0、1 列，不包含第 2 列。
# inputs 是输入特征：NumRooms 和 Alley。
# outputs 是输出结果：Price。
inputs, outputs = data.iloc[:, 0:2], data.iloc[:, 2]

# NaN 表示数据缺失。
# NumRooms 中有缺失值，需要用房间数的平均值进行填充，这个过程叫“插值”。
# mean：计算平均值。
# numeric_only=True：只计算数字列的平均值，不计算 Alley 这种文字列。
# fillna：把缺失值 NaN 替换成指定的值。
# 这里的平均值会按照列名 NumRooms 对应起来，只填充 NumRooms 列。
inputs = inputs.fillna(inputs.mean(numeric_only=True))
print(inputs)

# get_dummies：把文字分类列转换成机器学习可以使用的数字列，叫“独热编码”。
# 例如 Alley 中的 Pave 会变成 Alley_Pave 列：
# Pave -> True；缺失值 -> False。
# dummy_na=True：把原来的缺失值 NaN 也作为一种单独的类别保存。
inputs = pd.get_dummies(inputs, dummy_na=True)

# astype('float32')：把所有列统一转换成 32 位浮点数。
# 统一类型可以避免 pandas 生成 object 类型的数组，便于 PyTorch 读取。
inputs = inputs.astype('float32')
print(inputs)

# values：取得 DataFrame 或 Series 中的底层数据。
# torch.tensor：把 NumPy 数组转换成 PyTorch 张量。
# x 是输入特征，y 是要预测的房价结果。
# inputs 已经在上面转换成 float32，所以 x 可以被 PyTorch 正常读取。
x, y = torch.tensor(inputs.values), torch.tensor(outputs.values)
print(x, y)
