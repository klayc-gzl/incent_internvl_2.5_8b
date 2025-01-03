# import pandas as pd

# # 读取CSV文件
# file1 = "submit_updated_8477.csv"
# file2 = "submit_updated.csv"
# data1 = pd.read_csv(file1)
# data2 = pd.read_csv(file2)

# # 提取所需行
# part1 = data1.iloc[:500]  # 第1-500行
# part2 = data1.iloc[1000:5500]  # 第1001-5500行
# part3 = data2.iloc[500:1000]  # 第501-1000行
# part4 = data2.iloc[5500:]  # 第5501-10000行

# # 合并数据
# merged_data = pd.concat([part1, part2, part3, part4], ignore_index=True)

# # 保存到新的CSV文件
# output_file = "merged_submit.csv"
# merged_data.to_csv(output_file, index=False, encoding="utf-8-sig")

# print(f"Merged file saved as {output_file}")



import pandas as pd
from collections import Counter

# 定义CSV文件名列表
file_names = [
    "submit_updated_8477.csv",
    "submit_updated_8503.csv",
    "submit_updated_8524.csv",
    "final_result_8611.csv",
]

# 读取所有CSV文件并存储到列表
csv_data = [pd.read_csv(file) for file in file_names]

# 确保所有文件的id列一致
for i in range(1, len(csv_data)):
    if not csv_data[0]['id'].equals(csv_data[i]['id']):
        raise ValueError(f"The 'id' columns in {file_names[0]} and {file_names[i]} do not match.")

# 初始化结果DataFrame
result_df = csv_data[0][['id', 'instruction', 'input', 'image']].copy()

# 投票机制选择predict列
predict_columns = [df['predict'] for df in csv_data]
result_predict = []

for predicts in zip(*predict_columns):
    # 统计当前行不同predict值的出现次数
    predict_count = Counter(predicts)
    # 找出出现次数最多的predict值
    most_common = predict_count.most_common()
    if len(most_common) > 1 and most_common[0][1] == most_common[1][1]:
        # 如果出现次数相同，选择第一个表格的结果
        result_predict.append(predicts[0])
    else:
        # 否则选择出现次数最多的结果
        result_predict.append(most_common[0][0])

# 将最终predict列加入结果DataFrame
result_df['predict'] = result_predict

# 保存最终结果到CSV
result_df.to_csv("final_result_all.csv", index=False)

print("Voting-based results have been saved to 'final_result.csv'.")
