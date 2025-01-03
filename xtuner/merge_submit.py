import pandas as pd

# 读取submit.csv和test_result.csv文件
submit_df = pd.read_csv("submit.csv")
test_result_df = pd.read_csv("test_results.csv")

# 检查两者行数是否一致
if len(submit_df) != len(test_result_df):
    raise ValueError("The number of rows in submit.csv and test_result.csv do not match.")

# 用test_result.csv中的predict列替代submit.csv中的predict列
submit_df["predict"] = test_result_df["predict"]

# 保存更新后的submit.csv文件，使用utf-8-sig编码格式
submit_df.to_csv("submit_updated.csv", index=False, encoding="utf-8-sig")

print("The predict column has been successfully replaced. The updated file is saved as submit_updated.csv.")
