# from lmdeploy import pipeline, TurbomindEngineConfig, ChatTemplateConfig
# from lmdeploy.vl import load_image
# import re 
# model = '/root/xtuner/work_dirs/internvl_v2_internlm2_5_8b_lora_finetune_incent/lr35_ep10'
# image = load_image('/root/LLaMA-Factory/data/mire_train/images/2132d7f217279259584175905d1036-0.jpg')
# pipe = pipeline(model,chat_template_config=ChatTemplateConfig(model_name='internvl2-internlm2'), backend_config=TurbomindEngineConfig(session_len=8192))
# response = pipe(('你是一个电商客服专家，请根据用户与客服的多轮对话判断用户的意图分类标签。\n<用户与客服的对话 START>\n用户: <http>\n客服: 对于这款商品，您有什么想要知道的吗？温馨提示：图片中的【促销参考价】仅作参考，请依据活动参与标准下单，最终价格以订单为准。\n用户: <image>\n客服: 如需<dxm:highlight>寻找相同款式的产品</dxm:highlight>来自线下店铺或应用程序的，可以在本店的【店铺主页】顶部的【搜索框】中输入商品标签上的【六位商品编号】；<dxm:highlight>如果没有搜索结果</dxm:highlight>则表示店铺没有上架该商品，<dxm:highlight>搜索指南</dxm:highlight>在下方的图片中。如果不知道商品编号，可以在【淘宝主页】顶部的【搜索框】中找到一个【<dxm:highlight>相机标志</dxm:highlight>】，点击进入，识别您拍摄的照片并输入关键词三福，即可找到相关商品。\n用户: 这是指一套包含四个吗？\n客服: 目前没有找到相关商品的信息，如果您想了解商品的<dxm:highlight>基本信息</dxm:highlight>，可以在【商品标题】、【规格参数】或【商品详情】页面中查看哦~您可以稍微动手查一下呢！\n<用户与客服的对话 END>\n以下是可以参考的分类标签为：[\"反馈密封性不好\",\"是否好用\",\"是否会生锈\",\"排水方式\",\"包装区别\",\"发货数量\",\"反馈用后症状\",\"商品材质\",\"功效功能\",\"是否易褪色\",\"适用季节\",\"能否调光\",\"版本款型区别\",\"单品推荐\",\"用法用量\",\"控制方式\",\"上市时间\",\"商品规格\",\"信号情况\",\"养护方法\",\"套装推荐\",\"何时上货\",\"气泡\"]\n', image))
# match = re.search(r'<结论>(.*?)</结论>', response.text)
# if match:
#     result = match.group(1)
#     print(result)
# else:
#     print("未找到结论内容")

# import json
# import csv
# from lmdeploy import pipeline, TurbomindEngineConfig, ChatTemplateConfig
# from lmdeploy.vl import load_image
# # from lmdeploy.vl.constants import IMAGE_TOKEN

# # 加载模型
# model = '/root/xtuner/work_dirs/internvl_v2_internlm2_5_8b_lora_finetune_incent/lr35_ep10'
# pipe = pipeline(model,chat_template_config=ChatTemplateConfig(model_name='internvl2-internlm2'), backend_config=TurbomindEngineConfig(session_len=8192))

# # 读取 JSON 文件
# with open('/root/LLaMA-Factory/data/mire_test/test_updated.json', 'r', encoding='utf-8') as f:
#     data = json.load(f)

# # 存储输出结果
# results = []

# for item in data:
#     instruction = item['instruction']
#     images = item['image']

#     # if len(images) == 1:
#     # 单图情况
#     image = load_image(images[0])
#     response = pipe((instruction, image))
#     results.append({'predict': response.text})
#     # else:
#     #     # 多图情况
#     #     loaded_images = [load_image(img) for img in images]
#     #     numbered_prompt = "\n".join([f"Image-{i+1}: {IMAGE_TOKEN}" for i in range(len(images))])
#     #     full_prompt = f"{numbered_prompt}\n{instruction}"
#     #     response = pipe((full_prompt, loaded_images))
#     #     results.append({'predict': response.text})

# # 将结果保存为 CSV
# with open('/root/LLaMA-Factory/data/mire_test/test_results.csv', 'w', encoding='utf-8', newline='') as csvfile:
#     fieldnames = ['predict']
#     writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

#     writer.writeheader()
#     writer.writerows(results)


import json
import csv
import random
from tqdm import tqdm
from lmdeploy import pipeline, TurbomindEngineConfig, ChatTemplateConfig
from lmdeploy.vl import load_image

# 加载模型
model = '/root/xtuner/work_dirs/internvl_v2_internlm2_5_8b_lora_finetune_incent/lr35_ep10'
pipe = pipeline(model, chat_template_config=ChatTemplateConfig(model_name='internvl2-internlm2'), backend_config=TurbomindEngineConfig(session_len=8192))

# 预定义答案范围
valid_responses = [
    "反馈密封性不好", "是否好用", "是否会生锈", "排水方式", "包装区别", "发货数量", "反馈用后症状", "商品材质", 
    "功效功能", "是否易褪色", "适用季节", "能否调光", "版本款型区别", "单品推荐", "用法用量", "控制方式", 
    "上市时间", "商品规格", "信号情况", "养护方法", "套装推荐", "何时上货", "气泡", "实物拍摄(含售后)", 
    "商品分类选项", "商品头图", "商品详情页截图", "下单过程中出现异常（显示购买失败浮窗）", "订单详情页面", 
    "支付页面", "评论区截图页面", "物流页面-物流列表页面", "物流页面-物流跟踪页面", "物流页面-物流异常页面", 
    "退款页面", "退货页面", "换货页面", "购物车页面", "店铺页面", "活动页面", "优惠券领取页面", 
    "账单/账户页面", "投诉举报页面", "平台介入页面", "外部APP截图", "其他类别图片"
]
valid_responses1 = [
    "反馈密封性不好", "是否好用", "是否会生锈", "排水方式", "包装区别", "发货数量", "反馈用后症状", "商品材质", 
    "功效功能", "是否易褪色", "适用季节", "能否调光", "版本款型区别", "单品推荐", "用法用量", "控制方式", 
    "上市时间", "商品规格", "信号情况", "养护方法", "套装推荐", "何时上货", "气泡"
]


# 读取 JSON 文件
with open('/root/xtuner/mire_data/test/test_updated.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# 存储输出结果
results = []
index = 0
# 使用tqdm显示进度条
for item in tqdm(data, desc="Processing Items"):
    instruction = item['instruction']
    images = item['image']

    image = load_image(images[0])

    # 调用推理函数并检查结果是否符合预定义答案范围
    response = pipe((instruction, image))
    
    # 尝试最多3次推理，如果仍然不符合，随机选择一个有效答案
    retries = 0
    while response.text not in valid_responses and retries < 3:
        response = pipe((instruction, image))
        retries += 1
    
    # 如果经过3次推理结果仍不在有效范围内，随机选择一个有效答案
    if response.text not in valid_responses:
        print('3次推理均不在其中')
        index += 1
        if instruction[0] == '你':
            response.text = random.choice(valid_responses1)
        else:
            response.text = "其他类别图片"

    results.append({'predict': response.text})

# 将结果保存为 CSV
with open('/root/xtuner/mire_data/test/test_results.csv', 'w', encoding='utf-8', newline='') as csvfile:
    fieldnames = ['predict']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

    writer.writeheader()
    writer.writerows(results)

print('违规次数')
print(index)



# 思维链方式
# import re
# import json
# import csv
# import random
# from tqdm import tqdm
# from lmdeploy import pipeline, TurbomindEngineConfig, ChatTemplateConfig
# from lmdeploy.vl import load_image

# # 加载模型
# model = '/root/xtuner/work_dirs/internvl_v2_internlm2_5_8b_lora_finetune_incent/lr35_ep10'
# pipe = pipeline(model, chat_template_config=ChatTemplateConfig(model_name='internvl2-internlm2'), backend_config=TurbomindEngineConfig(session_len=8192))

# # 预定义答案范围
# valid_responses = [
#     "反馈密封性不好", "是否好用", "是否会生锈", "排水方式", "包装区别", "发货数量", "反馈用后症状", "商品材质", 
#     "功效功能", "是否易褪色", "适用季节", "能否调光", "版本款型区别", "单品推荐", "用法用量", "控制方式", 
#     "上市时间", "商品规格", "信号情况", "养护方法", "套装推荐", "何时上货", "气泡", "实物拍摄(含售后)", 
#     "商品分类选项", "商品头图", "商品详情页截图", "下单过程中出现异常（显示购买失败浮窗）", "订单详情页面", 
#     "支付页面", "评论区截图页面", "物流页面-物流列表页面", "物流页面-物流跟踪页面", "物流页面-物流异常页面", 
#     "退款页面", "退货页面", "换货页面", "购物车页面", "店铺页面", "活动页面", "优惠券领取页面", 
#     "账单/账户页面", "投诉举报页面", "平台介入页面", "外部APP截图", "其他类别图片"
# ]
# valid_responses1 = [
#     "反馈密封性不好", "是否好用", "是否会生锈", "排水方式", "包装区别", "发货数量", "反馈用后症状", "商品材质", 
#     "功效功能", "是否易褪色", "适用季节", "能否调光", "版本款型区别", "单品推荐", "用法用量", "控制方式", 
#     "上市时间", "商品规格", "信号情况", "养护方法", "套装推荐", "何时上货", "气泡"
# ]


# # 读取 JSON 文件
# with open('/root/LLaMA-Factory/data/mire_test/test_kb.json', 'r', encoding='utf-8') as f:
#     data = json.load(f)

# # 存储输出结果
# results = []
# index = 0
# # 使用tqdm显示进度条
# for item in tqdm(data, desc="Processing Items"):
#     instruction = item['instruction']
#     images = item['image']

#     image = load_image(images[0])

#     # 调用推理函数并检查结果是否符合预定义答案范围
#     response = pipe((instruction, image))

#     match = re.search(r'<结论>(.*?)</结论>', response.text)
#     if match:
#         result = match.group(1)
#     else:
#         result = "error"

#     # 尝试最多3次推理，如果仍然不符合，随机选择一个有效答案
#     retries = 0
#     while result not in valid_responses and retries < 3:
#         response = pipe((instruction, image))
#         match = re.search(r'<结论>(.*?)</结论>', response.text)
#         if match:
#             result = match.group(1)
#         else:
#             result = "error"
#         retries += 1
    
#     # 如果经过3次推理结果仍不在有效范围内，随机选择一个有效答案
#     if result not in valid_responses:
#         print('3次推理均不在其中')
#         print(result)
#         index += 1
#         if instruction[0] == '你':
#             result = random.choice(valid_responses1)
#         else:
#             result = "其他类别图片"

#     results.append({'predict': result})

# # 将结果保存为 CSV
# with open('/root/LLaMA-Factory/data/mire_test/test_results.csv', 'w', encoding='utf-8', newline='') as csvfile:
#     fieldnames = ['predict']
#     writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

#     writer.writeheader()
#     writer.writerows(results)

# print('违规次数')
# print(index)