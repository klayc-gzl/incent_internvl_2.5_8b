# incent_internvl_2.5_8b
www2025多模态对话系统意图识别挑战赛

# 1.环境配置

## 1.1.训练环境配置

新建虚拟环境并进入:

```Bash
conda create --name xtuner-env python=3.10 -y
conda activate xtuner-env
```
"xtuner-env"为训练环境名，可以根据个人喜好设置，在本教程中后续提到训练环境均指"xtuner-env"环境。

安装与deepspeed集成的xtuner和相关包：

```Bash
pip install xtuner==0.1.23 timm==1.0.9
pip install 'xtuner[deepspeed]'
pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu121
pip install transformers==4.39.0 tokenizers==0.15.2 peft==0.13.2 datasets==3.1.0 accelerate==1.2.0 huggingface-hub==0.26.5 
```
训练环境既为安装成功。

## 1.2.推理环境配置

配置推理所需环境：

```Bash
conda create -n lmdeploy python=3.10 -y
conda activate lmdeploy
pip install lmdeploy==0.6.1 gradio==4.44.1 timm==1.0.9
```

"lmdeploy"为推理使用环境名。


# 2.XTuner微调实践

## 2.1.准备基本配置文件

```Bash
cd xtuner
conda activate xtuner-env  # 或者是你自命名的训练环境
```
原始internvl的微调配置文件在路径`xtuner/xtuner/configs/internvl/v2`下

## 3.4.开始微调🐱🏍

运行命令，开始微调：

```Bash
xtuner train /root/xtuner/xtuner/configs/internvl/v2/internvl_v2_internlm2_5_8b_finetune.py --deepspeed deepspeed_zero3
```


微调后，把模型checkpoint的格式转化为便于测试的格式：

```Bash
python /root/xtuner/xtuner/configs/internvl/v1_5/convert_to_official.py /root/xtuner/xtuner/configs/internvl/v2/internvl_v2_internlm2_5_8b_finetune.py /root/xtuner/work_dirs/internvl_v2_internlm2_5_8b_lora_finetune_incent/iter_30000.pth /root/xtuner/work_dirs/internvl_v2_internlm2_5_8b_lora_finetune_incent/lr35_ep10
```

# 4.开始部署预测

```Bash
python lmdeploy1_test.py
```
该代码运行结果是只有predict列，需要你用这一列直接替换之前提交过submit的一列。




