from mindformers import AutoModel, AutoTokenizer, AutoConfig
import mindspore as ms
import os

ms.set_context(mode=ms.GRAPH_MODE, device_target="Ascend", device_id=0)

# **注意** LoRA微调模型替换成 “glm2_6b_lora”
config = AutoConfig.from_pretrained("glm2_6b_lora")

# 可以在此使用下行代码指定自定义权重进行推理，默认使用自动从obs上下载的预训练权重
config.checkpoint_name_or_path = "/home/ma-user/work/code/mindformers/scripts/mf_standalone/output/checkpoint/rank_0/glm2-6b-lora_rank_0-4_4.ckpt"

config.use_past = True
config.seq_length = 769
model = AutoModel.from_config(config)
tokenizer = AutoTokenizer.from_pretrained("glm2_6b")

# pre-build the network
sample_inputs = tokenizer(tokenizer.build_prompt("你好，请提取出姓名、性别、年龄、身份证号、诊断证明、纠纷经过、手术、科室、赔偿。"))["input_ids"]
sample_outputs = model.generate(sample_inputs, max_length=769)

stop_stream = False

welcome_prompt = "欢迎使用 ChatGLM2-6B 模型，输入内容即可进行对话，clear 清空对话历史，stop 终止程序"
fixed_prompt = "请提取出姓名、性别、年龄、身份证号、诊断证明、纠纷经过、手术、科室、赔偿。### Input:"

def main():
    past_key_values, history = None, []
    global stop_stream
    print(welcome_prompt)
    while True:
        query = input("\n用户：")
        if query.strip() == "stop":
            break
        if query.strip() == "clear":
            past_key_values, history = None, []
            os.system("clear")
            print(welcome_prompt)
            continue
        print("\nChatGLM：", end="")
        full_query = f"{fixed_prompt} {query} ### Response: "
        inputs = tokenizer(full_query)["input_ids"]
        
        outputs = model.generate(inputs, max_length=5120)
        decoded_output = tokenizer.decode(outputs, skip_special_tokens=True)
        
        if stop_stream:
            stop_stream = False
            break
        else:
            print(decoded_output, end="", flush=True)

    print("")

if __name__ == "__main__":
    main()
