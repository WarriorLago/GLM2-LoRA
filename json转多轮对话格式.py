"""
fastchat stanford alpaca 数据转换工具
"""

import argparse
import json
import pathlib

# 来自斯坦福 Alpaca 训练脚本的提示
PROMPT_DICT = {
    "prompt_input": (
        "下面是一条描述任务的指令，配有提供进一步上下文的输入。"
        "写一个适当完成请求的响应。\n\n"
        "### 指令:\n{instruction}\n\n### 输入:\n{input}\n\n### 回应:"
    ),
    "prompt_no_input": (
        "下面是一条描述任务的指令。"
        "写一个适当完成请求的响应。\n\n"
        "### 指令:\n{instruction}\n\n### 回应:"
    ),
}

def main(args_param):
    data_path = pathlib.Path(args_param.data_path)
    with data_path.open(encoding='utf-8') as f:  # 指定编码格式为 'utf-8'
        data = json.load(f)
    prompt_input, prompt_no_input = (
        PROMPT_DICT["prompt_input"],
        PROMPT_DICT["prompt_no_input"],
    )

    sources = [
        prompt_input.format_map(example)
        if example.get("input", "") != ""
        else prompt_no_input.format_map(example)
        for example in data
    ]

    targets = [example["output"] for example in data]

    new_data = []

    cnt = 1

    for s, t in zip(sources, targets):
        new_data.append(
            {
                "id": str(cnt),
                "conversations": [
                    {
                        "from": "human",
                        "value": s,
                    },
                    {
                        "from": "gpt",
                        "value": t,
                    },
                ],
            }
        )
        cnt += 1

    json.dump(new_data, open(args_param.output_path, "w", encoding='utf-8'), indent=2, ensure_ascii=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default=r'.\大作业数据集.json')
    parser.add_argument("--output_path", type=str, default=r'.\大作业数据集-多轮对话格式.json')
    args = parser.parse_args()
    main(args)
