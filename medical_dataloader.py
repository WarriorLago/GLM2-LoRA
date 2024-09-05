# 导入必要的库
import json  # 用于处理JSON文件
import os  # 用于处理操作系统相关的路径
from mindspore.dataset import GeneratorDataset  # 从MindSpore导入生成器数据集类
from mindformers.models.build_tokenizer import build_tokenizer  # 从Mindformers导入构建分词器函数
from mindformers.tools.logger import logger  # 从Mindformers导入日志工具
from mindformers.tools.register import MindFormerModuleType, MindFormerRegister  # 从Mindformers导入注册工具和模块类型

# 使用MindFormers的注册器注册数据集加载器类
@MindFormerRegister.register(MindFormerModuleType.DATASET_LOADER)
class MedicalDataLoader:
    # 创建新实例时调用的函数
    def __new__(cls, dataset_dir, phase, shuffle=True, origin_columns=None, max_seq_len=512, **kwargs):
        # 检查数据集文件是否存在
        if not os.path.isfile(dataset_dir):
            raise ValueError(f"{dataset_dir} does not exist.")

        # 检查阶段参数是否为"train"或"eval"
        if phase not in ["train", "eval"]:
            raise ValueError(f"Phase should be 'train' or 'eval'.")

        # 如果原始列为空，则设置为默认值
        if origin_columns is None:
            origin_columns = ["prompt", "answer"]

        # 检查原始列参数是否为长度为2的列表或元组
        if not isinstance(origin_columns, (tuple, list)) or len(origin_columns) != 2:
            raise TypeError(f"origin_columns should be a list or tuple with length 2, but got {type(origin_columns)} with length {len(origin_columns)}")

        # 创建MedicalDataset实例
        medical_dataset = MedicalDataset(dataset_dir, origin_columns, phase)

        # 打印数据集加载信息
        info = f"[DATASET] shuffle status is {shuffle}, phase is {phase}."
        logger.info(info)

        # 返回生成器数据集
        return GeneratorDataset(medical_dataset, origin_columns, shuffle=shuffle, **kwargs)

# 医疗数据集类
class MedicalDataset:
    def __init__(self, dataset_dir, origin_columns, phase="train"):
        # 检查数据集文件是否存在
        if not os.path.isfile(dataset_dir):
            raise ValueError(f"{dataset_dir} does not exist.")

        self.dataset_dir = dataset_dir  # 数据集目录
        self.phase = phase  # 阶段（训练或评估）
        self.prompt_column = origin_columns[0]  # 提示列
        self.response_column = origin_columns[1]  # 回答列

        self.examples = {self.prompt_column: [], self.response_column: []}  # 初始化示例字典
        self._load_data()  # 加载数据

    # 加载数据函数
    def _load_data(self):
        with open(self.dataset_dir, 'r', encoding='utf-8') as fp:
            data = json.load(fp)  # 加载JSON数据
            for i, entry in enumerate(data):
                conversations = entry.get("conversations", [])  # 获取对话列表
                prompt, response = self._build_prompt_response(conversations)  # 构建提示和回答
                if prompt and response:
                    self.examples[self.prompt_column].append(prompt)  # 添加提示到示例字典
                    self.examples[self.response_column].append(response)  # 添加回答到示例字典
                else:
                    logger.info(f"Drop {self.dataset_dir}:{i} due to invalid data")  # 记录无效数据日志

    # 构建提示和回答函数
    def _build_prompt_response(self, conversations):
        prompt = ""  # 初始化提示
        response = ""  # 初始化回答
        for turn in conversations:
            role = turn.get("from")  # 获取对话角色
            content = turn.get("value")  # 获取对话内容
            if role and content:
                if role == "human":
                    prompt += f"Human: {content}\n"  # 添加人类对话内容
                elif role == "gpt":
                    response += f"AI: {content}\n"  # 添加AI对话内容
        return prompt.strip(), response.strip()  # 返回去掉首尾空白字符的提示和回答

    # 获取数据集长度
    def __len__(self):
        return len(self.examples[self.prompt_column])

    # 获取数据集的某个元素
    def __getitem__(self, idx):
        return {
            self.prompt_column: self.examples[self.prompt_column][idx],
            self.response_column: self.examples[self.response_column][idx]
        }

# 主程序
if __name__ == "__main__":
    dataset_dir_train = '/home/ma-user/work/data/medical/train.json'  # 训练数据集路径
    dataset_dir_eval = '/home/ma-user/work/data/medical/dev.json'  # 评估数据集路径
    phase_train = "train"  # 训练阶段
    phase_eval = "eval"  # 评估阶段
    origin_columns = ["prompt", "answer"]  # 原始列

    # 创建训练和评估数据加载器
    train_loader = MedicalDataLoader(dataset_dir_train, phase_train, shuffle=True, origin_columns=origin_columns)
    eval_loader = MedicalDataLoader(dataset_dir_eval, phase_eval, shuffle=False, origin_columns=origin_columns)

    # 检查数据加载
    for data in train_loader.create_dict_iterator():
        print(data)  # 打印训练数据中的第一个数据项
        break  # 只打印一个数据项，之后跳出循环

    for data in eval_loader.create_dict_iterator():
        print(data)  # 打印评估数据中的第一个数据项
        break  # 只打印一个数据项，之后跳出循环
