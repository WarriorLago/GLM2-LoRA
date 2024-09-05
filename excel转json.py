import openpyxl
import json
import os


# 读取Excel文件
def read_excel(file_path):
    workbook = openpyxl.load_workbook(file_path)
    sheet = workbook.active

    data_list = []

    # 遍历每一行，提取第二列和第三列的信息
    for row in sheet.iter_rows(min_row=2, values_only=True):
        instruction = "请提取出姓名、性别、年龄、身份证号、诊断证明、纠纷经过、手术、科室、赔偿。"
        input_data = row[1]
        output_data = row[2]

        # 构造字典
        data_dict = {
            "instruction": instruction,
            "input": input_data,
            "output": output_data
        }

        # 添加到列表中
        data_list.append(data_dict)

    return data_list


# 保存为json文件
def save_to_json(data, json_path):
    with open(json_path, 'w', encoding='utf-8') as json_file:
        json.dump(data, json_file, ensure_ascii=False, indent=4)


# 主函数
if __name__ == "__main__":
    # Excel文件路径
    file_path = r'.\大作业数据集.xlsx'

    data = read_excel(file_path)

    # 设置json文件路径
    json_path = file_path.replace('.xlsx', '.json')

    # 保存为json文件
    save_to_json(data, json_path)
    print(f"数据已成功转换并保存为JSON文件：{json_path}")



