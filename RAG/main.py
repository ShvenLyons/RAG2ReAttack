# 导入自定义模块中的参数解析函数
from utils.argparser import get_args
# 导入辅助函数：随机种子设置、读取 JSON
from utils.helpers import fix_seeds, read_json
# 导入语言模型（Language Model）模块
from modules.LM import LM   
# 导入带检索增强的语言模型模块（Retrieval-Integrated Causal Language Model）
from modules.RALM import RICLM, EmbeddingRICLM
# 导入评估器模块
from modules.Evaluator import Evaluator
import os
import json
from tqdm import tqdm  # 用于进度条显示
import warnings
import re
def safe_filename(name):
    return re.sub(r'[<>:"/\\|?*]', '_', name)
warnings.filterwarnings("ignore")

# 主函数，接收多个参数组：用户参数、LLM参数、RIC参数、kNN参数、训练参数、数据参数
def main(my_args, llm_args, ric_args, training_args, data_args):
    # 固定随机种子以确保实验可复现
    fix_seeds(my_args.my_seed)
    # 如果任务是 I/O 模式（生成和保存模型输出）
    if my_args.task == "io":
        # 确保关键参数不为空
        assert my_args.api is not None
        assert llm_args.ollama_ckpt is not None
        assert llm_args.is_chat_model is not None
        assert data_args.io_input_path is not None
        assert data_args.io_output_root is not None
        print(f"\n[INFO] 使用模型后端：{my_args.api} | 模型名称：{llm_args.ollama_ckpt}\n")
        # 初始化语言模型对象
        lm = LM(my_args=my_args, llm_args=llm_args)
        print("\nLM 初始化完成...\n")
        # 使用 RICLM 包装语言模型，添加检索能力
        if my_args.index == "BM25":
            ric_lm = RICLM(ric_args=ric_args, data_args=data_args, lm=lm)
        elif my_args.index == "FAISS":
            ric_lm = EmbeddingRICLM(ric_args=ric_args, data_args=data_args, lm=lm)
        print("\nRICLM 初始化完成...\n")
        # 设置输出目录（按模型名分目录）
        model_name_safe = safe_filename(ric_lm.lm.model_name)
        io_results_dir = os.path.join(data_args.io_output_root, model_name_safe)
        os.makedirs(io_results_dir, exist_ok=True)
        # 加载输入数据（通常是一个 query 列表）
        js = read_json(data_args.io_input_path)
        print("\n目录、输入加载完成，开始生成响应...\n")
        # 对每条输入逐一生成响应
        for dict_item in tqdm(js):
            # 如果当前 ID 的输出结果已经存在，跳过（避免重复处理）
            if str(dict_item["id"]) + ".json" in os.listdir(io_results_dir):
                continue
            # 使用带检索的语言模型生成结果
            response = ric_lm.generate(query=dict_item["input"])
            # 获取生成的语言模型输出和检索文档信息
            lm_output = response["lm_output"]
            retrieved_docs_str = response["retrieved_docs_str"]
            # 保存为 JSON 文件
            file_to_save = os.path.join(io_results_dir, str(dict_item["id"]) + ".json")
            with open(file_to_save, "w") as f:
                json.dump({
                    "raw_query": dict_item["raw_query"],
                    "input": dict_item["input"],
                    "lm_output": lm_output, 
                    "retrieved_docs_str": retrieved_docs_str
                }, f, indent=4)

            # 加载输入数据（只有一条输入语句）
            input_data = ""
            # 使用带检索的语言模型生成结果
            response = ric_lm.generate(query=input_data)
            # 获取生成的语言模型输出和检索文档信息
            lm_output = response["lm_output"]
            retrieved_docs_str = response["retrieved_docs_str"]
            # 保存为 JSON 文件
            file_to_save = "response_output.json"  # 输出的文件名，可以根据需要调整
            with open(file_to_save, "w") as f:
                json.dump({
                    "input": input_data,
                    "lm_output": lm_output, 
                    "retrieved_docs_str": retrieved_docs_str
                }, f, indent=4)

    # 如果任务是“评估”模式（对输出结果进行评价）
    elif my_args.task == "eval":
        assert data_args.eval_input_dir is not None
        assert data_args.eval_output_dir is not None
        os.makedirs(data_args.eval_output_dir, exist_ok=True)
        # 遍历每个模型的输出目录
        for model_name in tqdm(os.listdir(data_args.eval_input_dir)):
            # 如果评估结果已经存在，跳过
            if os.path.exists(os.path.join(data_args.eval_output_dir, model_name + ".json")):
                continue
            # 收集该模型所有的 JSON 输出文件
            json_files = [os.path.join(data_args.eval_input_dir, model_name, f) 
                          for f in os.listdir(os.path.join(data_args.eval_input_dir, model_name)) 
                          if f.endswith(".json")]
            predictions_str, references_str, raw_querys = [], [], []
            print("\n 该模型所有的 JSON 输出文件收集完成\n")
            # 分别读取预测输出和参考答案
            for json_file in json_files:
                with open(json_file, "r") as f:
                    js = json.load(f)
                predictions_str.append(js["lm_output"])
                references_str.append(js["retrieved_docs_str"])
                raw_querys.append(js["raw_query"])
            print("\n 预测输出和参考答案读取完成\n")
            # 使用评估器计算指标（如 ROUGE、BLEU 等）
            evaluator = Evaluator(predictions_str=predictions_str, references_str=references_str, raw_querys=raw_querys)
            metrics = evaluator.compute_metrics()
            print("\n 评估完成\n")
            # 将评估指标写入对应模型的 JSON 文件
            with open(os.path.join(data_args.eval_output_dir, model_name + ".json"), "w") as f:
                json.dump(metrics, f, indent=4)
            print("\n JSON 文件写入完成\n")
    # 如果输入的任务类型不是支持的三种，抛出错误
    else:
        raise NotImplementedError
    
# 脚本执行入口
if __name__ == '__main__':
    # 解析并获取所有参数
    my_args, llm_args, ric_args, training_args, data_args = get_args()
    # 调用主函数执行任务
    main(my_args, llm_args, ric_args, training_args, data_args)
