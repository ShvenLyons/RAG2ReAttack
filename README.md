# RAG2ReAttack
In order to verify different attacks against RAG

# 项目结构说明
```text
./main.py                  ← 主执行逻辑
./utils/
├── argparser.py           ← 参数解析工具
├── helpers.py             ← 辅助函数
./modules/
├── Evaluator.py           ← 评估模块
├── Index.py               ← 索引模块 
├── LM.py                  ← 基础大模型相关模块
├── RALM.py                ← 检索增强大模型相关模块
├── Llama-2-7b-chat-hf/    ← 存放Llama2模型参数及文件夹
./data/
├── private/               ← 作为RAG外部知识库的文档数据
├── io_output/             ← 通用输入输出数据      
├── eval_output/           ← 评测输出         
├── eval_input/            ← 评测输入
├── datastore/             ← 检索存储/缓存等
./keys/
├── mine.txt               ← 存放API Token Key
```

