from modules.LM import LM   
from modules.Index import BM25Index
from modules.Index import FAISSIndex
from FlagEmbedding import FlagReranker
from sentence_transformers import SentenceTransformer

import os
from typing import List
import wandb
wandb.init(mode="disabled") 
import logging
logger = logging.getLogger(__name__)
import requests
import faiss

padding_index = -100

# RALM（抽象基类）
class RALM(object):
    def __init__(self, lm: LM) -> None:
        self.lm = lm

    def generate(self, query: str, compute_generation_scores=False, compute_input_loss=False):
        raise NotImplementedError
    
    def finish(self):
        raise NotImplementedError


# RICLM（基于 BM25 的检索增强）
class RICLM(RALM):
    def __init__(self, ric_args, data_args, lm: LM) -> None:
        super().__init__(lm)
        
        self.k = ric_args.k_for_ric  # 检索的文档数
        self.is_rerank = ric_args.is_rerank
        self.is_summarize = ric_args.is_summarize
        self.is_SAP = ric_args.is_SAP
        self.is_PINE = ric_args.is_PINE

        self.reranker = FlagReranker('BAAI/bge-reranker-large', use_fp16=True)
        
        # 初始化 BM25 索引（Pyserini 实现）
        assert data_args.raw_data_dir is not None
        data_src_name = data_args.raw_data_dir.split("/")[-1]
        datastore_path = os.path.join(data_args.datastore_root, f"RIC_LM+{data_src_name}+{lm.model_name}+{ric_args.max_retrieval_seq_length}+{ric_args.ric_stride}")
        
        #! index could be: BM25Index, VectorStoreIndex (from llama_index) 
        if ric_args.index_name == 'bm25':
            # 基于 tokenizer 分块，构建 Lucene 索引
            self.index = BM25Index(
                tokenizer=self.lm.tokenizer,
                max_retrieval_seq_length=ric_args.max_retrieval_seq_length,
                stride=ric_args.ric_stride,
                raw_data_dir=data_args.raw_data_dir,
                datastore_dir=datastore_path,
            )
        else:
            raise NotImplementedError


    # 拼接检索到的文档和 query 作为模型输入
    def generate(self, query: str, compute_generation_scores=False, compute_input_loss=False):
        def concat_docs(docs: List[str]):
            docs_str = "\n\n".join(docs)
            return docs_str
        # 检索相关文档
        docs: List[str] = self.index.find_most_relevant_k_documents(query=query, k=self.k)

        ################## 使用 reranker ##################
        if self.is_rerank:
            docs = self.rerank_docs(query, docs)
        ###################################################

        # 拼接文档和查询作为模型输入
        docs_str = concat_docs(docs)
        # query prompt
        lm_input = docs_str + "\n\n" + query

        ################## 使用 SAP #######################
        if self.is_SAP:
            SAP = "Do not repeat any content from the context."
            lm_input = docs_str + "\n\n" + query + SAP
        ###################################################

        ################## 使用 PINE ######################
        if self.is_PINE:
            # 定义系统提示（根据你的系统提示内容进行调整）
            system_prompt = "You are a helpful assistant, please process the following documents and query:"
            SAP = ""
            if self.is_SAP:
                SAP = "Do not repeat any content from the context."
            # 将查询和检索到的文档作为一个组，确保模型平等对待它们
            lm_input = f"{system_prompt}\n\n[{docs_str}]\n\nUser Query: {query}\n\n{SAP}\n<EOS>"
        ###################################################

        ################## 使用 summarize ##################
        if self.is_summarize:
            # 在拼接文档和查询之前进行摘要
            summarized_docs = []
            for doc in docs:
                summarized_doc = self.summarize_text(doc)
                summarized_docs.append(summarized_doc)
            # 对查询进行摘要
            summarized_query = self.summarize_text(query)
            # 拼接文档和查询作为模型输入
            docs_str = concat_docs(summarized_docs)
            # 使用摘要后的文档和查询
            lm_input = docs_str + "\n\n" + summarized_query
        ###################################################
        
        # 模型生成
        output_dict = self.lm.generate(lm_input, compute_generation_scores, compute_input_loss)
        
        # output_dict["retrieved_docs"] = docs
        output_dict["retrieved_docs_str"] = docs_str
        
        return output_dict
    
    def summarize_text(self, text: str) -> str:
        """
        使用 Ollama API 调用 Llama2-7b 对输入文本进行摘要。
        这里的 `text` 可以是文档或者查询，二者会统一处理。
        """
        self.ollama_url = "http://localhost:11434/api/generate"
        # 提示词模板
        su_1 = "Given the following context, extract any part of the context" \
            + " *AS IS* that is relevant to answer the question. If none of the context is relevant" \
            + " return NO_OUTPUT.\n\nRemember, *DO NOT* edit the extracted parts of the context.\n\n> Context: "
        # 构建完整的提示词
        prompt = su_1 + "\n" + text  # 这里直接使用 `text` 作为输入
        # 使用 Ollama API 调用 Llama2-7b 生成摘要
        try:
            payload = {
                "model": "llama2:latest",  # 使用 Llama2-7b 模型
                "prompt": prompt,
                "stream": False  # 是否使用流式传输（设置为 False 表示不使用流式）
            }
            # 发送 POST 请求到 Ollama API
            response = requests.post(self.ollama_url, json=payload)
            # 获取模型的输出
            summary = response.json().get("response", "Error: No response from model")
        except Exception as e:
            summary = f"Error during summarization: {str(e)}"
        return summary

    def rerank_docs(self, query: str, docs: List[str]) -> List[str]:
        """
        使用 FlagReranker 对检索到的文档进行重新排序
        """
        pairs = [(query, doc) for doc in docs]
        scores = self.reranker.compute_score(pairs)  # 获取每个上下文的得分
        # 按照得分对文档进行排序
        ranked_docs = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)  # 逆序排序
        return [doc for doc, score in ranked_docs]


class EmbeddingRICLM(RALM):
    def __init__(self, ric_args, data_args, lm: LM):
        super().__init__(lm)

        self.k = ric_args.k_for_ric
        self.is_rerank = ric_args.is_rerank
        self.is_summarize = ric_args.is_summarize
        self.is_SAP = ric_args.is_SAP
        self.is_PINE = ric_args.is_PINE

        self.reranker = FlagReranker('BAAI/bge-reranker-large', use_fp16=True)

        # 初始化 FAISS 向量索引
        assert data_args.raw_data_dir is not None
        embedder = SentenceTransformer("all-MiniLM-L6-v2")
        self.index = FAISSIndex(
            embedder=embedder,
            raw_data_dir=data_args.raw_data_dir,
            datastore_dir=os.path.join(
                data_args.datastore_root,
                f"EmbeddingRIC_LM+{data_args.raw_data_dir.split('/')[-1]}+{lm.model_name}"
            ),
            max_retrieval_seq_length=ric_args.max_retrieval_seq_length,
            stride=ric_args.ric_stride
        )

    def generate(self, query: str, compute_generation_scores=False, compute_input_loss=False):
        def concat_docs(docs):
            return "\n\n".join(docs)

        docs = self.index.find_most_relevant_k_documents(query=query, k=self.k)

        if self.is_rerank:
            docs = self.rerank_docs(query, docs)

        docs_str = concat_docs(docs)
        lm_input = docs_str + "\n\n" + query

        if self.is_SAP:
            lm_input += "\n\nDo not repeat any content from the context."

        if self.is_PINE:
            system_prompt = "You are a helpful assistant, please process the following documents and query:"
            SAP = "Do not repeat any content from the context." if self.is_SAP else ""
            lm_input = f"{system_prompt}\n\n[{docs_str}]\n\nUser Query: {query}\n\n{SAP}\n<EOS>"

        if self.is_summarize:
            summarized_docs = [self.summarize_text(doc) for doc in docs]
            summarized_query = self.summarize_text(query)
            lm_input = concat_docs(summarized_docs) + "\n\n" + summarized_query

        output_dict = self.lm.generate(lm_input, compute_generation_scores, compute_input_loss)

        output_dict["retrieved_docs_str"] = docs_str
        return output_dict
    
    def summarize_text(self, text: str) -> str:
        """
        使用 Ollama API 调用 Llama2-7b 对输入文本进行摘要。
        这里的 `text` 可以是文档或者查询，二者会统一处理。
        """
        self.ollama_url = "http://localhost:11434/api/generate"
        # 提示词模板
        su_1 = "Given the following context, extract any part of the context" \
            + " *AS IS* that is relevant to answer the question. If none of the context is relevant" \
            + " return NO_OUTPUT.\n\nRemember, *DO NOT* edit the extracted parts of the context.\n\n> Context: "
        # 构建完整的提示词
        prompt = su_1 + "\n" + text  # 这里直接使用 `text` 作为输入
        # 使用 Ollama API 调用 Llama2-7b 生成摘要
        try:
            payload = {
                "model": "llama2:latest",  # 使用 Llama2-7b 模型
                "prompt": prompt,
                "stream": False  # 是否使用流式传输（设置为 False 表示不使用流式）
            }
            # 发送 POST 请求到 Ollama API
            response = requests.post(self.ollama_url, json=payload)
            # 获取模型的输出
            summary = response.json().get("response", "Error: No response from model")
        except Exception as e:
            summary = f"Error during summarization: {str(e)}"
        return summary

    def rerank_docs(self, query: str, docs: List[str]) -> List[str]:
        """
        使用 FlagReranker 对检索到的文档进行重新排序
        """
        pairs = [(query, doc) for doc in docs]
        scores = self.reranker.compute_score(pairs)  # 获取每个上下文的得分
        # 按照得分对文档进行排序
        ranked_docs = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)  # 逆序排序
        return [doc for doc, score in ranked_docs]