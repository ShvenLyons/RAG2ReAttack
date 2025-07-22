import sys
sys.path.append(".")

import os
os.environ['TOKENIZERS_PARALLELISM'] = 'False'

import json
import subprocess
import numpy as np
from tqdm import tqdm
from typing import List
from pyserini.search.lucene import LuceneSearcher

from utils.helpers import read_raw_data_dir

import faiss
from sentence_transformers import SentenceTransformer

'''
文本处理：将原始语料分词、切块
构建索引：调用 Pyserini 的 Lucene 引擎构建 JSON 文档索引
检索接口：支持基于 BM25 的相关文档检索
应用场景：适用于 RAG、增强语言模型生成、轻量向量检索等任务
'''

# 基类 Index
class Index(object):
    def __init__(self, raw_data_dir, datastore_dir) -> None:
        # assert os.path.exists(raw_data_dir)
        self.raw_data_dir = raw_data_dir
        self.datastore_dir = datastore_dir
    
    def find_most_relevant_k_documents(query: str, k: int):
        raise NotImplementedError  # 抽象方法：子类需实现具体检索逻辑

# BM25Index 类 — Pyserini 实现检索索引构建与查询
class BM25Index(Index):
    def __init__(self, tokenizer, max_retrieval_seq_length: int, stride: int,
                 raw_data_dir, datastore_dir, recursive=True) -> None:
        super().__init__(raw_data_dir, datastore_dir)
        
        self.tokenizer = tokenizer
        self.max_retrieval_seq_length = max_retrieval_seq_length
        self.stride = stride
        
        # 如果索引目录不存在，则重新构建索引
        if (not os.path.exists(datastore_dir)) or (len(os.listdir(datastore_dir)) == 0):
            os.makedirs(datastore_dir, exist_ok=True)
            
            # Step 1：读取原始数据并分词
            print("==> Reading and tokenizing raw data...")
            data = read_raw_data_dir(raw_data_dir=raw_data_dir, recursive=recursive)
            # todo: process very long text?
            all_text = " ".join(data)
            all_words = all_text.split()

            # 每 step_size 个词一组，构建 token 块
            step_size = 1024
            chunks_to_tokenize = [all_words[i:i + step_size] for i in range(0, len(all_words), step_size)]
            chunks_to_tokenize = [" ".join(chunk) for chunk in chunks_to_tokenize]
            
            final_tokens = []
            for chunk in tqdm(chunks_to_tokenize):
                tokenizer.parallelism = 8  # 多线程加速
                tokenized_data = tokenizer(chunk)['input_ids']
                final_tokens.extend(tokenized_data)
            final_tokens = np.array(final_tokens)
            print(f"==> Number of tokens: {len(final_tokens)}.")
            
            # Step 2：将 tokens 切分为固定长度的块（Chunk）
            print("==> Making chunks...")
            tokens_as_chunks = self._get_token_chunks(
                final_tokens, 
                pad_token=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
            )
            print(f"==> {len(tokens_as_chunks)} chunks in total.")

            # Step 3：将 token 块转换为 JSONL 格式，供 Pyserini 建索引使用
            self.tokens_dir = os.path.join(datastore_dir, "tokens")
            os.makedirs(self.tokens_dir, exist_ok=True)
            with open(os.path.join(self.tokens_dir, "data.jsonl"), "w") as f:
                for chunk_id, token_chunk in enumerate(tokens_as_chunks):
                    assert len(token_chunk) <= max_retrieval_seq_length
                    text = tokenizer.decode(token_chunk)
                    f.write(json.dumps({
                        "id": str(chunk_id),
                        "contents": text,
                        "input_ids": token_chunk.tolist()
                    })+"\n")
        
            # Step 4：调用 Pyserini 命令行接口构建 Lucene 索引
            print("==> Start building index for %s at %s" % (self.tokens_dir, datastore_dir))
            command = """python -m pyserini.index.lucene \
            --collection JsonCollection \
            --input '%s' \
            --index '%s' \
            --generator DefaultLuceneDocumentGenerator \
            --storeRaw --threads 1""" % (self.tokens_dir, datastore_dir)
            ret_code = subprocess.run([command],
                                      shell=True,
                                      # stdout=subprocess.DEVNULL,
                                      # stderr=subprocess.STDOUT
                                      )
            if ret_code.returncode != 0:
                print("Failed to build the index")
                exit()
            else:
                print("Successfully built the index")
        # 如果索引已经存在
        else:
            print("==> Datastore exists at: ", datastore_dir)
        
        self.searcher = LuceneSearcher(datastore_dir)
    
    # 内部方法：将长 token 流按 stride 滑窗方式切成固定长度
    def _get_token_chunks(self, tokens: np.ndarray, pad_token: int) -> np.ndarray:
        assert tokens.ndim == 1, "Tokens should be flattened first!"
        num_tokens = len(tokens)
        tokens_as_chunks = []
        
        for begin_loc in range(0, num_tokens, self.stride):
            end_loc = min(begin_loc + self.max_retrieval_seq_length, num_tokens)
            token_chunk = tokens[begin_loc:end_loc].copy()
        
            if end_loc == num_tokens and len(token_chunk) < self.max_retrieval_seq_length:
                pads = np.array([pad_token for _ in range(self.max_retrieval_seq_length - len(token_chunk))])
                token_chunk = np.concatenate([token_chunk, pads])
        
            assert len(token_chunk) == self.max_retrieval_seq_length
            
            tokens_as_chunks.append(token_chunk)
        
        tokens_as_chunks = np.stack(tokens_as_chunks)
        return tokens_as_chunks
    
    # 查询接口：返回与 query 最相关的前 k 条文档
    def find_most_relevant_k_documents(self, query: str, k: int) -> List[str]:
        hits = self.searcher.search(query, k=k)
        docs = []
        for hit in hits:
            docid = hit.docid
            raw = self.searcher.doc(docid).raw()
            input_ids = json.loads(raw)["input_ids"]
            doc_str = self.tokenizer.decode(input_ids)
            docs.append(doc_str)
        return docs
    


class FAISSIndex(Index):
    def __init__(self, embedder: SentenceTransformer, max_retrieval_seq_length, stride,
                 raw_data_dir, datastore_dir, recursive=True) -> None:
        super().__init__(raw_data_dir, datastore_dir)

        self.index_path = os.path.join(datastore_dir, "faiss.index")
        self.docs_path = os.path.join(datastore_dir, "docs.json")
        self.embedder = embedder
        self.max_retrieval_seq_length = max_retrieval_seq_length
        self.stride = stride

        if not os.path.exists(self.index_path) or not os.path.exists(self.docs_path):
            print("==> Building FAISS index from scratch...")
            self._build_index(recursive)
        else:
            print("==> Loading FAISS index from disk...")
            self._load_index()

    def _get_text_chunks(self, text: str) -> List[str]:
        words = text.split()
        chunks = []
        for i in range(0, len(words), self.stride):
            chunk = words[i:i + self.max_retrieval_seq_length]
            if len(chunk) < 5:
                continue
            chunks.append(" ".join(chunk))
        return chunks

    def _build_index(self, recursive=True):
        raw_docs = read_raw_data_dir(self.raw_data_dir, recursive=recursive)
        self.docs: List[str] = []
        for doc in raw_docs:
            self.docs.extend(self._get_text_chunks(doc))

        self.embeddings: np.ndarray = self.embedder.encode(
            self.docs,
            batch_size=32,
            show_progress_bar=True,
            convert_to_numpy=True
        )

        dim = self.embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dim)
        self.index.add(self.embeddings)

        os.makedirs(self.datastore_dir, exist_ok=True)
        faiss.write_index(self.index, self.index_path)
        with open(self.docs_path, 'w', encoding='utf-8') as f:
            json.dump(self.docs, f, ensure_ascii=False)
        print(f"==> FAISS index and documents saved to {self.datastore_dir}.")

    def _load_index(self):
        self.index = faiss.read_index(self.index_path)
        with open(self.docs_path, 'r', encoding='utf-8') as f:
            self.docs = json.load(f)

    def find_most_relevant_k_documents(self, query: str, k: int) -> List[str]:
        query_embedding = self.embedder.encode([query], convert_to_numpy=True)
        scores, indices = self.index.search(query_embedding, k)
        return [self.docs[i] for i in indices[0]]


if __name__ == '__main__':
    pass