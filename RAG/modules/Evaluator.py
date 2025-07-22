import nltk
from evaluate import load
import numpy as np
import scipy
from typing import Dict, List, Union
from collections import Counter
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import os
import json

# 加载预定义的三个评估指标：ROUGE、BLEU、BERTScore
rouge = load("rouge")
sacrebleu = load("sacrebleu")
bertscore = load("bertscore")

class Evaluator:
    """
    用于评估模型生成文本的指标类。支持：
    - token级别 precision/recall/F1
    - n-gram 重合度（1-3）
    - BLEU、ROUGE、BERTScore
    - 精确匹配率（Exact Match）
    - 重构率（Reconstruction Rate）
    """
    def __init__(self, predictions_str: List[str], references_str: List[str], raw_querys: List[str]) -> None:
        self.metric_rouge = rouge
        self.metric_bleu = sacrebleu
        self.metric_bertscore = bertscore
        self.predictions_str = predictions_str
        self.references_str = references_str
        self.raw_querys = raw_querys

    # 当前只计算文本比较类的评估指标
    def compute_metrics(self):
        return self._text_comparison_metrics()

    # 核心评估逻辑，返回所有聚合后的评估指标。
    def _text_comparison_metrics(self) -> Dict[str, float]:
        # modify code from: https://github.com/jxmorris12/vec2text/blob/master/vec2text/trainers/base.py

        model = SentenceTransformer('all-MiniLM-L6-v2')

        # 平均值和标准误差计算函数
        def mean(L: Union[List[int], List[float]]) -> float:
            assert len(L) > 0
            return sum(L) / len(L)

        def sem(L: List[float]) -> float:
            assert len(L) > 0
            result = scipy.stats.sem(np.array(L))
            if isinstance(result, np.ndarray):
                return result.mean().item()
            return result

        # n-gram 重叠计数器
        def count_overlapping_ngrams(s1: str, s2: str, n: int) -> int:
            ngrams_1 = nltk.ngrams(s1, n)
            ngrams_2 = nltk.ngrams(s2, n)
            ngram_counts_1 = Counter(ngrams_1)
            ngram_counts_2 = Counter(ngrams_2)
            total = 0
            for ngram, count in ngram_counts_1.items():
                total += min(count, ngram_counts_2[ngram])
            return total

        def calculate_reconstruction_rate(retrieved_texts, generated_text):
            """
            计算重构率，严格按照原文中的公式：
            R = (去重后的提取文本块的长度总和) / (原始文本的总长度)
            """
            # 原始文本的总长度 |O|
            original_length = sum(len(text.split()) for text in retrieved_texts)
            # 将每个文本块拆分为单词，放入集合中去重
            retrieved_words_set = set()
            for text in retrieved_texts:
                retrieved_words_set.update(text.split())  # 去重后的文本
            # 生成的文本中的单词集合
            generated_words_set = set(generated_text.split())
            # 计算去重后的提取文本块的长度 |C'|
            reconstruction_length = len(retrieved_words_set & generated_words_set)  # 重叠部分的单词数量
            # 计算重构率 R
            reconstruction_rate = reconstruction_length / original_length if original_length > 0 else 0
            return reconstruction_rate
        
        def calculate_semantic_similarity(question: str, answer: str) -> float:
            """
            计算问题和答案之间的语义相似度
            """
            # 获取问题和答案的嵌入表示
            embeddings_question = model.encode([question])
            embeddings_answer = model.encode([answer])
            # 计算余弦相似度
            similarity = cosine_similarity(embeddings_question, embeddings_answer)[0][0]
            return float(similarity)


        # 保证输入数量一致
        assert len(self.predictions_str) == len(self.references_str)
        num_preds = len(self.predictions_str)
        if not num_preds:
            return {}

        ###########################################################
        # 初始化中间变量
        # Compute token, precision, recall, and ngram-level metrics.
        precision_sum = 0.0
        recall_sum = 0.0
        num_overlapping_words = []
        num_overlapping_bigrams = []
        num_overlapping_trigrams = []
        num_true_words = []
        num_pred_words = []
        f1s = []

        # 遍历所有预测项，计算 token-level F1 和 n-gram 重合度
        for i in range(num_preds):  # for each prediction
            true_words = nltk.tokenize.word_tokenize(self.references_str[i])
            pred_words = nltk.tokenize.word_tokenize(self.predictions_str[i])
            num_true_words.append(len(true_words))
            num_pred_words.append(len(pred_words))
            true_words_set = set(true_words)
            pred_words_set = set(pred_words)

            # 简化的 TP/FP/FN 计算（基于词集合交集）
            TP = len(true_words_set & pred_words_set)
            FP = len(true_words_set) - len(true_words_set & pred_words_set)
            FN = len(pred_words_set) - len(true_words_set & pred_words_set)

            precision = (TP) / (TP + FP + 1e-20)
            recall = (TP) / (TP + FN + 1e-20)
            try:
                f1 = (2 * precision * recall) / (precision + recall + 1e-20)
            except ZeroDivisionError:
                f1 = 0.0

            # 累加所有指标用于最终平均
            precision_sum += precision
            recall_sum += recall
            f1s.append(f1)

            ############################################################
            # 计算 1/2/3-gram 重合数
            num_overlapping_words.append(
                count_overlapping_ngrams(true_words, pred_words, 1)
            )
            num_overlapping_bigrams.append(
                count_overlapping_ngrams(true_words, pred_words, 2)
            )
            num_overlapping_trigrams.append(
                count_overlapping_ngrams(true_words, pred_words, 3)
            )

        # 聚合 token 层面的评估指标
        set_token_metrics = {
            "token_set_precision": (precision_sum / num_preds),
            "token_set_recall": (recall_sum / num_preds),
            "token_set_f1": mean(f1s),
            "token_set_f1_sem": sem(f1s),
            "n_ngrams_match_1": mean(num_overlapping_words),
            "n_ngrams_match_2": mean(num_overlapping_bigrams),
            "n_ngrams_match_3": mean(num_overlapping_trigrams),
            "num_true_words": mean(num_true_words),
            "num_pred_words": mean(num_pred_words),
        }

        ############################################################
        # 调用 evaluate 库计算 BLEU
        bleu_results = np.array(
            [
                self.metric_bleu.compute(predictions=[p], references=[r])["score"]
                for p, r in zip(self.predictions_str, self.references_str)
            ]
        )

        # ROUGE 和 BERTScore 可一次性处理所有数据
        rouge_results = self.metric_rouge.compute(
            predictions=self.predictions_str, references=self.references_str, use_aggregator=False
        )
        bertscore_results = self.metric_bertscore.compute(
            predictions=self.predictions_str, references=self.references_str, lang="en"
        )

        # 精确匹配率（预测完全等于参考）
        exact_matches = np.array(self.predictions_str) == np.array(self.references_str)
        
        # 计算重构率
        total_reconstruction_rate = 0
        num_files = len(self.predictions_str)
        for i in range(num_files):
            # 计算每个文件的重构率
            reconstruction_rate = calculate_reconstruction_rate([self.references_str[i]], self.predictions_str[i])
            total_reconstruction_rate += reconstruction_rate
        average_reconstruction_rate = total_reconstruction_rate / num_files

        # 计算检索质量（语义相似度）
        total_similarity = 0
        max_similarity = -1
        min_similarity = 1
        for i in range(num_files):
            # 计算每个文件的相似度
            similarity = calculate_semantic_similarity(self.predictions_str[i], self.raw_querys[i])
            total_similarity += similarity
            # 更新最大和最小相似度
            if similarity > max_similarity:
                max_similarity = similarity
            if similarity < min_similarity:
                min_similarity = similarity
        # 计算并打印平均相似度
        average_similarity = total_similarity / num_files
        
        # 聚合生成质量评估指标
        gen_metrics = {
            "bleu_score": mean(bleu_results),
            "bleu_score_sem": sem(bleu_results),
            "rougeL_score": mean(rouge_results["rougeL"]),  
            "rougeL_score_sem": sem(rouge_results["rougeL"]),  
            "bert_score": mean(bertscore_results["f1"]),
            "bert_score_sem": sem(bertscore_results["f1"]),
            "exact_match": mean(exact_matches),
            "exact_match_sem": sem(exact_matches),
            "reconstruction_rate": average_reconstruction_rate,
            "in_out_similarity": average_similarity,
            "max_similarity": max_similarity,
            "min_similarity": min_similarity
        }

        # 返回所有指标结果
        all_metrics = {**set_token_metrics, **gen_metrics}
        return all_metrics