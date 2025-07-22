import torch
from torch.nn import CrossEntropyLoss
from transformers import GenerationConfig, AutoTokenizer, AutoModelForCausalLM
import os
import requests

# LM 类：语言模型调用统一接口
class LM(object):
    def __init__(self, my_args, llm_args) -> None:
        self.api = my_args.api
        self.is_chat_model = llm_args.is_chat_model
        if llm_args.ollama_ckpt is not None:
            self.model_name = llm_args.ollama_ckpt
        elif llm_args.hf_ckpt:
            self.model_name = llm_args.hf_ckpt.split("/")[-1]
        
        # HuggingFace 本地模型设置（API 选择为 hf）
        if my_args.api == 'hf':
            # 加载 tokenizer 和模型
            self.tokenizer = AutoTokenizer.from_pretrained(llm_args.hf_ckpt)
            self.model = AutoModelForCausalLM.from_pretrained(llm_args.hf_ckpt, device_map='auto').cuda().eval()
            # 调整词表大小（防止 tokenizer 扩展）
            self.model.resize_token_embeddings(len(self.tokenizer))
            # 设置文本生成的超参数配置
            self.generation_config = GenerationConfig(
                max_new_tokens=llm_args.max_new_tokens,
                do_sample=llm_args.do_sample,
                temperature=llm_args.temperature,
                top_p=llm_args.top_p,
                top_k=llm_args.top_k,
                num_beams=llm_args.num_beams,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id,
            )
            self.loss_fn = CrossEntropyLoss(reduction="none")
            
        elif my_args.api == 'ollama':
            assert llm_args.ollama_ckpt is not None
            self.model_name = llm_args.ollama_ckpt
            self.ollama_url = "http://localhost:11434/api/generate"
            try:
                local_tokenizer_path = os.path.join(os.path.dirname(__file__), "Llama-2-7b-chat-hf")
                self.tokenizer = AutoTokenizer.from_pretrained(local_tokenizer_path)
                print(f"\n[INFO] 成功加载本地 tokenizer：{local_tokenizer_path}\n")
            except Exception as e:
                print(f"\n[ERROR] 无法加载 tokenizer，请确认路径正确：{local_tokenizer_path}\n")
                raise e
            self.generation_config = {
                "model_ckpt": llm_args.ollama_ckpt,
                "max_tokens": llm_args.max_new_tokens,
                "temperature": llm_args.temperature,
                "top_k": llm_args.top_k,
                "top_p": llm_args.top_p,
                "stop": llm_args.stop_tokens
            }
            if llm_args.is_chat_model:
                self.generation_config["system_prompt"] = llm_args.system_prompt

        else:
            raise NotImplementedError
        
        # assert self.tokenizer is not None
    
    # 统一生成接口
    def generate(self, lm_input: str, compute_generation_scores=False, compute_input_loss=False):
        """input a string, output a string"""
        
        output_dict = dict()
        # print("🔍 当前请求是否走代理：", requests.utils.get_environ_proxies(self.ollama_url))

        # HuggingFace 本地模型：生成、得分、PPL 分析等
        if self.api == 'hf':
            assert torch.cuda.is_available(), "CUDA is not available???"
            inputs = self.tokenizer(lm_input, return_tensors="pt")
            input_ids = inputs["input_ids"].cuda()  #! [1, *]
            assert input_ids.ndim == 2 and input_ids.shape[0] == 1
            # 模型生成
            with torch.no_grad():
                generation_output = self.model.generate(
                    input_ids=input_ids,
                    generation_config=self.generation_config,
                    return_dict_in_generate=True,
                    output_scores=True,
                )
                output_ids = generation_output.sequences[0]
                generated_tokens = output_ids[input_ids.shape[1]:]  # 截取新生成部分
                
                # 选项1：计算生成 token 的 score 和 prob
                if compute_generation_scores or compute_input_loss:
                    raise NotImplementedError("Ollama 不支持 logits、loss 计算")
                # if compute_generation_scores:
                #     # compute transition scores: https://huggingface.co/docs/transformers/main/en/main_classes/text_generation#transformers.GenerationMixin.compute_transition_scores 
                #     # discussion: https://discuss.huggingface.co/t/announcement-generation-get-probabilities-for-generated-output/30075 
                #     # to get probability from score, just use: exp(score)
                #     transition_scores = self.model.compute_transition_scores(
                #         generation_output.sequences, generation_output.scores, normalize_logits=True
                #     )
                #     assert generated_tokens.shape == transition_scores[0].shape
                #     generation_scores = transition_scores[0]
                #     output_dict["generation_scores"] = generation_scores
                #     output_dict["generation_probs"] = torch.exp(generation_scores)
                
                # # 选项2：对输入 prompt 本身计算 token loss / perplexity
                # if compute_input_loss:
                #     forward_output = self.model(
                #         input_ids=input_ids, labels=input_ids
                #     )
                #     logits = forward_output.logits  #! [1, *, 50257]
                #     logits_shift_left = logits[:, :-1, :]
                #     logits_shift_left = logits_shift_left.reshape(-1, logits_shift_left.size(-1))
                #     labels = input_ids[:, 1:]
                #     labels = labels.reshape(-1)
                #     token_losses = self.loss_fn(logits_shift_left, labels)
                #     output_dict["token_loss_list"] = token_losses.tolist()
                #     output_dict["total_input_loss"] = sum(output_dict["token_loss_list"]) / len(output_dict["token_loss_list"])
                #     # perplexity: https://huggingface.co/docs/transformers/perplexity 
                #     output_dict["token_ppl_list"] = torch.exp(token_losses).tolist()
                #     output_dict["total_input_ppl"] = math.exp(output_dict["total_input_loss"])
            # 解码输出文本
            lm_output = self.tokenizer.decode(generated_tokens)
            output_dict["lm_output"] = lm_output
        elif self.api == 'ollama':
            payload = {
                "model": self.model_name,
                "prompt": lm_input,
                "stream": False
            }
            # response = requests.post(self.ollama_url, json=payload)
            # response = requests.post(self.ollama_url, json=payload, proxies={})
            session = requests.Session()
            session.trust_env = False  # 禁用从环境变量继承代理
            response = session.post(self.ollama_url, json=payload)
            output_dict["lm_output"] = response.json()["response"]
            

        else:
            raise NotImplementedError
        
        return output_dict