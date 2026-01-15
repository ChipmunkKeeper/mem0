import os
import torch
from typing import List, Dict, Optional
from datetime import datetime
from transformers import AutoModelForCausalLM, AutoTokenizer

# 引入 mem0 的组件
from mem0 import Memory
from mem0.llms.base import LLMBase
from mem0.embeddings.huggingface import HuggingFaceEmbedding
from mem0.vector_stores.qdrant import Qdrant
from mem0.configs.embeddings.base import BaseEmbedderConfig

# ================= 配置区域 =================
# 显卡设置：如果有特定显卡需求，设置 visible devices
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"
USER_ID = "customer_bot_user_1"

# 路径配置
ROOT_MEMORY_PATH = "/home/jwuev/code/mem0/local_memory"
VECTOR_DB_PATH = "vector_db"
HISTORY_DB_PATH = os.path.join(ROOT_MEMORY_PATH, "history.db")
MEMORY_COLLECTION_PATH = os.path.join(ROOT_MEMORY_PATH, USER_ID)
# ===========================================

class LocalHuggingFaceLLM(LLMBase):
    """
    自定义的本地 LLM 类，直接使用 transformers 库运行，不依赖 API 服务。
    """
    def __init__(self, config=None):
        super().__init__(config)
        print(f"正在加载本地模型: {MODEL_ID} ...")
        
        # 加载分词器
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
        
        # 加载模型
        # device_map="auto" 会自动利用 GPU
        # torch_dtype="auto" 会自动使用 float16 或 bfloat16 (如果显卡支持)
        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            device_map="auto", 
            dtype="auto",
            trust_remote_code=True
        )
        print("本地模型加载完成。")

    def generate_response(self, messages: List[Dict[str, str]], response_format=None, tools=None, tool_choice="auto"):
        """
        实现 mem0 需要的生成接口
        """
        # 1. 使用 chat template 处理对话历史
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        # 2. 编码输入
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
        
        # 3. 推理生成
        # max_new_tokens 控制回答长度，可以根据显存调整
        with torch.no_grad():
            generated_ids = self.model.generate(
                **model_inputs,
                max_new_tokens=4096,
                temperature=0.3,
                top_p=0.9
            )
        
        # 4. 解码输出 (去除输入部分的 token)
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        
        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return response

class SupportChatbot:
    def __init__(self):
        # 1. 定义配置字典 (根据你之前的报错信息推测的配置)
        config = {
            "history_db_path": HISTORY_DB_PATH, # 每次对记忆操作的历史记录，当前路径为不同记忆库之间共享
            "vector_store": {
                "provider": "qdrant",
                "config": {
                    "collection_name": MEMORY_COLLECTION_PATH,
                    "path": VECTOR_DB_PATH,  
                    "embedding_model_dims": 1024, 
                }
            },
            "embedder": {
                "provider": "huggingface",
                "config": {
                    "model": "BAAI/bge-m3",
                    "model_kwargs": {"device": "cuda"}   
                }
            },
            "llm": {
                "provider": "huggingface",
                "config": {
                    "model": "Qwen/Qwen2.5-3B-Instruct", # 或者你的本地模型名称
                    "temperature": 0.1,
                }
            }
        }
        self.memory = Memory.from_config(config)

        self.llm = LocalHuggingFaceLLM()

        self.embedder = HuggingFaceEmbedding(
            config=BaseEmbedderConfig(
                model="BAAI/bge-m3", 
                huggingface_base_url=None,    
                model_kwargs={"device": "cuda"},  
            )
        )

        # 3. 初始化 Vector Store (Qdrant)
        self.vector_store = Qdrant(
            collection_name=MEMORY_COLLECTION_PATH,  
            embedding_model_dims=1024,      
            path=VECTOR_DB_PATH,         
            on_disk=True                    # 选填：开启持久化存储
        )

        # 定义系统上下文
        self.system_context = """
        You are a helpful customer support agent. Use the following guidelines:
        - Be polite and professional
        - Show empathy for customer issues
        - Reference past interactions when relevant
        - Maintain consistent information across conversations
        """

    def store_customer_interaction(self, user_id: str, message: str, response: str, metadata: Dict = None):
        if metadata is None:
            metadata = {}
        metadata["timestamp"] = datetime.now().isoformat()
        conversation = [{"role": "user", "content": message}, {"role": "assistant", "content": response}]
        self.memory.add(conversation, user_id=user_id, metadata=metadata)

    def handle_customer_query(self, user_id: str, query: str) -> str:
        # 获取相关历史
        relevant_history = self.memory.search(query=query, user_id=user_id, limit=5)

        # 构建上下文
        context_str = "Previous relevant interactions:\n"
        for item in relevant_history.get("results", []):
            context_str += f"Memory: {item.get('memory', '')}\n"
        context_str += "---\n"

        # 构建给模型的完整 Prompt
        # 注意：因为是本地调用 transformers，我们需要构建标准的 messages 列表
        messages = [
            {"role": "system", "content": self.system_context + "\n" + context_str},
            {"role": "user", "content": query}
        ]

        print("正在思考 (Local GPU)...")
        # 直接调用我们自定义 LLM 类的方法生成回复
        # 这样既复用了模型实例，又避免了 OpenAI Client
        response_text = self.llm.generate_response(messages)

        # 存储交互
        self.store_customer_interaction(
            user_id=user_id, message=query, response=response_text, metadata={"type": "support_query"}
        )

        return response_text

if __name__ == "__main__":
    # 第一次运行会下载模型，需要一点时间
    chatbot = SupportChatbot()
    print(f"Loaded Local Model: {MODEL_ID}")
    print("Welcome to Customer Support (Local Version)! Type 'exit' to end.")

    while True:
        try:
            query = input("Customer: ")
            if query.lower() == "exit":
                print("Goodbye!")
                # 清理显存（可选）
                if hasattr(chatbot, "llm"):
                    del chatbot.llm
                    torch.cuda.empty_cache()
                break
            
            response = chatbot.handle_customer_query(USER_ID, query)
            print("Support:", response, "\n")
            
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error: {e}")