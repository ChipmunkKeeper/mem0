import os
from typing import List, Dict
from mem0 import Memory
from datetime import datetime
from openai import OpenAI

# 设置 API KEY
# os.environ["DEEPSEEK_API_KEY"] = "sk-f4911a6f945f4c6282e7df830bfdc600" # deepseek
# BASE_URL = "https://api.deepseek.com"
# MODEL_NAME = "deepseek-chat"

os.environ["DEEPSEEK_API_KEY"] = "sk-eee609e2be48437ca8f7cfe4e61c59b0" # qwen免费deepseek v3.2
BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
MODEL_NAME = "deepseek-v3.2"
    
# embedding用本地
# os.environ["OPENAI_API_KEY"] = "sk-xxx" 

user_id = "customer_bot_user_1"

# 根目录
ROOT_MEMORY_PATH = "/home/jwuev/code/mem0/local_memory"
# 向量数据库文件夹，相对路径即可
VECTOR_DB_PATH = "vector_db"
# 历史记录文件 (SQLite 需要一个文件路径)
HISTORY_DB_PATH = os.path.join(ROOT_MEMORY_PATH, "history.db")
# 不同数据库的子路径，用于隔离记忆
MEMORY_COLLECTION_PATH = os.path.join(ROOT_MEMORY_PATH, user_id)

class SupportChatbot:
    def __init__(self):
        # 配置 Mem0 使用 DeepSeek (LLM) 和 BGE-M3 (Local Embedding)
        self.config = {
            # 1. 配置 LLM (用于记忆提取和更新)
            "llm": {
                "provider": "openai",  # 使用 openai 兼容 DeepSeek
                "config": {
                    "model": MODEL_NAME,
                    "openai_base_url": BASE_URL, 
                    "temperature": 0.1,
                    "max_tokens": 2000,
                    "api_key": os.environ["DEEPSEEK_API_KEY"],
                },
            },
            # 2. 配置 Embedder (使用本地 HuggingFace 模型)
            "embedder": {
                "provider": "huggingface",
                "config": {
                    "model": "BAAI/bge-m3",
                    "model_kwargs": {"device": "cuda"} 
                },
            },
            # 3. 配置 Vector Store (适配 Embedding 维度)
            "vector_store": {
                # "provider": "qdrant",  # 默认也是 qdrant
                "config": {
                    "collection_name": MEMORY_COLLECTION_PATH, # 本地记忆的子分类，不同记忆库
                    "path": VECTOR_DB_PATH, # 本地持久化路径
                    "embedding_model_dims": 1024,  # BGE-M3 的维度是 1024
                },
            },
            "history_db_path": HISTORY_DB_PATH # 每次对记忆操作的历史记录，当前路径为不同记忆库之间共享
        }

        # 初始化 Mem0
        self.memory = Memory.from_config(self.config)

        # 初始化用于对话生成的 Client
        self.client = OpenAI(
            api_key=os.environ["DEEPSEEK_API_KEY"], 
            base_url=BASE_URL
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
        """Store customer interaction in memory."""
        if metadata is None:
            metadata = {}

        metadata["timestamp"] = datetime.now().isoformat()
        conversation = [{"role": "user", "content": message}, {"role": "assistant", "content": response}]
        
        # 存储到 Mem0
        self.memory.add(conversation, user_id=user_id, metadata=metadata)

    def get_relevant_history(self, user_id: str, query: str) -> List[Dict]:
        """Retrieve relevant past interactions."""
        # mem0 的 search 也会自动使用 config 中定义的 BGE-M3 进行 embedding
        return self.memory.search(
            query=query,
            user_id=user_id,
            limit=5,
        )

    def handle_customer_query(self, user_id: str, query: str) -> str:
        # 获取相关历史
        relevant_history = self.get_relevant_history(user_id, query)

        # 构建上下文
        context = "Previous relevant interactions:\n"
        # 注意：根据 mem0 版本，返回结构可能略有不同，通常是 result['memory']
        for item in relevant_history.get("results", []): # 安全获取 results
             context += f"Memory: {item.get('memory', '')}\n"
        context += "---\n"

        prompt = f"""
        {self.system_context}

        {context}

        Current customer query: {query}

        Provide a helpful response that takes into account any relevant past interactions.
        """

        response = self.client.chat.completions.create(
            model=MODEL_NAME,
            extra_body={"enable_thinking": False}, # deepseek-v3.2可以深度思考
            messages=[{"role": "user", "content": prompt}],
            max_tokens=2000,
            temperature=0.1,
        )

        response_text = response.choices[0].message.content

        # 存储交互
        self.store_customer_interaction(
            user_id=user_id, message=query, response=response_text, metadata={"type": "support_query"}
        )

        return response_text



if __name__ == "__main__":
    chatbot = SupportChatbot()
    print("Welcome to Customer Support! Type 'exit' to end the conversation.")

    while True:
        try:
            query = input("Customer: ")
            if query.lower() == "exit":
                print("Thank you for using our support service. Goodbye!")
                
                # 在 Python 关闭前，手动关闭 Qdrant 客户端
                # 这会触发 unlock，此时 Python 的 import 功能还正常，不会报错
                try:
                    # Mem0 将 vector_store 封装在 memory 对象里
                    # 我们需要通过层级找到底层的 client 并关闭
                    if hasattr(chatbot.memory, "vector_store"):
                        # Qdrant client 通常挂载在 vector_store.client 上
                        if hasattr(chatbot.memory.vector_store, "client"):
                            print("正在安全关闭数据库连接...")
                            chatbot.memory.vector_store.client.close()
                except Exception as e:
                    print(f"关闭连接时发生非致命错误: {e}")
                # ===========================================
                
                break
            
            response = chatbot.handle_customer_query(user_id, query)
            print("Support:", response, "\n")
        except KeyboardInterrupt:
            # 如果是按 Ctrl+C 退出，也尝试关闭
            try:
                if hasattr(chatbot.memory.vector_store, "client"):
                    chatbot.memory.vector_store.client.close()
            except:
                pass
            break