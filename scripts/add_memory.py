import os
from typing import List, Dict
from mem0 import Memory
from datetime import datetime
from openai import OpenAI
import re


"""
用vllm形成本地服务器，再通过openai api接口调用

开始服务：
CUDA_VISIBLE_DEVICES=1 python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-7B-Instruct \
    --port 8000 \
    --gpu-memory-utilization 0.9

结束进程： kill -9 1946433

"""

BASE_URL = "http://localhost:8000/v1"
os.environ["LOCAL_API_KEY"] = "EMPTY"
MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"

# user_id = "customer_bot_user_1"
user_id = "wechat_wy" 

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
                    "api_key": os.environ["LOCAL_API_KEY"],
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
            api_key=os.environ["LOCAL_API_KEY"], 
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
    
    def import_chat_history(self, file_path: str, user_id: str):
        """
        读取微信格式的txt文件并导入到mem0记忆中
        """
        print(f"Start importing history from {file_path}...")
        
        count = 0
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        start_index = 0
        for i, line in enumerate(lines):
            if "--------" in line:
                start_index = i + 1
                break
        
        # 预编译正则，处理变长空格
        # 格式示例: 2025-06-06 12:44:00      我                   发送            不是昨晚就是44张吗，现在还是
        # 逻辑：日期(组1) 时间(组2)  任意空格  姓名(组3)  任意空格  状态(组4)  任意空格  内容(组5)
        pattern = re.compile(r'^(\d{4}-\d{2}-\d{2})\s+(\d{2}:\d{2}:\d{2})\s+(\S+)\s+(\S+)\s+(.*)$')

        messages_to_add = []
        from tqdm import tqdm
        progress_bar = tqdm(total=len(lines) - 1)

        for line in lines[start_index:]:
            line = line.strip()
            if not line:
                continue

            match = pattern.match(line)
            if match:
                date_str, time_str, name, status, content = match.groups()
                full_time = f"{date_str}T{time_str}"
                
                # 1. 转换角色语义
                # 假设 "我" 是 User, "吴悠" 是对话的另一方 (可以视为 Assistant 或 Third Party)
                # 为了让 Memory 更好理解，我们将非结构化文本转换为叙述句
                
                if name == "我":
                    # 将“我”的行为描述为 User 的行为
                    narrative = f"User (Me) sent a message: '{content}'"
                    role = "user"
                else:
                    # 将“其他人”的行为描述为 Context
                    narrative = f"Contact Person ({name}) sent a message: '{content}'"
                    role = "assistant" # 或者 user，取决于你想怎么存，但在 mem0 中 add 纯文本更灵活

                # 2. 构造 Memory Item
                # 技巧：我们不使用 .add(messages=[...]) 的对话模式，而是使用 .add(text, metadata)
                # 原因：导入的是历史事实，用叙述性文本更容易被提取成“记忆点”
                
                # 如果内容是表情，做个标记，防止 LLM 困惑
                if content == "[动画表情]":
                    content = "[Sent an animated sticker]"
                    narrative = f"{name} sent an animated sticker."

                memory_text = f"Interaction on {date_str} at {time_str}: {name} ({status}) said: {content}"
                
                # 3. 添加到 mem0
                self.memory.add(
                    memory_text, 
                    user_id=user_id, 
                    metadata={
                        "timestamp": full_time, 
                        "source": "wechat_import", 
                        "original_speaker": name,
                        "import_date": datetime.now().isoformat()
                    }
                )
                # print(f"Imported: {memory_text[:50]}...")
                count += 1
                progress_bar.update(1)
                
        progress_bar.close()
        print(f"Successfully imported {count} memory items.")

if __name__ == "__main__":
    chatbot = SupportChatbot()
    
    import_file_path = "/home/jwuev/code/mem0/local_memory/raw_data/chat_his.txt" 
    
    # 检查是否需要导入 (你可以做个判断，或者手动开启)
    if os.path.exists(import_file_path):
        print("发现聊天记录文件，是否导入? (y/n)")
        choice = input().strip().lower()
        if choice == 'y':
            chatbot.import_chat_history(import_file_path, user_id)
            # 导入完为了防止重复导入，你可以重命名文件，或者依赖 vector db 的去重机制(mem0 有一定去重但不完美)
            # os.rename(import_file_path, import_file_path + ".bak")

    print("\nSystem ready. Type 'exit' to quit.\n")
    
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