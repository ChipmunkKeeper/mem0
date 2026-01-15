import os
from typing import List, Dict
from mem0 import Memory
from datetime import datetime
from openai import OpenAI
import re


"""
用vllm形成本地服务器，再通过openai api接口调用

开始服务：
CUDA_VISIBLE_DEVICES=8 python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-7B-Instruct \
    --port 8000 \
    --gpu-memory-utilization 0.9

结束进程： kill -9 <PID>

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
        - If you're unsure about something, ask for clarification
        - Keep track of open issues and follow-ups
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
            extra_body={"enable_thinking": False},
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
    

    def import_chat_history(self, file_path: str, user_id: str, batch_size=20):
        """
        读取微信格式的txt文件，按批次导入到mem0记忆中
        """
        print(f"Start importing history from {file_path}...")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        start_index = 0
        for i, line in enumerate(lines):
            if "--------" in line:
                start_index = i + 1
                break
        
        pattern = re.compile(r'^(\d{4}-\d{2}-\d{2})\s+(\d{2}:\d{2}:\d{2})\s+(\S+)\s+(\S+)\s+(.*)$')
        
        batch_buffer = [] 
        current_batch_metadata = {} 
        
        from tqdm import tqdm
        progress_bar = tqdm(total=len(lines) - start_index)
        
        count = 0

        for line in lines[start_index:]:
            line = line.strip()
            if not line:
                continue

            match = pattern.match(line)
            if match:
                date_str, time_str, name, status, content = match.groups()
                full_time = f"{date_str}T{time_str}"
                
                # 处理特殊内容
                if content == "[动画表情]":
                    content = "[Sent an animated sticker]"
                
                # 构造单条文本，这里稍微改一下格式，让 LLM 更好理解这是一段连续对话
                # 例如： "2025-06-06 12:44:00 - User: 你好"
                role_label = "User (Me)" if name == "我" else f"Contact ({name})"
                formatted_line = f"[{full_time}] {role_label}: {content}"
                
                # 如果缓冲区是空的，记录这一批次的开始时间作为元数据
                if not batch_buffer:
                    current_batch_metadata = {
                        "timestamp": full_time,
                        "source": "wechat_import",
                        "import_date": datetime.now().isoformat()
                    }

                batch_buffer.append(formatted_line)
                
                # --- 触发批量写入条件 ---
                if len(batch_buffer) >= batch_size:
                    self._flush_buffer(batch_buffer, user_id, current_batch_metadata)
                    count += 1 # 记录批次数量
                    batch_buffer = [] # 清空
                    current_batch_metadata = {}

            progress_bar.update(1)

        # 循环结束后，处理剩余未满一批的数据
        if batch_buffer:
            self._flush_buffer(batch_buffer, user_id, current_batch_metadata)
            count += 1
            
        progress_bar.close()
        print(f"Successfully imported {count} batches of history.")

    def _flush_buffer(self, buffer: List[str], user_id: str, metadata: Dict):
        """辅助函数：将缓冲区的内容合并并写入 Memory"""
        if not buffer:
            return
            
        # 将列表合并成一个大的文本块
        # mem0 会自动分析这个文本块中的所有信息
        combined_text = "\n".join(buffer)
        
        # 调用一次 add，处理多条对话
        self.memory.add(
            combined_text, 
            user_id=user_id, 
            metadata=metadata
        )


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