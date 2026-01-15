import os
from mem0 import Memory


BASE_URL = "http://localhost:8000/v1"
os.environ["LOCAL_API_KEY"] = "EMPTY"
MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
user_id = "wechat_wy" 


ROOT_MEMORY_PATH = "/home/jwuev/code/mem0/local_memory"
VECTOR_DB_PATH = "vector_db"
HISTORY_DB_PATH = os.path.join(ROOT_MEMORY_PATH, "history.db")
MEMORY_COLLECTION_PATH = os.path.join(ROOT_MEMORY_PATH, user_id)

class MemoryRetriever:
    def __init__(self):
        self.config = {
            "llm": {
                "provider": "openai",
                "config": {
                    "model": MODEL_NAME,
                    "openai_base_url": BASE_URL,
                    "max_tokens": 2000,
                    "api_key": os.environ["LOCAL_API_KEY"],
                },
            },
            "embedder": {
                "provider": "huggingface",
                "config": {
                    "model": "BAAI/bge-m3",
                    "model_kwargs": {"device": "cuda"} 
                },
            },
            "vector_store": {
                "config": {
                    "collection_name": MEMORY_COLLECTION_PATH,
                    "path": VECTOR_DB_PATH,
                    "embedding_model_dims": 1024,
                },
            },
            "history_db_path": HISTORY_DB_PATH
        }

        print("正在初始化 Mem0 记忆库...")
        self.memory = Memory.from_config(self.config)
        print("初始化完成。")

    def search_memory(self, query: str, limit: int = 20):
        """
        仅检索，不生成对话
        """
        print(f"\n正在检索关于: '{query}' 的记忆...")
        
        results = self.memory.search(
            query=query,
            user_id=user_id,
            limit=limit
        )
        
        return results

if __name__ == "__main__":
    retriever = MemoryRetriever()
    
    print("\n=== 记忆检索模式 (输入 'exit' 退出) ===")
    
    while True:
        try:
            query = input("\n请输入检索关键词: ")
            if query.lower() == "exit":
                print("退出程序。")
                # 安全关闭连接
                try:
                    if hasattr(retriever.memory, "vector_store") and hasattr(retriever.memory.vector_store, "client"):
                        retriever.memory.vector_store.client.close()
                except Exception:
                    pass
                break
            
            # 执行检索
            # 注意：mem0 的 search 返回结构通常包含 'results' 列表
            search_results = retriever.search_memory(query)
            
            # 解析并打印结果
            # 不同的 mem0 版本返回结构略有不同，这里做通用处理
            items = search_results.get("results", []) if isinstance(search_results, dict) else search_results
            
            if not items:
                print("未找到相关记忆。")
            else:
                print(f"找到 {len(items)} 条相关记忆:\n")
                for i, item in enumerate(items):
                    # 提取记忆文本，通常在 'memory' 字段中
                    mem_content = item.get('memory', 'No content')
                    # 提取相关性分数 (如果有)
                    score = item.get('score', 'N/A')
                    # 提取时间戳 (如果有)
                    timestamp = item.get('metadata', {}).get('timestamp', 'Unknown time')
                    
                    print(f"[{i+1}] (Score: {score}) - {timestamp}")
                    print(f"    内容: {mem_content}")
                    print("-" * 50)

        except KeyboardInterrupt:
            print("\n强制退出。")
            break
        except Exception as e:
            print(f"发生错误: {e}")