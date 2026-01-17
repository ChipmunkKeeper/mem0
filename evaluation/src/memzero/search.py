# import json
# import os
# import time
# from collections import defaultdict
# from concurrent.futures import ThreadPoolExecutor

# from dotenv import load_dotenv
# from jinja2 import Template
# from openai import OpenAI
# from prompts import ANSWER_PROMPT, ANSWER_PROMPT_GRAPH
# from tqdm import tqdm

# from mem0 import MemoryClient

# load_dotenv()


# class MemorySearch:
#     def __init__(self, output_path="results.json", top_k=10, filter_memories=False, is_graph=False):
#         self.mem0_client = MemoryClient(
#             api_key=os.getenv("MEM0_API_KEY"),
#             org_id=os.getenv("MEM0_ORGANIZATION_ID"),
#             project_id=os.getenv("MEM0_PROJECT_ID"),
#         )
#         self.top_k = top_k
#         self.openai_client = OpenAI()
#         self.results = defaultdict(list)
#         self.output_path = output_path
#         self.filter_memories = filter_memories
#         self.is_graph = is_graph

#         if self.is_graph:
#             self.ANSWER_PROMPT = ANSWER_PROMPT_GRAPH
#         else:
#             self.ANSWER_PROMPT = ANSWER_PROMPT

#     def search_memory(self, user_id, query, max_retries=3, retry_delay=1):
#         start_time = time.time()
#         retries = 0
#         while retries < max_retries:
#             try:
#                 if self.is_graph:
#                     print("Searching with graph")
#                     memories = self.mem0_client.search(
#                         query,
#                         user_id=user_id,
#                         top_k=self.top_k,
#                         filter_memories=self.filter_memories,
#                         enable_graph=True,
#                         output_format="v1.1",
#                     )
#                 else:
#                     memories = self.mem0_client.search(
#                         query, user_id=user_id, top_k=self.top_k, filter_memories=self.filter_memories
#                     )
#                 break
#             except Exception as e:
#                 print("Retrying...")
#                 retries += 1
#                 if retries >= max_retries:
#                     raise e
#                 time.sleep(retry_delay)

#         end_time = time.time()
#         if not self.is_graph:
#             semantic_memories = [
#                 {
#                     "memory": memory["memory"],
#                     "timestamp": memory["metadata"]["timestamp"],
#                     "score": round(memory["score"], 2),
#                 }
#                 for memory in memories
#             ]
#             graph_memories = None
#         else:
#             semantic_memories = [
#                 {
#                     "memory": memory["memory"],
#                     "timestamp": memory["metadata"]["timestamp"],
#                     "score": round(memory["score"], 2),
#                 }
#                 for memory in memories["results"]
#             ]
#             graph_memories = [
#                 {"source": relation["source"], "relationship": relation["relationship"], "target": relation["target"]}
#                 for relation in memories["relations"]
#             ]
#         return semantic_memories, graph_memories, end_time - start_time

#     def answer_question(self, speaker_1_user_id, speaker_2_user_id, question, answer, category):
#         speaker_1_memories, speaker_1_graph_memories, speaker_1_memory_time = self.search_memory(
#             speaker_1_user_id, question
#         )
#         speaker_2_memories, speaker_2_graph_memories, speaker_2_memory_time = self.search_memory(
#             speaker_2_user_id, question
#         )

#         search_1_memory = [f"{item['timestamp']}: {item['memory']}" for item in speaker_1_memories]
#         search_2_memory = [f"{item['timestamp']}: {item['memory']}" for item in speaker_2_memories]

#         template = Template(self.ANSWER_PROMPT)
#         answer_prompt = template.render(
#             speaker_1_user_id=speaker_1_user_id.split("_")[0],
#             speaker_2_user_id=speaker_2_user_id.split("_")[0],
#             speaker_1_memories=json.dumps(search_1_memory, indent=4),
#             speaker_2_memories=json.dumps(search_2_memory, indent=4),
#             speaker_1_graph_memories=json.dumps(speaker_1_graph_memories, indent=4),
#             speaker_2_graph_memories=json.dumps(speaker_2_graph_memories, indent=4),
#             question=question,
#         )

#         t1 = time.time()
#         response = self.openai_client.chat.completions.create(
#             model=os.getenv("MODEL"), messages=[{"role": "system", "content": answer_prompt}], temperature=0.0
#         )
#         t2 = time.time()
#         response_time = t2 - t1
#         return (
#             response.choices[0].message.content,
#             speaker_1_memories,
#             speaker_2_memories,
#             speaker_1_memory_time,
#             speaker_2_memory_time,
#             speaker_1_graph_memories,
#             speaker_2_graph_memories,
#             response_time,
#         )

#     def process_question(self, val, speaker_a_user_id, speaker_b_user_id):
#         question = val.get("question", "")
#         answer = val.get("answer", "")
#         category = val.get("category", -1)
#         evidence = val.get("evidence", [])
#         adversarial_answer = val.get("adversarial_answer", "")

#         (
#             response,
#             speaker_1_memories,
#             speaker_2_memories,
#             speaker_1_memory_time,
#             speaker_2_memory_time,
#             speaker_1_graph_memories,
#             speaker_2_graph_memories,
#             response_time,
#         ) = self.answer_question(speaker_a_user_id, speaker_b_user_id, question, answer, category)

#         result = {
#             "question": question,
#             "answer": answer,
#             "category": category,
#             "evidence": evidence,
#             "response": response,
#             "adversarial_answer": adversarial_answer,
#             "speaker_1_memories": speaker_1_memories,
#             "speaker_2_memories": speaker_2_memories,
#             "num_speaker_1_memories": len(speaker_1_memories),
#             "num_speaker_2_memories": len(speaker_2_memories),
#             "speaker_1_memory_time": speaker_1_memory_time,
#             "speaker_2_memory_time": speaker_2_memory_time,
#             "speaker_1_graph_memories": speaker_1_graph_memories,
#             "speaker_2_graph_memories": speaker_2_graph_memories,
#             "response_time": response_time,
#         }

#         # Save results after each question is processed
#         with open(self.output_path, "w") as f:
#             json.dump(self.results, f, indent=4)

#         return result

#     def process_data_file(self, file_path):
#         with open(file_path, "r") as f:
#             data = json.load(f)

#         for idx, item in tqdm(enumerate(data), total=len(data), desc="Processing conversations"):
#             qa = item["qa"]
#             conversation = item["conversation"]
#             speaker_a = conversation["speaker_a"]
#             speaker_b = conversation["speaker_b"]

#             speaker_a_user_id = f"{speaker_a}_{idx}"
#             speaker_b_user_id = f"{speaker_b}_{idx}"

#             for question_item in tqdm(
#                 qa, total=len(qa), desc=f"Processing questions for conversation {idx}", leave=False
#             ):
#                 result = self.process_question(question_item, speaker_a_user_id, speaker_b_user_id)
#                 self.results[idx].append(result)

#                 # Save results after each question is processed
#                 with open(self.output_path, "w") as f:
#                     json.dump(self.results, f, indent=4)

#         # Final save at the end
#         with open(self.output_path, "w") as f:
#             json.dump(self.results, f, indent=4)

#     def process_questions_parallel(self, qa_list, speaker_a_user_id, speaker_b_user_id, max_workers=1):
#         def process_single_question(val):
#             result = self.process_question(val, speaker_a_user_id, speaker_b_user_id)
#             # Save results after each question is processed
#             with open(self.output_path, "w") as f:
#                 json.dump(self.results, f, indent=4)
#             return result

#         with ThreadPoolExecutor(max_workers=max_workers) as executor:
#             results = list(
#                 tqdm(executor.map(process_single_question, qa_list), total=len(qa_list), desc="Answering Questions")
#             )

#         # Final save at the end
#         with open(self.output_path, "w") as f:
#             json.dump(self.results, f, indent=4)

#         return results


import json
import os
import time
from collections import defaultdict
from dotenv import load_dotenv
from jinja2 import Template
from openai import OpenAI
from tqdm import tqdm

try:
    from prompts import ANSWER_PROMPT, ANSWER_PROMPT_GRAPH
except ImportError:
    ANSWER_PROMPT = """
    Based on the memories provided, answer the question.
    Memories of Speaker 1: {{ speaker_1_memories }}
    Memories of Speaker 2: {{ speaker_2_memories }}
    Question: {{ question }}
    """
    ANSWER_PROMPT_GRAPH = ANSWER_PROMPT

from mem0 import Memory

load_dotenv()

BASE_URL = "http://localhost:8000/v1"
MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"

ROOT_MEMORY_PATH = "/home/jwuev/code/mem0/evaluation/local_memory"
VECTOR_DB_PATH = "vector_db"
HISTORY_DB_PATH = os.path.join(ROOT_MEMORY_PATH, "history.db")
MEMORY_COLLECTION_PATH = os.path.join(ROOT_MEMORY_PATH, "test_dataset_memory")

class MemorySearch:
    def __init__(self, output_path="results.json", top_k=10, filter_memories=False, is_graph=False):
        self.config = {
            "llm": {
                "provider": "openai",
                "config": {
                    "model": MODEL_NAME,
                    "openai_base_url": BASE_URL,
                    "api_key": "EMPTY",
                    "temperature": 0.1,
                    "max_tokens": 200000,
                },
            },
            "embedder": {
                "provider": "huggingface",
                "config": {
                    "model": "/home/jwuev/.cache/huggingface/hub/models--BAAI--bge-m3/snapshots/5617a9f61b028005a4858fdac845db406aefb181",
                    "model_kwargs": {"device": "cuda"}
                },
            },
            "vector_store": {
                "provider": "qdrant",
                "config": {
                    "collection_name": MEMORY_COLLECTION_PATH,
                    "path": VECTOR_DB_PATH,
                    "embedding_model_dims": 1024,
                },
            },
            "history_db_path": HISTORY_DB_PATH,
            "version": "v1.1"
        }

        if is_graph:
            self.config["graph_store"] = {
                "provider": "neo4j",
                "config": {
                    "url": "neo4j://localhost:7687",
                    "username": "neo4j",
                    "password": "password"
                }
            }

        self.memory = Memory.from_config(self.config)
        self.openai_client = OpenAI(base_url=BASE_URL, api_key="EMPTY")
        
        self.top_k = top_k
        self.results = defaultdict(list)
        self.output_path = output_path
        self.filter_memories = filter_memories
        self.is_graph = is_graph

        self.ANSWER_PROMPT = ANSWER_PROMPT_GRAPH if self.is_graph else ANSWER_PROMPT

    def search_memory(self, user_id, query, max_retries=3, retry_delay=1):
        """执行记忆搜索"""
        start_time = time.time()
        retries = 0
        memories = None
        
        while retries < max_retries:
            try:
                if self.is_graph:
                    search_results = self.memory.search(
                        query, user_id=user_id, limit=self.top_k
                    )
                    memories = search_results
                else:
                    memories = self.memory.search(
                        query, user_id=user_id, limit=self.top_k
                    )
                break
            except Exception as e:
                print(f"Error searching memory for {user_id}: {e}")
                retries += 1
                if retries >= max_retries:
                    raise e
                time.sleep(retry_delay)

        end_time = time.time()
        
        if not self.is_graph:
            semantic_memories = [
                {
                    "memory": m["memory"],
                    "timestamp": m.get("metadata", {}).get("timestamp", "unknown"),
                    "score": round(m.get("score", 0.0), 2),
                }
                for m in (memories if isinstance(memories, list) else [])
            ]
            graph_memories = None
        else:
            if isinstance(memories, dict) and "results" in memories:
                semantic_raw = memories["results"]
                relations_raw = memories.get("relations", [])
            elif isinstance(memories, list):
                semantic_raw = memories
                relations_raw = []
            else:
                semantic_raw = []
                relations_raw = []

            semantic_memories = [
                {
                    "memory": m["memory"],
                    "timestamp": m.get("metadata", {}).get("timestamp", "unknown"),
                    "score": round(m.get("score", 0.0), 2),
                }
                for m in semantic_raw
            ]
            
            graph_memories = [
                {"source": r["source"], "relationship": r["relationship"], "target": r["target"]}
                for r in relations_raw
            ]
            
        return semantic_memories, graph_memories, end_time - start_time

    def answer_question(self, speaker_1_user_id, speaker_2_user_id, question, answer, category):
        """调用本地 LLM 回答问题"""
        speaker_1_memories, speaker_1_graph_memories, speaker_1_memory_time = self.search_memory(
            speaker_1_user_id, question
        )
        speaker_2_memories, speaker_2_graph_memories, speaker_2_memory_time = self.search_memory(
            speaker_2_user_id, question
        )

        search_1_memory = [f"{item['timestamp']}: {item['memory']}" for item in speaker_1_memories]
        search_2_memory = [f"{item['timestamp']}: {item['memory']}" for item in speaker_2_memories]

        template = Template(self.ANSWER_PROMPT)
        
        render_kwargs = {
            "speaker_1_user_id": speaker_1_user_id.split("_")[0],
            "speaker_2_user_id": speaker_2_user_id.split("_")[0],
            "speaker_1_memories": json.dumps(search_1_memory, indent=4, ensure_ascii=False),
            "speaker_2_memories": json.dumps(search_2_memory, indent=4, ensure_ascii=False),
            "question": question,
        }
        
        if self.is_graph:
            render_kwargs["speaker_1_graph_memories"] = json.dumps(speaker_1_graph_memories, indent=4, ensure_ascii=False)
            render_kwargs["speaker_2_graph_memories"] = json.dumps(speaker_2_graph_memories, indent=4, ensure_ascii=False)

        answer_prompt = template.render(**render_kwargs)

        t1 = time.time()
        # 调用本地 LLM
        response = self.openai_client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "system", "content": answer_prompt}],
            temperature=0.0
        )
        t2 = time.time()
        
        return (
            response.choices[0].message.content,
            speaker_1_memories,
            speaker_2_memories,
            speaker_1_memory_time,
            speaker_2_memory_time,
            speaker_1_graph_memories,
            speaker_2_graph_memories,
            t2 - t1,
        )

    def process_question(self, val, speaker_a_user_id, speaker_b_user_id):
        """处理单个问题并返回结果字典"""
        question = val.get("question", "")
        answer = val.get("answer", "")
        category = val.get("category", -1)
        evidence = val.get("evidence", [])
        adversarial_answer = val.get("adversarial_answer", "")

        (
            response,
            speaker_1_memories,
            speaker_2_memories,
            speaker_1_memory_time,
            speaker_2_memory_time,
            speaker_1_graph_memories,
            speaker_2_graph_memories,
            response_time,
        ) = self.answer_question(speaker_a_user_id, speaker_b_user_id, question, answer, category)

        result = {
            "question": question,
            "answer": answer,
            "category": category,
            "evidence": evidence,
            "response": response,
            "adversarial_answer": adversarial_answer,
            "speaker_1_memories": speaker_1_memories,
            "speaker_2_memories": speaker_2_memories,
            "num_speaker_1_memories": len(speaker_1_memories),
            "num_speaker_2_memories": len(speaker_2_memories),
            "speaker_1_memory_time": speaker_1_memory_time,
            "speaker_2_memory_time": speaker_2_memory_time,
            "speaker_1_graph_memories": speaker_1_graph_memories,
            "speaker_2_graph_memories": speaker_2_graph_memories,
            "response_time": response_time,
        }
        return result

    def process_data_file(self, file_path):
        """主入口：读取文件并处理所有对话和问题"""
        with open(file_path, "r", encoding='utf-8') as f:
            data = json.load(f)

        for idx, item in tqdm(enumerate(data), total=len(data), desc="Processing conversations"):
            qa = item["qa"]
            conversation = item["conversation"]
            speaker_a = conversation["speaker_a"]
            speaker_b = conversation["speaker_b"]

            speaker_a_user_id = f"{speaker_a}_{idx}"
            speaker_b_user_id = f"{speaker_b}_{idx}"

            for question_item in tqdm(
                qa, total=len(qa), desc=f"Processing questions for conv {idx}", leave=False
            ):
                result = self.process_question(question_item, speaker_a_user_id, speaker_b_user_id)
                self.results[idx].append(result)

                self.save_results()

        self.save_results()

    def process_questions_sequential(self, qa_list, speaker_a_user_id, speaker_b_user_id):
        """
        替代原本的 process_questions_parallel。
        纯单线程顺序执行。
        """
        results_list = []
        for val in tqdm(qa_list, desc="Answering Questions"):
            result = self.process_question(val, speaker_a_user_id, speaker_b_user_id)
            results_list.append(result)
            
            # 如果你希望在这里也更新全局 results 并保存，可以取消注释下面两行：
            # (注意：这取决于此方法是如何被调用的，通常 process_data_file 已经处理了保存)
            # self.results["current_batch"].append(result) 
            # self.save_results()
            
        return results_list

    def save_results(self):
        """辅助方法：保存结果到文件"""
        try:
            with open(self.output_path, "w", encoding='utf-8') as f:
                json.dump(self.results, f, indent=4, ensure_ascii=False)
        except Exception as e:
            print(f"Warning: Failed to save results: {e}")