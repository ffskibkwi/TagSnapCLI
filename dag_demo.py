# -*- coding: utf-8 -*-
"""
本脚本构建一个基于 ChromaDB 的持久化向量数据库。

功能：
1.  重用 ollama_demo.py 的函数来连接 Ollama 服务并生成文本的嵌入向量。
2.  读取一个本地 JSON 文件 (init_tag_lab.json) 作为词库。
3.  如果向量数据库为空，则将词库中的每个词转换为向量，并存入 ChromaDB。
4.  将 ChromaDB 数据库持久化保存在本地磁盘（在名为 'chroma_db' 的文件夹中）。
5.  启动一个交互式命令行，接收用户输入的文本。
6.  为用户输入的文本生成向量，并在数据库中查询最相似的3个词语。
7.  打印查询结果，包括最相似的词语和它们的相似度距离。

V2 更新：
- 新增向量归一化功能，以确保距离计算的准确性和可比性。
- 明确指定使用 'cosine' 余弦距离，这在处理语义相似度时通常更优。
"""

import json
import sys
from pathlib import Path
import numpy as np  # 引入 numpy 用于高效的向量运算

# --- 动态检查并导入依赖 ---
try:
    import chromadb
except ImportError:
    print("错误: 核心依赖 'chromadb' 未安装。", file=sys.stderr)
    print("请在命令行中运行: pip install chromadb", file=sys.stderr)
    sys.exit(1)

# --- 从 ollama_demo.py 导入可重用的函数 ---
try:
    from ollama_demo import load_embedding_config, call_ollama_embeddings, _extract_embedding
except ImportError:
    print("错误: 无法从 ollama_demo.py 导入所需函数。", file=sys.stderr)
    print("请确保 ollama_demo.py 与 dag_demo.py 在同一目录下。", file=sys.stderr)
    sys.exit(1)

# --- 定义全局常量 ---
DB_PATH = Path.cwd() / "chroma_db_cosine" # 使用新目录以避免与旧数据库冲突
COLLECTION_NAME = "tag_embeddings_cosine"
CONFIG_FILE = Path.cwd() / "config.ini"
VOCAB_FILE = Path.cwd() / "init_tag_lab.json"


def normalize_vector(vector: list) -> list:
    """
    将输入的向量进行 L2 归一化处理。

    Args:
        vector (list): 原始向量。

    Returns:
        list: 归一化后的向量（单位向量）。
    """
    if not vector:
        return []
    # 使用 numpy 计算向量的 L2 范数（即向量的长度）
    np_vector = np.array(vector, dtype=np.float32)
    norm = np.linalg.norm(np_vector)
    # 如果范数为0（即零向量），则直接返回原向量以避免除以零的错误
    if norm == 0:
        return vector
    # 向量的每个分量都除以范数
    normalized_vector = np_vector / norm
    return normalized_vector.tolist()

def main() -> int:
    """
    程序的主入口函数。
    负责初始化向量数据库、按需填充数据，并处理用户的实时查询。
    """
    # --- 步骤 1: 初始化 ChromaDB 客户端 ---
    print(f"正在加载或创建向量数据库于: {DB_PATH}")
    client = chromadb.PersistentClient(path=str(DB_PATH))

    # --- 步骤 2: 加载或创建一个集合 (Collection) ---
    # 明确指定使用 'cosine' 距离，这对于语义相似度任务通常更优。
    print(f"加载或创建集合: {COLLECTION_NAME} (使用余弦距离)")
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}
    )

    # --- 步骤 3: 检查数据库是否需要填充数据 ---
    if collection.count() == 0:
        print("数据库为空，正在从词库文件填充数据...")
        
        if not VOCAB_FILE.exists():
            print(f"错误: 词库文件 {VOCAB_FILE} 不存在。", file=sys.stderr)
            return 1
        with open(VOCAB_FILE, "r", encoding="utf-8") as f:
            words = json.load(f)
        
        if not isinstance(words, list) or not words:
            print(f"错误: {VOCAB_FILE} 内容格式不正确或为空。", file=sys.stderr)
            return 1

        try:
            base_url, model = load_embedding_config(CONFIG_FILE)
            print(f"使用 Ollama 服务: {base_url} | 模型: {model}")
        except Exception as e:
            print(f"错误: 加载 Ollama 配置失败: {e}", file=sys.stderr)
            return 1

        print(f"正在为 {len(words)} 个词生成向量并存入数据库，请稍候...")
        embeddings_to_add = []
        for i, word in enumerate(words):
            try:
                resp_json = call_ollama_embeddings(base_url, model, word)
                vector = _extract_embedding(resp_json)
                # 在存入数据库前进行归一化
                normalized_vector = normalize_vector(vector)
                embeddings_to_add.append(normalized_vector)
                print(f"  ({i+1}/{len(words)}) 已处理: '{word}'")
            except Exception as e:
                print(f"错误: 为 '{word}' 生成向量时失败: {e}", file=sys.stderr)
                return 1
        
        collection.add(
            embeddings=embeddings_to_add,
            metadatas=[{"word": w} for w in words],
            ids=words
        )
        print("数据库填充完成！")
    else:
        print(f"数据库已加载，包含 {collection.count()} 个词条。")

    # --- 步骤 4: 启动交互式查询循环 ---
    print("\n--- 开始查询 ---")
    print("输入文本进行相似度查询，输入 'quit' 或 'exit' 退出。")
    
    try:
        base_url, model = load_embedding_config(CONFIG_FILE)
    except Exception as e:
        print(f"错误: 查询前加载 Ollama 配置失败: {e}", file=sys.stderr)
        return 1

    while True:
        try:
            query_text = input("> ").strip()
            if not query_text:
                continue
            if query_text.lower() in ["quit", "exit"]:
                print("程序已退出。")
                break

            print("正在为查询文本生成向量...")
            try:
                resp_json = call_ollama_embeddings(base_url, model, query_text)
                query_vector = _extract_embedding(resp_json)
                # 对查询向量也进行归一化
                normalized_query_vector = normalize_vector(query_vector)
            except Exception as e:
                print(f"错误: 生成查询向量失败: {e}", file=sys.stderr)
                continue

            print("正在查询相似词...")
            results = collection.query(
                query_embeddings=[normalized_query_vector],
                n_results=3
            )

            print("\n--- 查询结果 ---")
            distances = results.get('distances', [[]])[0]
            metadatas = results.get('metadatas', [[]])[0]
            
            if not metadatas:
                print("未找到匹配结果。")
            else:
                for i, (meta, dist) in enumerate(zip(metadatas, distances)):
                    word = meta.get('word', 'N/A')
                    # 余弦距离的值在 0 (最相似) 到 2 (最不相似) 之间
                    print(f"{i+1}. 词语: {word:<15} (余弦距离: {dist:.4f})")
            print("-" * 20 + "\n")

        except (KeyboardInterrupt, EOFError):
            print("\n程序已退出。")
            break
        except Exception as e:
            print(f"查询过程中发生未知错误: {e}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    # 在运行前检查 numpy 是否安装
    try:
        import numpy
    except ImportError:
        print("错误: 核心依赖 'numpy' 未安装。", file=sys.stderr)
        print("请在命令行中运行: pip install numpy", file=sys.stderr)
        sys.exit(1)
    sys.exit(main())