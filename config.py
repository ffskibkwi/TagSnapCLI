import os
import json
import sys
import textwrap
from pathlib import Path
import configparser
import textwrap
import numpy as np
import re

# 动态检查并导入依赖
try:
    import chromadb
except ImportError:
    print("错误: 核心依赖 'chromadb' 未安装。", file=sys.stderr)
    print("请在命令行中运行: pip install chromadb", file=sys.stderr)
    sys.exit(1)

# 从 ollama_demo.py 导入可重用的函数
try:
    from ollama_demo import load_embedding_config, call_ollama_embeddings, _extract_embedding
except ImportError:
    print("错误: 无法从 ollama_demo.py 导入所需函数。", file=sys.stderr)
    print("请确保 ollama_demo.py 与当前文件在同一目录下。", file=sys.stderr)
    sys.exit(1)

APP_NAME = "TagSnapCLI"
CONFIG_FILE = Path.cwd() / "config.ini"
PROMPTS_DIR = Path.cwd() / "prompts"
SEGMENTER_FILE = PROMPTS_DIR / "segmenter.prompt"
TAG_FILE = PROMPTS_DIR / "tag.prompt"
TAG_SEG_FILE = PROMPTS_DIR / "tag_seg.prompt"
TAG_ADD_CHECK_FILE = PROMPTS_DIR / "tag_add_check.prompt"
FIELD_ADD_CHECK_FILE = PROMPTS_DIR / "field_add_check.prompt"
ADD_CHECK_FILE = PROMPTS_DIR / "add_check.prompt"

# 初始词库文件
INIT_KEYWORD_FILE = Path.cwd() / "init_tag_lab.json"
INIT_FIELD_FILE = Path.cwd() / "init_field_lab.json"
INIT_TYPE_FILE = Path.cwd() / "init_type_lab.json"

# 当前类型列表文件
CURRENT_TYPE_FILE = Path.cwd() / "curr_type_lab.txt"

# 向量数据库
DB_PATH = Path.cwd() / "chroma_db_cosine"
COLLECTION_KEYWORD = "keyword_embeddings_cosine"
COLLECTION_FIELD = "field_embeddings_cosine"


def load_config() -> dict:
    """读取本地 config.ini 并返回配置字典。"""
    if not CONFIG_FILE.exists():
        raise FileNotFoundError(
            f"未找到 {CONFIG_FILE.name}，请先运行: python main.py init 或根据 README 创建配置文件"
        )

    parser = configparser.ConfigParser()
    parser.read(CONFIG_FILE, encoding="utf-8")

    if not parser.has_section("gemini"):
        raise KeyError("config.ini 缺少 [gemini] 配置段")

    api_key = parser.get("gemini", "api_key", fallback=None)
    model_name = parser.get("gemini", "model", fallback="gemini-1.5-flash")

    if not api_key:
        raise ValueError("[gemini] api_key 不能为空")

    proxy_config = {
        "http": parser.get("proxy", "http", fallback=""),
        "https": parser.get("proxy", "https", fallback=""),
    }

    text_note_dir = ""
    if parser.has_section("file"):
        text_note_dir = parser.get("file", "text_note_dir", fallback="").strip()

    return {
        "api_key": api_key,
        "model": model_name,
        "proxy": proxy_config,
        "text_note_dir": text_note_dir,
    }


def setup_proxy(proxy_config: dict) -> None:
    """设置代理。"""
    if proxy_config.get("http"):
        os.environ["http_proxy"] = proxy_config["http"]
    if proxy_config.get("https"):
        os.environ["https_proxy"] = proxy_config["https"]


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


def _init_vector_collection(collection_name: str, vocab_words: list) -> tuple:
    """
    初始化向量数据库，如果数据库为空则填充数据。
    
    Returns:
        tuple: (chromadb_client, collection)
    """
    print(f"正在加载或创建向量数据库于: {DB_PATH}")
    client = chromadb.PersistentClient(path=str(DB_PATH))
    
    # 创建或获取集合，使用余弦距离
    print(f"加载或创建集合: {collection_name} (使用余弦距离)")
    collection = client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"}
    )
    
    # 检查数据库是否需要填充数据
    if collection.count() == 0:
        print("数据库为空，正在从词库文件填充数据...")
        words = vocab_words or []
        if not isinstance(words, list) or not words:
            raise ValueError("错误: 词库内容格式不正确或为空。")

        # 去重并清洗空项，保持顺序
        cleaned_words = []
        seen = set()
        for w in words:
            if not isinstance(w, (str, int, float)):
                continue
            s = str(w).strip()
            if not s or s in seen:
                continue
            seen.add(s)
            cleaned_words.append(s)
        words = cleaned_words

        try:
            base_url, model = load_embedding_config(CONFIG_FILE)
            print(f"使用 Ollama 服务: {base_url} | 模型: {model}")
        except Exception as e:
            raise RuntimeError(f"错误: 加载 Ollama 配置失败: {e}")

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
                raise RuntimeError(f"错误: 为 '{word}' 生成向量时失败: {e}")
        
        collection.add(
            embeddings=embeddings_to_add,
            metadatas=[{"word": w} for w in words],
            ids=words
        )
        print("数据库填充完成！")
    else:
        print(f"数据库已加载，包含 {collection.count()} 个词条。")
    
    return client, collection


def _load_vocab_list(path: Path) -> list:
    """
    尝试以多种方式解析初始词库文件，容忍注释、尾逗号或非严格 JSON。
    优先解析为 JSON 数组；失败时尝试提取引号中的字符串；再失败则逐行读取非空行。
    """
    if not path.exists():
        raise FileNotFoundError(f"未找到词库文件: {path}")

    text = path.read_text(encoding="utf-8", errors="replace")
    # 去除 BOM
    if text and text[0] == "\ufeff":
        text = text.lstrip("\ufeff")

    # 移除 // 与 # 行注释，以及 /* ... */ 块注释
    lines = []
    for ln in text.splitlines():
        if ln.strip().startswith("//") or ln.strip().startswith("#"):
            continue
        lines.append(ln)
    text_nc = "\n".join(lines)
    text_nc = re.sub(r"/\*.*?\*/", "", text_nc, flags=re.S)

    # 首选严格 JSON
    try:
        data = json.loads(text_nc)
        if isinstance(data, list):
            return [str(x).strip() for x in data if isinstance(x, (str, int, float)) and str(x).strip()]
    except Exception:
        pass

    # 尝试提取引号中的字符串（适用于 JSON 数组但含尾逗号等问题）
    try:
        tokens = re.findall(r'"([^"]+)"', text_nc)
        if tokens:
            return [t.strip() for t in tokens if t.strip()]
    except Exception:
        pass

    # 最后回退：逐行解析非空行
    fallback = []
    for ln in text_nc.splitlines():
        val = ln.strip().strip(',')
        if val:
            fallback.append(val)
    if fallback:
        return fallback

    raise ValueError(f"无法从 {path.name} 解析出有效的词条列表")


def init_keyword_vector_database() -> tuple:
    """
    初始化/加载关键词向量库。
    返回 (client, keyword_collection)
    """
    if not INIT_KEYWORD_FILE.exists():
        raise FileNotFoundError(f"错误: 关键词初始文件 {INIT_KEYWORD_FILE} 不存在。")
    words = _load_vocab_list(INIT_KEYWORD_FILE)
    return _init_vector_collection(COLLECTION_KEYWORD, words)


def init_field_vector_database() -> tuple:
    """
    初始化/加载领域向量库。
    返回 (client, field_collection)
    """
    if not INIT_FIELD_FILE.exists():
        raise FileNotFoundError(f"错误: 领域初始文件 {INIT_FIELD_FILE} 不存在。")
    words = _load_vocab_list(INIT_FIELD_FILE)
    return _init_vector_collection(COLLECTION_FIELD, words)


def _get_similar(text: str, collection, n_results: int) -> list:
    """
    获取与输入文本最相似的标签。
    
    Args:
        text (str): 输入文本
        collection: ChromaDB集合
        n_results (int): 返回结果数量
        
    Returns:
        list: 最相似的标签列表
    """
    try:
        base_url, model = load_embedding_config(CONFIG_FILE)
    except Exception as e:
        raise RuntimeError(f"错误: 加载 Ollama 配置失败: {e}")
    
    # 为查询文本生成向量
    try:
        resp_json = call_ollama_embeddings(base_url, model, text)
        query_vector = _extract_embedding(resp_json)
        normalized_query_vector = normalize_vector(query_vector)
    except Exception as e:
        raise RuntimeError(f"错误: 生成查询向量失败: {e}")
    
    # 在数据库中查询相似标签
    results = collection.query(
        query_embeddings=[normalized_query_vector],
        n_results=n_results
    )
    
    metadatas = results.get('metadatas', [[]])[0]
    if not metadatas:
        return []
    
    return [meta.get('word', 'N/A') for meta in metadatas]


def get_similar_keywords(text: str, keyword_collection, n_results: int = 10) -> list:
    return _get_similar(text, keyword_collection, n_results)


def get_similar_fields(text: str, field_collection, n_results: int = 3) -> list:
    return _get_similar(text, field_collection, n_results)


def update_keyword_vector_database(collection, new_keywords: list) -> None:
    """
    将新标签添加到向量数据库中。
    
    Args:
        collection: ChromaDB集合
        new_tags (list): 新标签列表
    """
    if not new_keywords:
        return
    
    try:
        base_url, model = load_embedding_config(CONFIG_FILE)
    except Exception as e:
        print(f"警告: 无法更新向量数据库 - 加载 Ollama 配置失败: {e}")
        return
    
    print(f"正在将 {len(new_keywords)} 个新关键词添加到向量数据库...")
    embeddings_to_add = []
    metadatas_to_add = []
    ids_to_add = []
    
    for tag in new_keywords:
        # 检查标签是否已存在
        try:
            existing = collection.get(ids=[tag])
            if existing['ids']:
                continue  # 标签已存在，跳过
        except:
            pass  # 标签不存在，继续添加
        
        try:
            resp_json = call_ollama_embeddings(base_url, model, tag)
            vector = _extract_embedding(resp_json)
            normalized_vector = normalize_vector(vector)
            embeddings_to_add.append(normalized_vector)
            metadatas_to_add.append({"word": tag})
            ids_to_add.append(tag)
        except Exception as e:
            print(f"警告: 为标签 '{tag}' 生成向量时失败: {e}")
    
    if embeddings_to_add:
        collection.add(
            embeddings=embeddings_to_add,
            metadatas=metadatas_to_add,
            ids=ids_to_add
        )
        print(f"成功添加 {len(embeddings_to_add)} 个新关键词到向量数据库")


def update_field_vector_database(collection, new_fields: list) -> None:
    """
    将新领域添加到向量数据库中。
    """
    if not new_fields:
        return
    try:
        base_url, model = load_embedding_config(CONFIG_FILE)
    except Exception as e:
        print(f"警告: 无法更新向量数据库 - 加载 Ollama 配置失败: {e}")
        return
    print(f"正在将 {len(new_fields)} 个新领域添加到向量数据库...")
    embeddings_to_add, metadatas_to_add, ids_to_add = [], [], []
    for fld in new_fields:
        try:
            existing = collection.get(ids=[fld])
            if existing['ids']:
                continue
        except:
            pass
        try:
            resp_json = call_ollama_embeddings(base_url, model, fld)
            vector = _extract_embedding(resp_json)
            normalized_vector = normalize_vector(vector)
            embeddings_to_add.append(normalized_vector)
            metadatas_to_add.append({"word": fld})
            ids_to_add.append(fld)
        except Exception as e:
            print(f"警告: 为领域 '{fld}' 生成向量时失败: {e}")
    if embeddings_to_add:
        collection.add(
            embeddings=embeddings_to_add,
            metadatas=metadatas_to_add,
            ids=ids_to_add
        )
        print(f"成功添加 {len(embeddings_to_add)} 个新领域到向量数据库")


def load_tag_prompt() -> str:
    """
    从 prompts/tag.prompt 中加载标签生成的提示词。
    """
    if not TAG_FILE.exists():
        raise FileNotFoundError(
            f"未找到 {TAG_FILE.as_posix()}，请确保文件存在"
        )
    
    try:
        prompt_text = TAG_FILE.read_text(encoding="utf-8")
    except Exception as e:
        raise ValueError(f"无法读取prompt文件: {e}")
    
    # 统一换行并去除最外层公共缩进
    prompt_text = prompt_text.replace("\r\n", "\n").replace("\r", "\n")
    prompt_text = textwrap.dedent(prompt_text).strip()
    
    if not prompt_text:
        raise ValueError(f"{TAG_FILE} 中的提示词内容为空")
    
    return prompt_text


def load_tag_seg_prompt() -> str:
    """
    从 prompts/tag_seg.prompt 中加载标签分析的提示词。
    """
    if not TAG_SEG_FILE.exists():
        raise FileNotFoundError(
            f"未找到 {TAG_SEG_FILE.as_posix()}，请确保文件存在"
        )
    
    try:
        prompt_text = TAG_SEG_FILE.read_text(encoding="utf-8")
    except Exception as e:
        raise ValueError(f"无法读取prompt文件: {e}")
    
    # 统一换行并去除最外层公共缩进
    prompt_text = prompt_text.replace("\r\n", "\n").replace("\r", "\n")
    prompt_text = textwrap.dedent(prompt_text).strip()
    
    if not prompt_text:
        raise ValueError(f"{TAG_SEG_FILE} 中的提示词内容为空")
    
    return prompt_text


def load_tag_add_check_prompt() -> str:
    """
    从 prompts/tag_add_check.prompt 中加载新增标签判重的提示词。
    """
    if not TAG_ADD_CHECK_FILE.exists():
        raise FileNotFoundError(
            f"未找到 {TAG_ADD_CHECK_FILE.as_posix()}，请确保文件存在"
        )
    
    try:
        prompt_text = TAG_ADD_CHECK_FILE.read_text(encoding="utf-8")
    except Exception as e:
        raise ValueError(f"无法读取prompt文件: {e}")
    
    # 统一换行并去除最外层公共缩进
    prompt_text = prompt_text.replace("\r\n", "\n").replace("\r", "\n")
    prompt_text = textwrap.dedent(prompt_text).strip()
    
    if not prompt_text:
        raise ValueError(f"{TAG_ADD_CHECK_FILE} 中的提示词内容为空")
    
    return prompt_text


def load_add_check_prompt() -> str:
    """
    从 prompts/add_check.prompt 中加载合并的领域/关键词判重提示词。
    """
    if not ADD_CHECK_FILE.exists():
        raise FileNotFoundError(
            f"未找到 {ADD_CHECK_FILE.as_posix()}，请确保文件存在"
        )
    try:
        prompt_text = ADD_CHECK_FILE.read_text(encoding="utf-8")
    except Exception as e:
        raise ValueError(f"无法读取prompt文件: {e}")
    prompt_text = prompt_text.replace("\r\n", "\n").replace("\r", "\n")
    prompt_text = textwrap.dedent(prompt_text).strip()
    if not prompt_text:
        raise ValueError(f"{ADD_CHECK_FILE} 中的提示词内容为空")
    return prompt_text


def load_prompt() -> str:
    '''读取 prompts/segmenter.prompt 中的提示词。'''
    if not SEGMENTER_FILE.exists():
        raise FileNotFoundError(
            f"未找到 {SEGMENTER_FILE.as_posix()}，请先运行: python main.py init 或根据 README 创建提示文件"
        )

    try:
        prompt_text = SEGMENTER_FILE.read_text(encoding="utf-8")
    except Exception as e:
        raise ValueError(f"无法读取prompt文件: {e}")

    # 统一换行并去除最外层公共缩进
    prompt_text = prompt_text.replace("\r\n", "\n").replace("\r", "\n")
    prompt_text = textwrap.dedent(prompt_text).strip()

    if not prompt_text:
        # 提供一个合理默认值（语义分割）
        prompt_text = (
            "你是一个精准的中文语义分割助手。请将输入文本按语义主题划分为若干段落：\n"
            "- 保障每一段内部主题一致，跨段主题尽量独立；\n"
            "- 为每段生成短标题（不超过12字），并给出1-3句简要描述；\n"
            "- 保留关键数据、时间、人物与因果；\n"
            "- 输出 JSON 数组，每个元素包含 title、summary、text 字段；\n"
            "- 不要附加其他说明。"
        )
    return prompt_text


def init_files(force: bool = False) -> None:
    """生成示例 config.ini 与 segmenter.prompt。"""
    sample_config = textwrap.dedent(
        """
        [gemini]
        # 必填：你的 Google Generative AI API Key
        api_key = YOUR_API_KEY_HERE
        # 可选：模型名称，常用：gemini-1.5-flash 或 gemini-1.5-pro
        model = gemini-1.5-flash

        [embedding]
        # Ollama 服务地址，用于向量嵌入
        ip_addr = http://127.0.0.1:11434
        # 嵌入模型名称，推荐使用 bge-m3
        model = bge-m3

        [proxy]
        # 可选：HTTP/HTTPS 代理。示例：http://127.0.0.1:7890
        http =
        https =

        [file]
        # 可选：用于保存文本笔记的目录（Markdown）。
        # 例如：E:/Notes/TextNotes
        text_note_dir =
        """
    ).strip()

    sample_prompt = textwrap.dedent(
        """
        你是一个精准的中文语义分割助手。请将输入文本按语义主题划分为若干段落：
        - 保障每一段内部主题一致，跨段主题尽量独立；
        - 为每段生成短标题（不超过12字），并给出1-3句简要描述；
        - 保留关键数据、时间、人物与因果；
        - 输出 JSON 数组，每个元素包含 title、summary、text 字段；
        - 不要附加其他说明。
        """
    ).strip()

    PROMPTS_DIR.mkdir(parents=True, exist_ok=True)
    for path, content in [(CONFIG_FILE, sample_config), (SEGMENTER_FILE, sample_prompt)]:
        if path.exists() and not force:
            print(f"跳过：{path.name} 已存在。如需覆盖，请加 --force")
            continue
        path.write_text(content + "\n", encoding="utf-8")
        print(f"已生成 {path.name}")

    # 初始化 curr_type_lab.txt（如不存在且有 init_type_lab.json）
    if not CURRENT_TYPE_FILE.exists() and INIT_TYPE_FILE.exists():
        try:
            types_list = json.loads(INIT_TYPE_FILE.read_text(encoding="utf-8"))
            if isinstance(types_list, list) and types_list:
                CURRENT_TYPE_FILE.write_text("\n".join(t.strip() for t in types_list if isinstance(t, str) and t.strip()) + "\n", encoding="utf-8")
                print("已生成 curr_type_lab.txt")
        except Exception as e:
            print(f"警告：初始化 curr_type_lab.txt 失败：{e}")


def load_current_types() -> list:
    if not CURRENT_TYPE_FILE.exists():
        return []
    try:
        lines = CURRENT_TYPE_FILE.read_text(encoding="utf-8").splitlines()
        return [ln.strip() for ln in lines if ln.strip()]
    except Exception:
        return []


def append_types_if_missing(new_types: list) -> None:
    if not new_types:
        return
    existing = set(load_current_types())
    added = False
    for t in new_types:
        if isinstance(t, str):
            v = t.strip()
            if v and v not in existing:
                existing.add(v)
                added = True
    if added:
        try:
            CURRENT_TYPE_FILE.write_text("\n".join(sorted(existing)) + "\n", encoding="utf-8")
        except Exception as e:
            print(f"警告：写入 curr_type_lab.txt 失败：{e}")


