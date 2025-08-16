import typer

from config import (
    init_files, load_config, load_prompt, setup_proxy,
    init_vector_database, get_similar_tags, load_tag_prompt, update_vector_database
)
from analysis import build_model, segment_text, analyze_text_with_tags, extract_all_tags
from interface import interactive_loop


app = typer.Typer(help="TagSnapCLI - 常驻交互式语义分割助手")


@app.command()
def init(force: bool = typer.Option(False, "--force", help="如存在则覆盖生成")):
    """生成示例 config.ini 与 prompts/segmenter.ini，并检查向量数据库依赖。"""
    init_files(force=force)
    
    # 检查向量数据库相关依赖
    print("\n检查向量数据库依赖...")
    try:
        import chromadb
        import numpy
        print("✓ 向量数据库依赖已满足")
    except ImportError as e:
        print(f"✗ 缺少依赖: {e}")
        print("请运行: pip install chromadb numpy")
    
    # 检查ollama_demo.py
    try:
        from ollama_demo import load_embedding_config, call_ollama_embeddings, _extract_embedding
        print("✓ ollama_demo.py 可用")
    except ImportError:
        print("✗ 缺少 ollama_demo.py 或其中的函数")
    
    # 检查init_tag_lab.json
    from pathlib import Path
    vocab_file = Path.cwd() / "init_tag_lab.json"
    if vocab_file.exists():
        print("✓ init_tag_lab.json 存在")
    else:
        print("✗ 缺少 init_tag_lab.json 词库文件")
    
    # 检查tag.ini
    tag_file = Path.cwd() / "prompts" / "tag.ini"
    if tag_file.exists():
        print("✓ prompts/tag.ini 存在")
    else:
        print("✗ 缺少 prompts/tag.ini 文件")


@app.command()
def run(temperature: float = typer.Option(0.3, help="生成温度(0-1)")):
    """启动常驻交互式界面，持续接收文本并输出标签分析结果。"""
    cfg = load_config()
    setup_proxy(cfg["proxy"])  # 设置环境代理
    
    # 初始化向量数据库
    print("正在初始化向量数据库...")
    try:
        client, collection = init_vector_database()
    except Exception as e:
        print(f"向量数据库初始化失败: {e}")
        raise typer.Exit(1)
    
    # 加载标签分析的prompt
    try:
        tag_prompt = load_tag_prompt()
    except Exception as e:
        print(f"加载标签prompt失败: {e}")
        raise typer.Exit(1)
    
    # 构建使用标签prompt的模型
    model = build_model(cfg["api_key"], cfg["model"], tag_prompt)
    print("\n系统已就绪，开始标签分析模式...")

    def on_submit(text: str):
        try:
            # 步骤1: 获取与文本最相似的10个标签
            print("正在检索相似标签...")
            similar_tags = get_similar_tags(text, collection, n_results=10)
            
            if not similar_tags:
                return {
                    "text": "错误：未能从向量数据库中检索到任何相似标签",
                    "usage": None
                }
            
            # 步骤2: 使用Gemini进行标签分析
            print("正在进行标签分析...")
            analysis_result = analyze_text_with_tags(model, text, similar_tags, temperature=temperature)
            
            # 步骤3: 提取所有标签并更新向量数据库
            all_tags = extract_all_tags(analysis_result)
            supplementary_tags = analysis_result.get("result", {}).get("supplementary_tags", [])
            
            if supplementary_tags:
                print(f"正在更新向量数据库，添加 {len(supplementary_tags)} 个新标签...")
                update_vector_database(collection, supplementary_tags)
            
            return {
                "analysis_result": analysis_result,
                "similar_tags": similar_tags,
                "usage": analysis_result.get("usage")
            }
            
        except Exception as e:
            return {
                "text": f"分析过程中发生错误: {str(e)}",
                "usage": None
            }

    interactive_loop(on_submit)


if __name__ == "__main__":
    app()


