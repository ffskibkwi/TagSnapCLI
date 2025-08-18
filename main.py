import typer

from config import (
    init_files, load_config, load_prompt, setup_proxy,
    init_vector_database, get_similar_tags, load_tag_seg_prompt, update_vector_database,
    load_tag_add_check_prompt
)
from analysis import build_model, segment_text, analyze_text_with_tags, extract_all_tags, adjudicate_supplementary_tags
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
    
    # 检查tag_seg.prompt
    tag_seg_file = Path.cwd() / "prompts" / "tag_seg.prompt"
    if tag_seg_file.exists():
        print("✓ prompts/tag_seg.prompt 存在")
    else:
        print("✗ 缺少 prompts/tag_seg.prompt 文件")

    # 检查tag_add_check.prompt
    tag_add_check_file = Path.cwd() / "prompts" / "tag_add_check.prompt"
    if tag_add_check_file.exists():
        print("✓ prompts/tag_add_check.prompt 存在")
    else:
        print("✗ 缺少 prompts/tag_add_check.prompt 文件")


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
        tag_prompt = load_tag_seg_prompt()
    except Exception as e:
        print(f"加载标签prompt失败: {e}")
        raise typer.Exit(1)
    
    # 构建使用标签prompt的模型
    model = build_model(cfg["api_key"], cfg["model"], tag_prompt)

    # 构建用于新增标签判重的模型
    try:
        tag_add_check_prompt = load_tag_add_check_prompt()
        judge_model = build_model(cfg["api_key"], cfg["model"], tag_add_check_prompt)
    except Exception as e:
        print(f"加载判重prompt失败: {e}")
        judge_model = None
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
            
            # 步骤3: 提取并判重补充标签，必要时更新向量数据库
            result_obj = analysis_result.get("result", {}) or {}
            tagging_details = result_obj.get("tagging_details", {}) or {}
            matched_tags = list(tagging_details.get("matched_tags", []) or [])
            supplementary_tags = list(tagging_details.get("supplementary_tags", []) or [])

            # 针对每个补充标签，先用 bge-m3 做一次相似检索
            items_for_llm = []
            moved_to_matched = set()
            supp_tag_to_similars = {}
            for supp in supplementary_tags:
                try:
                    existing_similars = get_similar_tags(supp, collection, n_results=8)
                except Exception:
                    existing_similars = []
                supp_tag_to_similars[supp] = existing_similars
                # 特殊情况：相似集中存在与补充标签同名项 → 直接视为已存在，加入 matched
                if any(s.strip() == supp.strip() for s in (existing_similars or [])):
                    moved_to_matched.add(supp)
                else:
                    items_for_llm.append({
                        "supplementary_tag": supp,
                        "existing_similar_tags": existing_similars or []
                    })

            # 去重更新 matched_tags
            if moved_to_matched:
                matched_set = set(matched_tags)
                matched_set.update(moved_to_matched)
                matched_tags = list(matched_set)

            accepted_new_tags = []

            # 若可用，调用判重 LLM，对剩余补充标签进行裁决
            if judge_model is not None and items_for_llm:
                print("正在进行新增标签判重…")
                try:
                    judge_res = adjudicate_supplementary_tags(
                        judge_model,
                        document_content=text,
                        matched_tags=matched_tags,
                        items=items_for_llm,
                        temperature=0.0,
                    )
                    judgements = judge_res.get("judgements", [])
                except Exception as e:
                    print(f"判重失败，跳过本次新增：{e}")
                    judgements = []

                # 处理裁决结果
                for j in judgements:
                    judged_tag = j.get("judged_tag")
                    decision = j.get("decision")
                    final_tag = j.get("final_tag")
                    if not judged_tag or not final_tag:
                        continue
                    if decision == "ACCEPT_NEW" and final_tag.strip() == judged_tag.strip():
                        accepted_new_tags.append(judged_tag)
                    else:
                        # 视为重定向到已有标签，同步到 matched
                        if final_tag:
                            if final_tag not in matched_tags:
                                matched_tags.append(final_tag)

            # 根据移动与裁决结果，刷新补充标签与匹配标签集合
            final_matched = sorted(set(matched_tags))
            # 保留仅被接受为新增的补充标签
            final_supplementary = sorted(set(accepted_new_tags))

            # 写回到 analysis_result，用于 UI 展示
            analysis_result.setdefault("result", {}).setdefault("tagging_details", {})
            analysis_result["result"]["tagging_details"]["matched_tags"] = final_matched
            analysis_result["result"]["tagging_details"]["supplementary_tags"] = final_supplementary

            # 仅将被接受的新增补充标签写入向量库
            if final_supplementary:
                print(f"正在更新向量数据库，添加 {len(final_supplementary)} 个新标签...")
                update_vector_database(collection, final_supplementary)
            
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


