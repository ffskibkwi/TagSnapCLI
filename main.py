import typer
import re
import hashlib
from datetime import datetime
from pathlib import Path

from config import (
    init_files, load_config, load_prompt, setup_proxy,
    init_keyword_vector_database, init_field_vector_database,
    get_similar_keywords, get_similar_fields,
    load_tag_seg_prompt, load_add_check_prompt,
    update_keyword_vector_database, update_field_vector_database,
    load_current_types, append_types_if_missing
)
from analysis import (
    build_model, segment_text, analyze_text_with_candidates,
    adjudicate_fields_and_keywords
)
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
    
    # 检查初始 keyword/field/type 资源
    from pathlib import Path
    tag_vocab_file = Path.cwd() / "init_tag_lab.json"
    field_vocab_file = Path.cwd() / "init_field_lab.json"
    type_vocab_file = Path.cwd() / "init_type_lab.json"
    if tag_vocab_file.exists():
        print("✓ init_tag_lab.json (keywords) 存在")
    else:
        print("✗ 缺少 init_tag_lab.json (keywords) 词库文件")
    if field_vocab_file.exists():
        print("✓ init_field_lab.json (fields) 存在")
    else:
        print("✗ 缺少 init_field_lab.json (fields) 词库文件")
    if type_vocab_file.exists():
        print("✓ init_type_lab.json (types) 存在")
    else:
        print("✗ 缺少 init_type_lab.json (types) 词库文件")
    
    # 检查tag_seg.prompt
    tag_seg_file = Path.cwd() / "prompts" / "tag_seg.prompt"
    if tag_seg_file.exists():
        print("✓ prompts/tag_seg.prompt 存在")
    else:
        print("✗ 缺少 prompts/tag_seg.prompt 文件")

    # 检查判重 prompts
    tag_add_check_file = Path.cwd() / "prompts" / "tag_add_check.prompt"
    if tag_add_check_file.exists():
        print("✓ prompts/tag_add_check.prompt 存在")
    else:
        print("✗ 缺少 prompts/tag_add_check.prompt 文件")
    field_add_check_file = Path.cwd() / "prompts" / "field_add_check.prompt"
    if field_add_check_file.exists():
        print("✓ prompts/field_add_check.prompt 存在")
    else:
        print("✗ 缺少 prompts/field_add_check.prompt 文件")


@app.command()
def run(
    temperature: float = typer.Option(0.3, help="生成温度(0-1)"),
    debug: bool = typer.Option(False, "--debug", help="输出与LLM交互的详细日志")
):
    """启动常驻交互式界面，持续接收文本并输出标签分析结果。"""
    cfg = load_config()
    setup_proxy(cfg["proxy"])  # 设置环境代理
    
    # 初始化向量数据库（关键词 与 领域）
    print("正在初始化关键词与领域向量数据库...")
    try:
        kw_client, keyword_collection = init_keyword_vector_database()
        fd_client, field_collection = init_field_vector_database()
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

    # 构建用于新增领域/关键词联合判重的模型
    try:
        add_check_prompt = load_add_check_prompt()
        add_judge_model = build_model(cfg["api_key"], cfg["model"], add_check_prompt)
    except Exception as e:
        print(f"加载合并判重prompt失败: {e}")
        add_judge_model = None
    try:
        if 'debug' in locals() and debug and add_judge_model is None:
            print("[DEBUG] 合并判重模型未就绪，将跳过 LLM 判重，仅执行向量近邻检查。")
    except Exception:
        pass
    print("\n系统已就绪，开始标签分析模式...")

    def on_submit(text: str):
        try:
            # 预处理：从原始文本提取一级标题与来源链接
            def extract_title_and_source(raw_text: str):
                title_val = None
                source_val = None
                for line in raw_text.splitlines():
                    # 一级标题：以单个 # 开头，且不是 ##/###。允许前导空白。
                    m_title = re.match(r"^\s*#(?!#)\s+(.+)$", line)
                    if m_title and not title_val:
                        title_val = m_title.group(1).strip()
                    # 来源：以 source: 开头（大小写不敏感），允许前导空白
                    m_src = re.match(r"^\s*source:\s*(.+)$", line, flags=re.IGNORECASE)
                    if m_src and not source_val:
                        source_val = m_src.group(1).strip()
                    if title_val and source_val:
                        break
                return title_val, source_val

            title_extracted, source_url = extract_title_and_source(text)
            if debug:
                print("[DEBUG] 提取到的标题:", title_extracted)
                print("[DEBUG] 提取到的来源URL:", source_url)

            # 步骤1: 生成候选（类型/领域/关键词）
            print("正在生成候选类型/领域/关键词...")
            candidate_types = load_current_types()
            candidate_fields = get_similar_fields(text, field_collection, n_results=3)
            candidate_keywords = get_similar_keywords(text, keyword_collection, n_results=10)
            
            if not candidate_keywords and not candidate_fields and not candidate_types:
                return {"text": "错误：未能生成任何候选（类型/领域/关键词）", "usage": None}
            
            # 步骤2: 使用Gemini进行综合分析（类型/领域/关键词）
            print("正在进行综合分析（类型/领域/关键词）...")
            if debug:
                print("[DEBUG] 候选 types:", candidate_types)
                print("[DEBUG] 候选 fields:", candidate_fields)
                print("[DEBUG] 候选 keywords:", candidate_keywords)
            analysis_result = analyze_text_with_candidates(
                model, text,
                candidate_types=candidate_types,
                candidate_fields=candidate_fields,
                candidate_keywords=candidate_keywords,
                temperature=temperature, debug=debug
            )
            if debug:
                usage_a = analysis_result.get("usage") or {}
                if usage_a:
                    print(
                        f"[DEBUG] 分析 tokens: input={usage_a.get('input', 0)} prompt={usage_a.get('prompt', 0)} "
                        f"output={usage_a.get('output', 0)} total={usage_a.get('total', 0)}"
                    )
            
            # 步骤3: 提取并判重补充（关键词/领域），必要时更新向量数据库
            result_obj = analysis_result.get("result", {}) or {}
            # 若未从原文提取到 title，则尝试使用 LLM 的备选标题
            if not title_extracted:
                suggested_title = (result_obj or {}).get("suggested_title")
                if isinstance(suggested_title, str) and suggested_title.strip():
                    title_extracted = suggested_title.strip()
                    if debug:
                        print("[DEBUG] 使用 LLM 备选标题:", title_extracted)
            # 兼容新旧schema：优先从 classification_details/keyword_details 获取
            classification_details = result_obj.get("classification_details", {}) or {}
            keyword_details = result_obj.get("keyword_details", {}) or {}

            matched_types = list(classification_details.get("matched_types", result_obj.get("matched_types", [])) or [])
            supplementary_types = list(classification_details.get("supplementary_types", result_obj.get("supplementary_types", [])) or [])
            matched_fields = list(classification_details.get("matched_fields", result_obj.get("matched_fields", [])) or [])
            supplementary_fields = list(classification_details.get("supplementary_fields", result_obj.get("supplementary_fields", [])) or [])
            matched_keywords = list(keyword_details.get("matched_keywords", result_obj.get("matched_keywords", [])) or [])
            supplementary_keywords = list(keyword_details.get("supplementary_keywords", result_obj.get("supplementary_keywords", [])) or [])

            # 准备关键词与领域的近邻与复核输入
            if debug:
                print("[DEBUG] 初始 matched_types:", matched_types)
                print("[DEBUG] 初始 supplementary_types:", supplementary_types)
                print("[DEBUG] 初始 matched_fields:", matched_fields)
                print("[DEBUG] 初始 supplementary_fields:", supplementary_fields)
                print("[DEBUG] 初始 matched_keywords:", matched_keywords)
                print("[DEBUG] 初始 supplementary_keywords:", supplementary_keywords)

            kw_items_for_llm = []
            kw_moved_to_matched = set()
            for supp in supplementary_keywords:
                try:
                    existing_similars = get_similar_keywords(supp, keyword_collection, n_results=8)
                except Exception:
                    existing_similars = []
                if debug:
                    print(f"[DEBUG] 补充关键词 '{supp}' 的近邻相似项:", existing_similars)
                if any(s.strip() == supp.strip() for s in (existing_similars or [])):
                    kw_moved_to_matched.add(supp)
                else:
                    kw_items_for_llm.append({
                        "supplementary_keyword": supp,
                        "existing_similar_keywords": existing_similars or []
                    })

            fld_items_for_llm = []
            fld_moved_to_matched = set()
            for supp in supplementary_fields:
                try:
                    existing_similars = get_similar_fields(supp, field_collection, n_results=5)
                except Exception:
                    existing_similars = []
                if debug:
                    print(f"[DEBUG] 补充领域 '{supp}' 的近邻相似项:", existing_similars)
                if any(s.strip() == supp.strip() for s in (existing_similars or [])):
                    fld_moved_to_matched.add(supp)
                else:
                    fld_items_for_llm.append({
                        "supplementary_field": supp,
                        "existing_similar_fields": existing_similars or []
                    })

            # 去重更新 matched
            if kw_moved_to_matched:
                matched_keywords = sorted(set(matched_keywords).union(kw_moved_to_matched))
                if debug:
                    print("[DEBUG] 因近邻重名而直接加入 matched 的关键词:", sorted(kw_moved_to_matched))
            if fld_moved_to_matched:
                matched_fields = sorted(set(matched_fields).union(fld_moved_to_matched))
                if debug:
                    print("[DEBUG] 因近邻重名而直接加入 matched 的领域:", sorted(fld_moved_to_matched))

            accepted_new_keywords = []
            accepted_new_fields = []

            # 合并调用：领域与关键词联合判重
            if add_judge_model is not None and (kw_items_for_llm or fld_items_for_llm):
                print("正在进行新增领域/关键词联合判重…")
                try:
                    judge_res = adjudicate_fields_and_keywords(
                        add_judge_model,
                        document_content=text,
                        matched_fields=matched_fields,
                        supplementary_fields_items=fld_items_for_llm,
                        matched_keywords=matched_keywords,
                        supplementary_keywords_items=kw_items_for_llm,
                        temperature=0.0,
                        debug=debug,
                    )
                    fld_judgements = judge_res.get("field_judgements", [])
                    kw_judgements = judge_res.get("keyword_judgements", [])
                    if debug:
                        usage_j = judge_res.get("usage") or {}
                        if usage_j:
                            print(
                                f"[DEBUG] 合并判重 tokens: input={usage_j.get('input', 0)} prompt={usage_j.get('prompt', 0)} "
                                f"output={usage_j.get('output', 0)} total={usage_j.get('total', 0)}"
                            )
                except Exception as e:
                    print(f"联合判重失败，跳过本次新增：{e}")
                    fld_judgements, kw_judgements = [], []

                for j in kw_judgements:
                    judged_tag = j.get("judged_keyword") or j.get("judged_tag")
                    decision = j.get("decision")
                    final_tag = j.get("final_keyword") or j.get("final_tag")
                    if not judged_tag or not final_tag:
                        continue
                    if debug:
                        print("[DEBUG] 关键词判重结果项:", j)
                    if decision == "ACCEPT_NEW" and final_tag.strip() == judged_tag.strip():
                        accepted_new_keywords.append(judged_tag)
                    else:
                        if final_tag and final_tag not in matched_keywords:
                            matched_keywords.append(final_tag)

                for j in fld_judgements:
                    judged_field = j.get("judged_field") or j.get("judged_tag")
                    decision = j.get("decision")
                    final_field = j.get("final_field") or j.get("final_tag")
                    if not judged_field or not final_field:
                        continue
                    if debug:
                        print("[DEBUG] 领域判重结果项:", j)
                    if decision == "ACCEPT_NEW" and final_field.strip() == judged_field.strip():
                        accepted_new_fields.append(judged_field)
                    else:
                        if final_field and final_field not in matched_fields:
                            matched_fields.append(final_field)

            # 根据移动与裁决结果，刷新集合并写回结果
            final_matched_keywords = sorted(set(matched_keywords))
            final_supplementary_keywords = sorted(set(accepted_new_keywords))
            final_matched_fields = sorted(set(matched_fields))
            final_supplementary_fields = sorted(set(accepted_new_fields))
            final_matched_types = sorted(set(matched_types))
            final_supplementary_types = sorted(set(supplementary_types))

            analysis_result.setdefault("result", {})
            analysis_result["result"]["matched_keywords"] = final_matched_keywords
            analysis_result["result"]["supplementary_keywords"] = final_supplementary_keywords
            analysis_result["result"]["matched_fields"] = final_matched_fields
            analysis_result["result"]["supplementary_fields"] = final_supplementary_fields
            analysis_result["result"]["matched_types"] = final_matched_types
            analysis_result["result"]["supplementary_types"] = final_supplementary_types

            if debug:
                print("[DEBUG] 最终 keywords:", final_matched_keywords, "+", final_supplementary_keywords)
                print("[DEBUG] 最终 fields:", final_matched_fields, "+", final_supplementary_fields)
                print("[DEBUG] 最终 types:", final_matched_types, "+", final_supplementary_types)

            # 仅将被接受的新增写入向量库
            if final_supplementary_keywords:
                print(f"正在更新关键词向量数据库，添加 {len(final_supplementary_keywords)} 个新关键词...")
                if debug:
                    print("[DEBUG] 即将写入关键词向量库:", final_supplementary_keywords)
                update_keyword_vector_database(keyword_collection, final_supplementary_keywords)

            if final_supplementary_fields:
                print(f"正在更新领域向量数据库，添加 {len(final_supplementary_fields)} 个新领域...")
                if debug:
                    print("[DEBUG] 即将写入领域向量库:", final_supplementary_fields)
                update_field_vector_database(field_collection, final_supplementary_fields)

            # 步骤4: 保存 Markdown 文件（在复核后，使用最终标签集合）
            try:
                note_dir = (cfg.get("text_note_dir") or "").strip()
                if note_dir and title_extracted:
                    # 生成唯一 id：YYYYMMDD + '_' + sha256(title + 第一个segment_summary) 的后8位（大写）
                    # 获取分段摘要，并取第一个段落的 summary 作为参与哈希的内容
                    seg_summaries = (result_obj.get("segmented_summaries") or [])
                    first_seg_sum = ""
                    if isinstance(seg_summaries, list) and seg_summaries:
                        first_item = seg_summaries[0]
                        if isinstance(first_item, dict):
                            maybe_sum = first_item.get("segment_summary")
                            if isinstance(maybe_sum, str):
                                first_seg_sum = maybe_sum
                        elif isinstance(first_item, str):
                            first_seg_sum = first_item
                    date_str = datetime.now().strftime("%Y%m%d")
                    concat_text = f"{title_extracted}{first_seg_sum}"
                    tail = hashlib.sha256(concat_text.encode("utf-8")).hexdigest()[-8:].upper()
                    uid = f"{date_str}_{tail}"

                    # 汇总所有标签（关键词 + 领域 + 类型），并将标签内部空格替换为 '-'
                    all_tags = sorted(set(
                        final_matched_keywords + final_supplementary_keywords +
                        final_matched_fields + final_supplementary_fields +
                        final_matched_types + final_supplementary_types
                    ))
                    processed_tags = []
                    for _t in all_tags:
                        if isinstance(_t, str):
                            _norm = re.sub(r"\s+", "-", _t.strip())
                            if _norm:
                                processed_tags.append(_norm)

                    # 组装正文：Summary 与分段摘要（带标题与项目符号）
                    overall_summary = (result_obj.get("overall_summary") or "")
                    if not isinstance(overall_summary, str):
                        overall_summary = str(overall_summary)
                    body_lines = []
                    body_lines.append("## Summary")
                    body_lines.append(overall_summary.strip())
                    body_lines.append("## Segment summaries")
                    for seg in seg_summaries:
                        seg_sum_val = None
                        if isinstance(seg, dict):
                            seg_sum_val = seg.get("segment_summary")
                        elif isinstance(seg, str):
                            seg_sum_val = seg
                        if isinstance(seg_sum_val, str) and seg_sum_val.strip():
                            body_lines.append(f" - {seg_sum_val.strip()}")

                    # 构建 Markdown 文本（在 front-matter 中加入 fields 列表，格式与 tags 相同）
                    front_matter = [
                        "---",
                        f"id: {uid}",
                    ]
                    # fields 列表（使用最终 matched + supplementary 的去重合集）
                    all_fields = sorted(set(final_matched_fields + final_supplementary_fields))
                    processed_fields = []
                    for _f in all_fields:
                        if isinstance(_f, str):
                            _normf = re.sub(r"\s+", "-", _f.strip())
                            if _normf:
                                processed_fields.append(_normf)
                    if processed_fields:
                        front_matter.append("fields:")
                        front_matter.extend([f"  - {f}" for f in processed_fields])
                    else:
                        front_matter.append("fields: []")
                    if processed_tags:
                        front_matter.append("tags:")
                        front_matter.extend([f"  - {t}" for t in processed_tags])
                    else:
                        front_matter.append("tags: []")
                    front_matter.extend([
                        f"url: {source_url or ''}",
                        "---",
                    ])
                    md_text = "\n".join(front_matter + body_lines) + "\n"

                    # 规范化文件名并保存
                    def sanitize_filename(name: str) -> str:
                        # 移除 Windows 不允许的字符
                        name = re.sub(r"[\\/:*?\"<>|]", "", name)
                        # 去掉首尾空格和点
                        name = name.strip().strip('.')
                        return name or uid

                    file_name = sanitize_filename(title_extracted) + ".md"
                    save_dir = Path(note_dir)
                    save_dir.mkdir(parents=True, exist_ok=True)
                    save_path = save_dir / file_name
                    save_path.write_text(md_text, encoding="utf-8")
                    if debug:
                        print(f"[DEBUG] Markdown 已保存: {save_path}")
                else:
                    if debug:
                        print("[DEBUG] 未保存 Markdown：text_note_dir 未配置或标题为空。")
            except Exception as e:
                print(f"保存 Markdown 失败: {e}")
            
            # 构造更完整的结果用于 UI 展示
            # 按最新 schema 组织返回结构
            display_payload = {
                "result": {
                    "overall_summary": result_obj.get("overall_summary"),
                    "suggested_title": result_obj.get("suggested_title"),
                    "classification_details": {
                        "matched_types": final_matched_types,
                        "supplementary_types": final_supplementary_types,
                        "matched_fields": final_matched_fields,
                        "supplementary_fields": final_supplementary_fields,
                    },
                    "keyword_details": {
                        "matched_keywords": final_matched_keywords,
                        "supplementary_keywords": final_supplementary_keywords,
                    },
                    "segmented_summaries": result_obj.get("segmented_summaries", []),
                },
                "raw_output": analysis_result.get("raw_output"),
                "usage": analysis_result.get("usage"),
            }
            
            return {
                "analysis_result": display_payload,
                "similar_tags": candidate_keywords,
                "usage": analysis_result.get("usage"),
            }
            
        except Exception as e:
            return {
                "text": f"分析过程中发生错误: {str(e)}",
                "usage": None
            }

    interactive_loop(on_submit)


if __name__ == "__main__":
    app()


