import json
from typing import Optional, Dict, Any, List, Tuple


def build_model(api_key: str, model_name: str, system_instruction: str):
    """构建并返回 Gemini 模型实例。"""
    try:
        import google.generativeai as genai
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "未安装 google-generativeai，请先运行: pip install google-generativeai"
        ) from exc

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(
        model_name=model_name,
        system_instruction=system_instruction,
    )
    return model


def segment_text(model, text: str, temperature: float = 0.3) -> Dict[str, Any]:
    """调用 Gemini 对文本进行语义分割，返回结果与 token 用量信息。"""
    response = model.generate_content(
        text,
        generation_config={
            "temperature": max(0.0, min(1.0, float(temperature))),
        },
    )

    output: Optional[str] = getattr(response, "text", None)
    if not output:
        try:
            candidates = response.candidates or []
            if candidates and candidates[0].content and candidates[0].content.parts:
                output = "".join(
                    part.text for part in candidates[0].content.parts if hasattr(part, "text")
                )
        except Exception:
            output = None

    if not output:
        raise RuntimeError("未从模型获取到有效输出")

    usage_meta = getattr(response, "usage_metadata", None)
    usage = None
    if usage_meta is not None:
        # 兼容字段名
        prompt_tokens = getattr(usage_meta, "prompt_token_count", None)
        completion_tokens = getattr(usage_meta, "candidates_token_count", None)
        total_tokens = getattr(usage_meta, "total_token_count", None)
        usage = {
            "prompt": prompt_tokens,
            "completion": completion_tokens,
            "total": total_tokens,
        }

    return {"text": output.strip(), "usage": usage}


def analyze_text_with_tags(model, text: str, candidate_tags: List[str], temperature: float = 0.3) -> Dict[str, Any]:
    """
    使用标签分析文本，返回匹配的标签和补充标签。
    
    Args:
        model: Gemini 模型实例
        text (str): 要分析的文档内容
        candidate_tags (List[str]): 候选标签列表（应该有10个）
        temperature (float): 生成温度
        
    Returns:
        Dict[str, Any]: 包含分析结果和token使用信息的字典
    """
    # 构建符合tag_seg.ini prompt要求的输入格式
    input_data = {
        "document_content": text,
        "candidate_tags": candidate_tags
    }
    
    # 将输入数据转换为JSON字符串，作为用户输入传递给模型
    json_input = json.dumps(input_data, ensure_ascii=False, indent=2)
    
    response = model.generate_content(
        json_input,
        generation_config={
            "temperature": max(0.0, min(1.0, float(temperature))),
        },
    )

    output: Optional[str] = getattr(response, "text", None)
    if not output:
        try:
            candidates = response.candidates or []
            if candidates and candidates[0].content and candidates[0].content.parts:
                output = "".join(
                    part.text for part in candidates[0].content.parts if hasattr(part, "text")
                )
        except Exception:
            output = None

    if not output:
        raise RuntimeError("未从模型获取到有效输出")

    # 尝试解析JSON输出
    try:
        # 提取JSON部分（可能被其他文本包围）
        output_clean = output.strip()
        
        # 查找JSON开始和结束位置
        start_idx = output_clean.find('{')
        end_idx = output_clean.rfind('}') + 1
        
        if start_idx != -1 and end_idx > start_idx:
            json_str = output_clean[start_idx:end_idx]
            parsed_result = json.loads(json_str)
        else:
            # 如果没有找到完整的JSON，尝试直接解析
            parsed_result = json.loads(output_clean)
            
    except json.JSONDecodeError as e:
        # 如果JSON解析失败，返回原始文本
        parsed_result = {
            "overall_summary": f"JSON解析失败: {str(e)}",
            "tagging_details": {
                "matched_tags": [],
                "supplementary_tags": [],
                "tagging_notes": f"JSON解析失败: {str(e)}\n原始输出: {output}"
            },
            "segmented_summaries": []
        }

    usage_meta = getattr(response, "usage_metadata", None)
    usage = None
    if usage_meta is not None:
        # 兼容字段名
        prompt_tokens = getattr(usage_meta, "prompt_token_count", None)
        completion_tokens = getattr(usage_meta, "candidates_token_count", None)
        total_tokens = getattr(usage_meta, "total_token_count", None)
        usage = {
            "prompt": prompt_tokens,
            "completion": completion_tokens,
            "total": total_tokens,
        }

    return {
        "result": parsed_result,
        "raw_output": output.strip(),
        "usage": usage
    }


def extract_all_tags(analysis_result: Dict[str, Any]) -> List[str]:
    """
    从分析结果中提取所有标签（匹配的+补充的）。
    
    Args:
        analysis_result (Dict[str, Any]): analyze_text_with_tags的返回结果
        
    Returns:
        List[str]: 所有标签的列表
    """
    result = analysis_result.get("result", {})
    tagging_details = result.get("tagging_details", {})
    matched_tags = tagging_details.get("matched_tags", [])
    supplementary_tags = tagging_details.get("supplementary_tags", [])
    
    all_tags = []
    if isinstance(matched_tags, list):
        all_tags.extend(matched_tags)
    if isinstance(supplementary_tags, list):
        all_tags.extend(supplementary_tags)
    
    return all_tags


