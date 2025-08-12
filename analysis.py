from typing import Optional, Dict, Any


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


def summarize_text(model, text: str, temperature: float = 0.3) -> Dict[str, Any]:
    """调用 Gemini 对文本进行概括，返回文本与 token 用量信息。"""
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


