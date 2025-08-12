from typing import Optional


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


def summarize_text(model, text: str, temperature: float = 0.3) -> str:
    """调用 Gemini 对文本进行概括。"""
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

    return output.strip()


