import os
import sys
import textwrap
from pathlib import Path
import configparser
from typing import Optional

import typer

APP_NAME = "TagSnapCLI"
CONFIG_FILE = Path.cwd() / "config.ini"
PROMPT_FILE = Path.cwd() / "prompt.ini"

app = typer.Typer(help=f"{APP_NAME} - 基于 Typer 的 LLM Agent CLI")


def load_config() -> dict:
    """读取本地 config.ini 并返回配置字典。"""
    if not CONFIG_FILE.exists():
        raise FileNotFoundError(
            f"未找到 {CONFIG_FILE.name}，请先运行: python cli.py init 或根据 README 创建配置文件"
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

    return {"api_key": api_key, "model": model_name, "proxy": proxy_config}


def setup_proxy(proxy_config: dict) -> None:
    """设置代理。"""
    # 按用户提供的参考实现
    if proxy_config.get("http"):
        os.environ["http_proxy"] = proxy_config["http"]
    if proxy_config.get("https"):
        os.environ["https_proxy"] = proxy_config["https"]


def load_prompt() -> str:
    """读取 prompt.ini 中的总结提示语。"""
    if not PROMPT_FILE.exists():
        raise FileNotFoundError(
            f"未找到 {PROMPT_FILE.name}，请先运行: python cli.py init 或根据 README 创建提示文件"
        )

    parser = configparser.ConfigParser()
    parser.read(PROMPT_FILE, encoding="utf-8")

    if not parser.has_section("summarize"):
        raise KeyError("prompt.ini 缺少 [summarize] 配置段")

    prompt = parser.get("summarize", "prompt", fallback="")
    if not prompt:
        # 提供一个合理的默认值
        prompt = (
            "你是一个高效的中文文本总结助手。用简洁、客观、分点的方式概括给定文本的关键信息。"
            "禁止夸大，不编造事实，不加入主观评价。"
        )
    return prompt


def build_model(api_key: str, model_name: str, system_instruction: str):
    """构建并返回 Gemini 模型实例。"""
    try:
        import google.generativeai as genai
    except Exception as exc:  # pragma: no cover - 运行时依赖导入错误
        raise RuntimeError(
            "未安装 google-generativeai，请先运行: pip install google-generativeai"
        ) from exc

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(
        model_name=model_name,
        system_instruction=system_instruction,
    )
    return model


def read_input_text(text: Optional[str], file: Optional[Path]) -> str:
    """根据参数或标准输入读取用户长文本内容。"""
    if text:
        return text
    if file:
        return Path(file).read_text(encoding="utf-8")

    # 尝试从 stdin 读取
    if not sys.stdin.isatty():
        data = sys.stdin.read()
        if data.strip():
            return data

    # 交互式提示（可在 PowerShell 中粘贴后 Ctrl+Z 回车 结束）
    typer.echo(
        "未提供 --text 或 --file，将从标准输入读取。粘贴文本后，Windows 按 Ctrl+Z 然后回车提交。\n"
    )
    try:
        data = sys.stdin.read()
    except KeyboardInterrupt:
        raise typer.Abort()
    if not data.strip():
        raise typer.BadParameter("未获取到任何文本输入")
    return data


@app.command()
def init(force: bool = typer.Option(False, "--force", help="如存在则覆盖生成")):
    """初始化示例 config.ini 与 prompt.ini。"""
    sample_config = textwrap.dedent(
        """
        [gemini]
        # 必填：你的 Google Generative AI API Key
        api_key = YOUR_API_KEY_HERE
        # 可选：模型名称，常用：gemini-1.5-flash 或 gemini-1.5-pro
        model = gemini-1.5-flash

        [proxy]
        # 可选：HTTP/HTTPS 代理。示例：http://127.0.0.1:7890
        http =
        https =
        """
    ).strip()

    sample_prompt = textwrap.dedent(
        """
        [summarize]
        # 用于文本概括的提示词
        prompt = 你是一个高效的中文文本总结助手。请对用户提供的长文本进行准确、客观的概括：\
                 - 使用1-5个要点分条总结；\
                 - 保留关键数据、结论与限制条件；\
                 - 不编造事实，不加入主观评价；\
                 - 输出只包含总结内容，不要前后缀说明。
        """
    ).strip()

    for path, content in [(CONFIG_FILE, sample_config), (PROMPT_FILE, sample_prompt)]:
        if path.exists() and not force:
            typer.echo(f"跳过：{path.name} 已存在。如需覆盖，请加 --force")
            continue
        path.write_text(content + "\n", encoding="utf-8")
        typer.echo(f"已生成 {path.name}")


@app.command()
def summarize(
    text: Optional[str] = typer.Option(None, "--text", "-t", help="直接传入要概括的长文本"),
    file: Optional[Path] = typer.Option(
        None, "--file", "-f", exists=True, file_okay=True, dir_okay=False, readable=True, help="包含长文本的文件路径"
    ),
    temperature: float = typer.Option(0.3, help="生成温度(0-1)"),
):
    """对输入的长文本进行概括，使用本地 prompt.ini 中的提示词及 config.ini 中的 Gemini 设置。"""

    cfg = load_config()
    setup_proxy(cfg["proxy"])  # 环境变量级别生效
    prompt = load_prompt()

    model = build_model(cfg["api_key"], cfg["model"], prompt)

    user_text = read_input_text(text, file)

    try:
        response = model.generate_content(
            user_text,
            generation_config={
                "temperature": max(0.0, min(1.0, float(temperature))),
            },
        )
    except Exception as exc:
        raise RuntimeError(f"调用 Gemini 接口失败：{exc}") from exc

    output = getattr(response, "text", None)
    if not output:
        # 某些情况下需要从 candidates 里取
        try:
            candidates = response.candidates or []
            if candidates and candidates[0].content and candidates[0].content.parts:
                output = "".join(part.text for part in candidates[0].content.parts if hasattr(part, "text"))
        except Exception:
            output = None

    if not output:
        raise RuntimeError("未从模型获取到有效输出")

    typer.echo(output.strip())


if __name__ == "__main__":
    app()


