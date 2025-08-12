import os
import textwrap
from pathlib import Path
import configparser


APP_NAME = "TagSnapCLI"
CONFIG_FILE = Path.cwd() / "config.ini"
PROMPT_FILE = Path.cwd() / "prompt.ini"


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

    return {"api_key": api_key, "model": model_name, "proxy": proxy_config}


def setup_proxy(proxy_config: dict) -> None:
    """设置代理。"""
    if proxy_config.get("http"):
        os.environ["http_proxy"] = proxy_config["http"]
    if proxy_config.get("https"):
        os.environ["https_proxy"] = proxy_config["https"]


def load_prompt() -> str:
    """读取 prompt.ini 中的总结提示语。"""
    if not PROMPT_FILE.exists():
        raise FileNotFoundError(
            f"未找到 {PROMPT_FILE.name}，请先运行: python main.py init 或根据 README 创建提示文件"
        )

    parser = configparser.ConfigParser()
    parser.read(PROMPT_FILE, encoding="utf-8")

    if not parser.has_section("summarize"):
        raise KeyError("prompt.ini 缺少 [summarize] 配置段")

    prompt = parser.get("summarize", "prompt", fallback="")
    if not prompt:
        prompt = (
            "你是一个高效的中文文本总结助手。用简洁、客观、分点的方式概括给定文本的关键信息。"
            "禁止夸大，不编造事实，不加入主观评价。"
        )
    return prompt


def init_files(force: bool = False) -> None:
    """生成示例 config.ini 与 prompt.ini。"""
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
            print(f"跳过：{path.name} 已存在。如需覆盖，请加 --force")
            continue
        path.write_text(content + "\n", encoding="utf-8")
        print(f"已生成 {path.name}")


