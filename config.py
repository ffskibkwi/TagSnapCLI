import os
import textwrap
from pathlib import Path
import configparser
import textwrap


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
    '''读取 prompt.ini 中的多行提示词，兼容包含换行的长文本。

    支持以下写法：
    - INI 多行值（在 prompt = 之后的行缩进即可视为同一值）
        prompt =
            第一行
            第二行
    - 三引号包裹（单行/多行均可，示例为双引号三引号）：
        prompt = """
        第一行
        第二行
        """
      （也可使用单引号三引号，不在此处直接展示以避免文档转义冲突）
    - 如果没有 [summarize] 或没有 prompt 键，则回退为将整个文件作为提示词原文。
    '''
    if not PROMPT_FILE.exists():
        raise FileNotFoundError(
            f"未找到 {PROMPT_FILE.name}，请先运行: python main.py init 或根据 README 创建提示文件"
        )

    # 用 RawConfigParser 保持原始字符串（不做 % 插值），以更好地兼容长文本
    parser = configparser.RawConfigParser()
    try:
        parser.read(PROMPT_FILE, encoding="utf-8")
    except Exception:
        parser = None

    prompt_text: str = ""

    if parser and parser.has_section("summarize") and parser.has_option("summarize", "prompt"):
        prompt_text = parser.get("summarize", "prompt", raw=True, fallback="")
    else:
        # 回退：将整个文件视为提示词原文
        prompt_text = PROMPT_FILE.read_text(encoding="utf-8")

    # 处理三引号包裹
    stripped = prompt_text.strip()
    if (stripped.startswith('"""') and stripped.endswith('"""')) or (
        stripped.startswith("'''") and stripped.endswith("'''")
    ):
        prompt_text = stripped[3:-3]

    # 统一换行并去除最外层公共缩进
    prompt_text = prompt_text.replace("\r\n", "\n").replace("\r", "\n")
    prompt_text = textwrap.dedent(prompt_text).strip("\n")

    if not prompt_text:
        # 提供一个合理默认值
        prompt_text = (
            "你是一个高效的中文文本总结助手。用简洁、客观、分点的方式概括给定文本的关键信息。"
            "禁止夸大，不编造事实，不加入主观评价。"
        )
    return prompt_text


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
        # 用于文本概括的提示词（支持多行，缩进行即可）：
        prompt =
            你是一个高效的中文文本总结助手。请对用户提供的长文本进行准确、客观的概括：
            - 使用1-5个要点分条总结；
            - 保留关键数据、结论与限制条件；
            - 不编造事实，不加入主观评价；
            - 输出只包含总结内容，不要前后缀说明。
        """
    ).strip()

    for path, content in [(CONFIG_FILE, sample_config), (PROMPT_FILE, sample_prompt)]:
        if path.exists() and not force:
            print(f"跳过：{path.name} 已存在。如需覆盖，请加 --force")
            continue
        path.write_text(content + "\n", encoding="utf-8")
        print(f"已生成 {path.name}")


