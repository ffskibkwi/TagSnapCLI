import os
import textwrap
from pathlib import Path
import configparser
import textwrap


APP_NAME = "TagSnapCLI"
CONFIG_FILE = Path.cwd() / "config.ini"
PROMPTS_DIR = Path.cwd() / "prompts"
SEGMENTER_FILE = PROMPTS_DIR / "segmenter.ini"


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
    '''读取 prompts/segmenter.ini 中的多行提示词，兼容包含换行的长文本。

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
    - 优先从 [prompt] 段读取键 prompt；若无则尝试 [segmenter] 与 [summarize]；
    - 若仍未找到，则回退为将整个文件作为提示词原文。
    '''
    if not SEGMENTER_FILE.exists():
        raise FileNotFoundError(
            f"未找到 {SEGMENTER_FILE.as_posix()}，请先运行: python main.py init 或根据 README 创建提示文件"
        )

    # 用 RawConfigParser 保持原始字符串（不做 % 插值），以更好地兼容长文本
    parser = configparser.RawConfigParser()
    try:
        parser.read(SEGMENTER_FILE, encoding="utf-8")
    except Exception:
        parser = None

    prompt_text: str = ""

    if parser:
        section_candidates = ["prompt", "segmenter", "summarize"]
        for section in section_candidates:
            if parser.has_section(section) and parser.has_option(section, "prompt"):
                prompt_text = parser.get(section, "prompt", raw=True, fallback="")
                break
    else:
        # 回退：将整个文件视为提示词原文
        prompt_text = SEGMENTER_FILE.read_text(encoding="utf-8")

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
        [prompt]
        # 用于语义分割的提示词（支持多行，缩进行即可）：
        prompt =
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


