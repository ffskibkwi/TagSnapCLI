import typer

from config import init_files, load_config, load_prompt, setup_proxy
from analysis import build_model, summarize_text
from interface import interactive_loop


app = typer.Typer(help="TagSnapCLI - 常驻交互式文本概括助手")


@app.command()
def init(force: bool = typer.Option(False, "--force", help="如存在则覆盖生成")):
    """生成示例 config.ini 与 prompt.ini。"""
    init_files(force=force)


@app.command()
def run(temperature: float = typer.Option(0.3, help="生成温度(0-1)")):
    """启动常驻交互式界面，持续接收文本并输出概括。"""
    cfg = load_config()
    setup_proxy(cfg["proxy"])  # 设置环境代理
    prompt = load_prompt()
    model = build_model(cfg["api_key"], cfg["model"], prompt)

    def on_submit(text: str) -> str:
        return summarize_text(model, text, temperature=temperature)

    interactive_loop(on_submit)


if __name__ == "__main__":
    app()


