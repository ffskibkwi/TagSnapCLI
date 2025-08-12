from typing import Callable

import typer
from rich.console import Console
from rich.panel import Panel
from rich.box import ROUNDED


console = Console()


def interactive_loop(
    on_submit: Callable[[str], str],
) -> None:
    """常驻交互式循环。接收用户输入并调用回调处理，直到收到退出指令。"""
    console.print(
        Panel(
            "欢迎使用 TagSnapCLI 概括助手\n"
            "- 粘贴/输入任意长度文本后回车提交\n"
            "- 输入 \\q 或 \\quit 退出\n",
            title="TagSnapCLI",
            box=ROUNDED,
            border_style="cyan",
        )
    )

    def read_inside_rounded_box() -> str:
        # 计算框宽度（留出两侧页边距）
        term_width = console.size.width
        inner_width = max(20, min(80, term_width - 6))
        top = f"[white]╭{'─' * inner_width}╮[/white]"
        bottom = f"[white]╰{'─' * inner_width}╯[/white]"

        # 顶部边框
        console.print(top)
        # 左边框 + 输入光标。此处将光标放在左墙后的空格处
        prompt_str = "[white]│ [/white]"
        # 使用 console.input 渲染富文本提示，从而使光标位于白色竖线之后
        text = console.input(prompt_str)
        # 底部边框
        console.print(bottom)
        return text

    while True:
        try:
            user_text = read_inside_rounded_box()
        except typer.Abort:
            console.print("已取消。", style="yellow")
            break

        if user_text.strip() in {"\\q", "\\quit"}:
            console.print("已退出。", style="green")
            break

        if not user_text.strip():
            console.print("请输入非空文本。", style="yellow")
            continue

        try:
            result = on_submit(user_text)
            console.print(Panel(result, title="概括结果", box=ROUNDED, border_style="green"))
        except Exception as exc:  # 展示错误但不中断循环
            console.print(Panel(str(exc), title="错误", box=ROUNDED, border_style="red"))


