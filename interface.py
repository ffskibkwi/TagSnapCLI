from typing import Callable
import sys

import typer
from rich.console import Console
from rich.panel import Panel
from rich.box import ROUNDED
from rich.text import Text

# prompt_toolkit 用于自适应尺寸的输入框
try:
    from prompt_toolkit.application import Application
    from prompt_toolkit.key_binding import KeyBindings
    from prompt_toolkit.layout import Layout
    from prompt_toolkit.widgets import Frame, TextArea
    from prompt_toolkit.styles import Style
    PROMPT_TOOLKIT_AVAILABLE = True
except Exception:
    PROMPT_TOOLKIT_AVAILABLE = False


console = Console()


def interactive_loop(
    on_submit: Callable[[str], str],
) -> None:
    """常驻交互式循环。接收用户输入并调用回调处理，直到收到退出指令。"""
    console.print(
        Panel(
            "欢迎使用 TagSnapCLI 概括助手\n"
            "- Enter 换行，Ctrl+J 或 Ctrl+S 提交\n"
            "- 输入 \\q 或 \\quit 退出\n",
            title="TagSnapCLI",
            box=ROUNDED,
            border_style="cyan",
        )
    )

    def read_inside_rounded_box() -> str:
        if PROMPT_TOOLKIT_AVAILABLE:
            textarea = TextArea(
                multiline=True,  # 允许自动换行并随内容增高
                wrap_lines=True,
                focus_on_click=True,
            )

            frame = Frame(body=textarea, title="", style="class:input-frame")

            kb = KeyBindings()

            # 使用 Ctrl+J 或 Ctrl+S 提交，保留 Enter 作为换行
            @kb.add("c-j")
            def _(event):
                event.app.exit(result=textarea.text)

            @kb.add("c-s")
            def _(event):
                event.app.exit(result=textarea.text)

            style = Style.from_dict({
                # 边框颜色为白色（兼容旧版本 prompt_toolkit 的颜色名）
                "frame.border": "ansiwhite",
                "input-frame": "",
            })

            app = Application(
                layout=Layout(frame),
                key_bindings=kb,
                full_screen=False,  # 非全屏，嵌入在现有终端输出
                mouse_support=False,
                style=style,
            )
            return app.run()

        # 回退方案：使用固定框 + ANSI 移动（不支持自增高）
        term_width = console.size.width
        inner_width = max(20, min(80, term_width - 6))
        top = f"[white]╭{'─' * inner_width}╮[/white]"
        mid = f"[white]│{' ' * inner_width}│[/white]"
        bottom = f"[white]╰{'─' * inner_width}╯[/white]"
        console.print(top)
        console.print(mid)
        console.print(bottom)
        sys.stdout.write("\x1b[2A\r\x1b[1C")
        sys.stdout.flush()
        text = input()
        sys.stdout.write("\x1b[1B\r")
        sys.stdout.flush()
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


