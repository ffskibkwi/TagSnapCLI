from typing import Callable, Dict, Any
import sys
import json

import typer
from rich.console import Console
from rich.panel import Panel
from rich.box import ROUNDED
from rich.text import Text
from rich.live import Live
from rich.spinner import Spinner
from rich.table import Table
from rich.columns import Columns
from rich.syntax import Syntax

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

# pyperclip 用于剪贴板操作
try:
    import pyperclip
    PYPERCLIP_AVAILABLE = True
except ImportError:
    PYPERCLIP_AVAILABLE = False


console = Console()


def format_tag_analysis_result(result: Dict[str, Any]) -> Text:
    """
    格式化标签分析结果为可读的显示内容。
    
    Args:
        result (Dict[str, Any]): 分析结果
        
    Returns:
        Text: 格式化后的显示内容
    """
    output = Text()
    
    if "analysis_result" in result:
        analysis = result["analysis_result"]
        similar_tags = result.get("similar_tags", [])
        
        # 显示候选关键词（来自相似检索）
        if similar_tags:
            output.append("⚡ 候选关键词(Top10):\n", style="bold blue")
            output.append(", ".join(similar_tags[:10]))
            output.append("\n\n")
        
        parsed_result = analysis.get("result", {})
        
        # 显示总体摘要
        overall_summary = parsed_result.get("overall_summary", "")
        if overall_summary:
            output.append("📋 总体摘要:\n", style="bold blue")
            output.append(overall_summary)
            output.append("\n\n")
        
        # 优先显示新结构：types/fields/keywords
        notes = ""
        classification_details = parsed_result.get("classification_details", {}) or {}
        keyword_details = parsed_result.get("keyword_details", {}) or {}

        matched_types = classification_details.get("matched_types", parsed_result.get("matched_types", [])) or []
        supplementary_types = classification_details.get("supplementary_types", parsed_result.get("supplementary_types", [])) or []
        matched_fields = classification_details.get("matched_fields", parsed_result.get("matched_fields", [])) or []
        supplementary_fields = classification_details.get("supplementary_fields", parsed_result.get("supplementary_fields", [])) or []
        matched_keywords = keyword_details.get("matched_keywords", parsed_result.get("matched_keywords", [])) or []
        supplementary_keywords = keyword_details.get("supplementary_keywords", parsed_result.get("supplementary_keywords", [])) or []

        if matched_types or supplementary_types or matched_fields or supplementary_fields or matched_keywords or supplementary_keywords:
            # Types
            output.append("✅ 匹配的类型(Types):\n", style="bold green")
            if matched_types:
                for t in matched_types:
                    output.append(f"  • {t}\n")
            else:
                output.append("  (none)\n")
            if supplementary_types:
                output.append("\n✨ 补充类型:\n", style="bold magenta")
                for t in supplementary_types:
                    output.append(f"  • {t}\n")

            # Fields
            output.append("\n✅ 匹配的领域(Fields):\n", style="bold green")
            if matched_fields:
                for f in matched_fields:
                    output.append(f"  • {f}\n")
            else:
                output.append("  (none)\n")
            if supplementary_fields:
                output.append("\n✨ 补充领域:\n", style="bold magenta")
                for f in supplementary_fields:
                    output.append(f"  • {f}\n")

            # Keywords
            output.append("\n✅ 匹配的关键词(Keywords):\n", style="bold green")
            if matched_keywords:
                for k in matched_keywords:
                    output.append(f"  • {k}\n")
            else:
                output.append("  (none)\n")
            if supplementary_keywords:
                output.append("\n✨ 补充关键词:\n", style="bold magenta")
                for k in supplementary_keywords:
                    output.append(f"  • {k}\n")
        else:
            # 兼容旧结构：tagging_details
            tagging_details = parsed_result.get("tagging_details", {})
            matched_tags = tagging_details.get("matched_tags", [])
            if matched_tags:
                output.append("✅ 匹配的标签:\n", style="bold green")
                for tag in matched_tags:
                    output.append(f"  • {tag}\n")
            else:
                output.append("❌ 未找到匹配的标签\n", style="yellow")
            supplementary_tags = tagging_details.get("supplementary_tags", [])
            if supplementary_tags:
                output.append("\n✨ 补充标签:\n", style="bold magenta")
                for tag in supplementary_tags:
                    output.append(f"  • {tag}\n")
            notes = tagging_details.get("tagging_notes", "")
            if notes:
                output.append("\n📝 分析说明:\n", style="bold cyan")
                output.append(notes)
                output.append("\n")

        # 额外备注（新结构下可能存在 extraction_notes）
        extraction_notes = parsed_result.get("extraction_notes", "")
        if extraction_notes:
            output.append("\n📝 说明:\n", style="bold cyan")
            output.append(extraction_notes)
            output.append("\n")
        
        # 显示分段摘要
        segmented_summaries = parsed_result.get("segmented_summaries", [])
        if segmented_summaries:
            output.append("\n📑 分段摘要:\n", style="bold yellow")
            for i, segment in enumerate(segmented_summaries, 1):
                if isinstance(segment, dict):
                    segment_summary = segment.get("segment_summary", "")
                else:
                    segment_summary = str(segment)
                if segment_summary:
                    output.append(f"  {i}. {segment_summary}\n")
        
        # 如果有JSON解析错误，显示原始输出（兼容新旧结构）
        if isinstance(notes, str) and "解析失败" in notes:
            raw_output = analysis.get("raw_output", "")
            if raw_output:
                output.append("\n🔍 原始输出:\n", style="dim")
                output.append(raw_output[:500])  # 限制长度
                if len(raw_output) > 500:
                    output.append("…", style="dim")
    
    elif "text" in result:
        # 错误情况或原有的简单格式
        output.append(result["text"])
    
    else:
        output.append("未知的结果格式")
    
    return output


def interactive_loop(
    on_submit: Callable[[str], Dict[str, Any]],
) -> None:
    """常驻交互式循环。接收用户输入并调用回调处理，直到收到退出指令。"""
    console.print(
        Panel(
            "欢迎使用 TagSnapCLI 标签分析助手\n"
            "- Enter 换行，Ctrl+Enter 提交\n"
            "- Ctrl+P 从剪贴板粘贴并提交\n"
            "- Ctrl+Q 或输入 \\q 退出",
            title="TagSnapCLI - AI 标签分析",
            box=ROUNDED,
            border_style="cyan",
        )
    )

    def read_inside_rounded_box() -> str:
        if PROMPT_TOOLKIT_AVAILABLE:
            textarea = TextArea(
                multiline=True,
                wrap_lines=True,
                focus_on_click=True,
            )

            frame = Frame(body=textarea, title="", style="class:input-frame")

            kb = KeyBindings()

            # 使用 Ctrl+Enter 提交，保留 Enter 作为换行
            @kb.add("c-m")
            def _(event):
                event.app.exit(result=textarea.text)

            @kb.add("c-q")
            def _(event):
                event.app.exit(result="\\q")

            @kb.add("c-p")
            def _(event):
                if PYPERCLIP_AVAILABLE:
                    try:
                        clipboard_content = pyperclip.paste()
                        if clipboard_content and clipboard_content.strip():
                            event.app.exit(result=clipboard_content)
                        else:
                            # 剪贴板为空，不执行任何操作
                            pass
                    except Exception:
                        # 剪贴板访问失败，不执行任何操作
                        pass
                else:
                    # pyperclip 未安装，不执行任何操作
                    pass

            style = Style.from_dict({
                "frame.border": "ansiwhite",
                "input-frame": "",
            })

            app = Application(
                layout=Layout(frame),
                key_bindings=kb,
                full_screen=False,
                mouse_support=False,
                style=style,
            )
            return app.run()

        # 回退方案
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
        except (typer.Abort, EOFError):
            console.print("已取消。", style="yellow")
            break

        if user_text.strip() in {"\\q", "\\quit"}:
            console.print("已退出。", style="green")
            break

        if not user_text.strip():
            console.print("请输入非空文本。", style="yellow")
            continue

        with Live(Spinner("dots", text="正在分析标签…", style="cyan"), console=console, refresh_per_second=12):
            try:
                result = on_submit(user_text)
            except Exception as exc:
                result = {"text": f"错误：{exc}", "usage": None}

        if isinstance(result, dict):
            result_panel_text = format_tag_analysis_result(result)
            usage = result.get("usage")
            
            if usage and usage.get("total"):
                tokens_line = (
                    f"\n[dim]tokens: input={usage.get('input', 0)} output={usage.get('output', 0)} total={usage.get('total', 0)}[/dim]"
                )
                result_panel_text.append(tokens_line)
        else:
            result_panel_text = Text(str(result))

        console.print(Panel(result_panel_text, title="🏷️ 标签分析结果", box=ROUNDED, border_style="green"))


def display_tag_analysis_table(analysis_result: Dict[str, Any], similar_tags: list) -> Table:
    """
    创建一个表格显示所有标签信息。
    
    Args:
        analysis_result (Dict[str, Any]): 分析结果
        similar_tags (list): 相似标签
        
    Returns:
        Table: 格式化的表格
    """
    table = Table(title="标签分析详情")
    
    table.add_column("类型", style="cyan", no_wrap=True)
    table.add_column("数量", style="magenta")
    table.add_column("标签列表", style="green")
    
    result = analysis_result.get("result", {})
    tagging_details = result.get("tagging_details", {})
    matched_tags = tagging_details.get("matched_tags", [])
    supplementary_tags = tagging_details.get("supplementary_tags", [])
    
    table.add_row(
        "相似标签",
        str(len(similar_tags)),
        ", ".join(similar_tags)
    )
    
    table.add_row(
        "匹配标签",
        str(len(matched_tags)),
        ", ".join(matched_tags) if matched_tags else "无"
    )
    
    table.add_row(
        "补充标签",
        str(len(supplementary_tags)),
        ", ".join(supplementary_tags) if supplementary_tags else "无"
    )
    
    return table
