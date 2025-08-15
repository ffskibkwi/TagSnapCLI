# -*- coding: utf-8 -*-
"""
本脚本是一个用于调用 Ollama 服务生成文本嵌入向量（Embeddings）的客户端演示程序。

功能：
1. 从指定的配置文件（默认为 config.ini）读取 Ollama 服务地址和模型名称。
2. 接收一个文本字符串作为输入。
3. 通过 HTTP POST 请求调用 Ollama 的 /api/embeddings 接口。
4. 兼容多种常见的 API 响应格式，从中提取出向量数据。
5. 打印向量的维度和前几个值作为预览。

依赖：
- 仅使用 Python 标准库，无需安装任何第三方包。

使用方法：
python ollama_demo.py --text "你要向量化的句子"
"""

import argparse
import configparser
import json
import sys
from pathlib import Path
from urllib.request import Request, urlopen
from urllib.error import URLError, HTTPError
from typing import List, Any


def load_embedding_config(config_path: Path):
	"""
	从指定的 .ini 配置文件中加载并返回 Ollama 服务的配置。

	Args:
		config_path (Path): 配置文件的路径对象。

	Returns:
		tuple[str, str]: 一个包含 (服务地址, 模型名称) 的元组。

	Raises:
		FileNotFoundError: 如果配置文件不存在。
		KeyError: 如果配置文件中缺少 [embedding] 配置段。
		ValueError: 如果 ip_addr 或 model 的值为空。
	"""
	# 初始化配置解析器
	parser = configparser.ConfigParser()
	# 检查配置文件是否存在，不存在则抛出异常
	if not config_path.exists():
		raise FileNotFoundError(f"未找到配置文件: {config_path}")
	
	# 读取配置文件内容
	parser.read(config_path, encoding="utf-8")
	
	# 检查是否存在必需的 [embedding] 配置段
	if not parser.has_section("embedding"):
		raise KeyError("config.ini 缺少 [embedding] 配置段")
	
	# 获取 ip_addr，如果未配置则使用默认值，并去除首尾空格
	ip_addr = parser.get("embedding", "ip_addr", fallback="http://127.0.0.1:11434").strip()
	# 获取 model，如果未配置则使用默认值，并去除首尾空格
	model = parser.get("embedding", "model", fallback="bge-m3").strip()
	
	# 确保 ip_addr 和 model 都有值
	if not ip_addr or not model:
		raise ValueError("[embedding] ip_addr 与 model 不能为空")
	
	return ip_addr, model


def _extract_embedding(resp_json: Any) -> List[float]:
	"""
	从 Ollama API 返回的 JSON 响应中稳健地提取嵌入向量。

	该函数设计用于兼容多种可能的 JSON 结构，以提高程序的兼容性。

	Args:
		resp_json (Any): 从 JSON 解析得到的 Python 对象（通常是字典或列表）。

	Returns:
		List[float]: 提取出的嵌入向量列表。

	Raises:
		RuntimeError: 如果无法从任何已知的结构中解析出向量。
	"""
	# 策略 1: 尝试解析 Ollama 原生格式 { "embedding": [...] } 或 { "embedding": [[...]] }
	if isinstance(resp_json, dict) and "embedding" in resp_json:
		emb = resp_json.get("embedding")
		if isinstance(emb, list):
			# 如果 embedding 是一个列表的列表 (例如 [[...]])，则取第一个元素
			return emb[0] if emb and isinstance(emb[0], list) else emb

	# 策略 2: 尝试解析一些变体格式 { "embeddings": [...] }
	if isinstance(resp_json, dict) and "embeddings" in resp_json:
		embs = resp_json.get("embeddings")
		if isinstance(embs, list):
			# 兼容 [{"embedding": [...]}] 结构
			if embs and isinstance(embs[0], dict) and "embedding" in embs[0]:
				return embs[0]["embedding"]
			# 兼容 [[floats]] 结构
			if embs and isinstance(embs[0], list):
				return embs[0]
			# 兼容 [floats] 结构
			return embs

	# 策略 3: 尝试解析 OpenAI 兼容格式 { "data": [{ "embedding": [...] }, ...] }
	if isinstance(resp_json, dict) and "data" in resp_json:
		data = resp_json.get("data")
		if isinstance(data, list) and data:
			first = data[0]
			if isinstance(first, dict) and "embedding" in first:
				return first["embedding"]

	# 策略 4: 兜底方案，如果返回的就是一个向量列表
	if isinstance(resp_json, list):
		# 兼容 [[...]] 和 [...] 两种情况
		return resp_json[0] if resp_json and isinstance(resp_json[0], list) else resp_json

	# 如果以上所有策略都失败，则抛出异常
	raise RuntimeError(f"无法解析返回结果: {resp_json}")


def call_ollama_embeddings(base_url: str, model: str, text: str, timeout: float = 60.0):
	"""
	调用 Ollama 的 /api/embeddings 接口来获取给定文本的嵌入向量。

	该函数会尝试使用 'prompt' 和 'input' 两种字段名发送请求，以兼容
	Ollama 原生 API 和一些遵循 OpenAI 格式的 API 实现。

	Args:
		base_url (str): Ollama 服务的基础 URL (例如 "http://127.0.0.1:11434")。
		model (str): 要使用的嵌入模型的名称 (例如 "bge-m3")。
		text (str): 需要生成向量的输入文本。
		timeout (float, optional): 请求超时时间（秒）。默认为 60.0。

	Returns:
		Any: Ollama API 返回的原始 JSON 响应（已解析为 Python 对象）。

	Raises:
		RuntimeError: 如果发生网络错误或 HTTP 错误。
	"""
	# 拼接成完整的 API 端点 URL，并确保 URL 末尾只有一个斜杠
	endpoint = base_url.rstrip("/") + "/api/embeddings"

	# 定义一个内部函数来执行 POST 请求，以减少代码重复
	def _post(body: dict) -> Any:
		# 将 Python 字典序列化为 UTF-8 编码的 JSON 字节串
		data = json.dumps(body).encode("utf-8")
		# 创建一个 Request 对象，包含 URL、数据、请求头和请求方法
		request = Request(
			endpoint,
			data=data,
			headers={"Content-Type": "application/json"},
			method="POST",
		)
		# 发送请求并打开一个上下文管理器来处理响应
		with urlopen(request, timeout=timeout) as response:
			# 读取响应内容，并以 UTF-8 解码
			resp_text = response.read().decode("utf-8", errors="replace")
			# 将响应的 JSON 字符串解析为 Python 对象
			return json.loads(resp_text)

	try:
		# --- 兼容性尝试 ---
		# 第一次尝试：使用 Ollama 原生的 'prompt' 字段名
		resp_json = _post({"model": model, "prompt": text})
		try:
			# 尝试从返回结果中提取向量
			vec = _extract_embedding(resp_json)
			# 如果成功提取出一个非空向量，则直接返回原始 JSON 响应
			if isinstance(vec, list) and len(vec) > 0:
				return resp_json
		except Exception:
			# 如果提取失败，则忽略异常，继续进行下一次尝试
			pass
		
		# 第二次尝试：兼容某些使用 'input' 字段名的实现（类似 OpenAI API）
		resp_json_input = _post({"model": model, "input": text})
		return resp_json_input

	# --- 错误处理 ---
	except HTTPError as e:
		# 捕获 HTTP 错误（如 404, 500），并提供更详细的错误信息
		error_body = e.read().decode('utf-8', errors='replace')
		raise RuntimeError(f"HTTP {e.code} 调用失败: {error_body}") from e
	except URLError as e:
		# 捕获网络连接相关的错误（如无法连接服务器）
		raise RuntimeError(f"无法连接到 Ollama 服务 {endpoint}: {e}") from e


def main() -> int:
	"""
	程序的主入口函数。

	负责解析命令行参数，协调配置加载、API 调用和结果打印的整个流程。

	Returns:
		int: 程序的退出码，0 表示成功，1 表示失败。
	"""
	# --- 命令行参数解析 ---
	parser = argparse.ArgumentParser(description="Ollama bge-m3 向量生成演示")
	parser.add_argument("--text", default="这是一个用于测试 bge-m3 嵌入向量的示例句子。", help="要生成向量的文本")
	parser.add_argument("--config", default=str(Path.cwd() / "config.ini"), help="配置文件路径，默认当前目录下 config.ini")
	parser.add_argument("--debug", action="store_true", help="打印原始返回 JSON")
	args = parser.parse_args()

	config_path = Path(args.config)
	try:
		# --- 主逻辑 ---
		# 1. 加载配置
		base_url, model = load_embedding_config(config_path)
		print(f"使用服务: {base_url} | 模型: {model}")
		
		# 2. 调用 Ollama API
		resp_json = call_ollama_embeddings(base_url, model, args.text)
		
		# 3. (可选) 打印原始返回，用于调试
		if args.debug:
			print("原始返回:")
			print(json.dumps(resp_json, ensure_ascii=False, indent=2))
		
		# 4. 从响应中提取向量
		vector = _extract_embedding(resp_json)
		length = len(vector)
		
		# 5. 格式化并打印结果
		preview = ", ".join(f"{v:.4f}" for v in vector[:8])
		print(f"文本: {args.text}")
		print(f"向量维度: {length}")
		print(f"前8个值: [{preview}] ...")
		
		# 6. 对异常情况给出提示
		if length == 0:
			print("提示: 返回的向量长度为 0。请检查 Ollama 端 bge-m3 模型是否可用，或将 --debug 输出提供给我以便进一步分析。")
		
		# 成功执行，返回退出码 0
		return 0
	except Exception as e:
		# --- 统一异常处理 ---
		# 捕获所有在执行过程中可能发生的异常，并打印错误信息到标准错误流
		print(f"错误: {e}", file=sys.stderr)
		# 执行失败，返回退出码 1
		return 1


if __name__ == "__main__":
	# 当脚本作为主程序直接运行时，执行 main 函数，
	# 并将 main 函数的返回码作为程序的退出状态码。
	sys.exit(main())