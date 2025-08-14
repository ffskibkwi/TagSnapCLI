import argparse
import configparser
import json
import math
import sys
from pathlib import Path
from urllib.request import Request, urlopen
from urllib.error import URLError, HTTPError
from typing import List, Any


def load_embedding_config(config_path: Path):
	parser = configparser.ConfigParser()
	if not config_path.exists():
		raise FileNotFoundError(f"未找到配置文件: {config_path}")
	parser.read(config_path, encoding="utf-8")
	if not parser.has_section("embedding"):
		raise KeyError("config.ini 缺少 [embedding] 配置段")
	ip_addr = parser.get("embedding", "ip_addr", fallback="http://127.0.0.1:11434").strip()
	model = parser.get("embedding", "model", fallback="bge-m3").strip()
	if not ip_addr or not model:
		raise ValueError("[embedding] ip_addr 与 model 不能为空")
	return ip_addr, model


def _extract_embedding(resp_json: Any) -> List[float]:
	# 1) { "embedding": [...] } 或 { "embedding": [[...]] }
	if isinstance(resp_json, dict) and "embedding" in resp_json:
		emb = resp_json.get("embedding")
		if isinstance(emb, list):
			# 形如 [[...]] 时取第一个
			return emb[0] if emb and isinstance(emb[0], list) else emb

	# 2) { "embeddings": [...] } 可能是 [floats] 或 [{"embedding": [...]}]
	if isinstance(resp_json, dict) and "embeddings" in resp_json:
		embs = resp_json.get("embeddings")
		if isinstance(embs, list):
			if embs and isinstance(embs[0], dict) and "embedding" in embs[0]:
				return embs[0]["embedding"]
			# 形如 [floats] 或 [[floats]]
			if embs and isinstance(embs[0], list):
				return embs[0]
			return embs

	# 3) OpenAI 风格 { "data": [{ "embedding": [...] }, ...] }
	if isinstance(resp_json, dict) and "data" in resp_json:
		data = resp_json.get("data")
		if isinstance(data, list) and data:
			first = data[0]
			if isinstance(first, dict) and "embedding" in first:
				return first["embedding"]

	# 4) 兜底：直接是向量或批量向量
	if isinstance(resp_json, list):
		return resp_json[0] if resp_json and isinstance(resp_json[0], list) else resp_json

	raise RuntimeError(f"无法解析返回结果: {resp_json}")


def call_ollama_embeddings(base_url: str, model: str, text: str, timeout: float = 60.0):
	endpoint = base_url.rstrip("/") + "/api/embeddings"

	def _post(body: dict) -> Any:
		data = json.dumps(body).encode("utf-8")
		request = Request(
			endpoint,
			data=data,
			headers={"Content-Type": "application/json"},
			method="POST",
		)
		with urlopen(request, timeout=timeout) as response:
			resp_text = response.read().decode("utf-8", errors="replace")
			return json.loads(resp_text)

	try:
		# 1) Ollama 原生字段：prompt
		resp_json = _post({"model": model, "prompt": text})
		try:
			vec = _extract_embedding(resp_json)
			if isinstance(vec, list) and len(vec) > 0:
				return resp_json
		except Exception:
			pass
		# 2) 兼容某些实现使用 input 字段
		resp_json_input = _post({"model": model, "input": text})
		return resp_json_input
	except HTTPError as e:
		raise RuntimeError(f"HTTP {e.code} 调用失败: {e.read().decode('utf-8', errors='replace')}") from e
	except URLError as e:
		raise RuntimeError(f"无法连接到 Ollama 服务 {endpoint}: {e}") from e


def _cosine_similarity(vec_a: List[float], vec_b: List[float]) -> float:
	if not vec_a or not vec_b:
		return 0.0
	limit = min(len(vec_a), len(vec_b))
	va = vec_a[:limit]
	vb = vec_b[:limit]
	dot = sum(a * b for a, b in zip(va, vb))
	norm_a = math.sqrt(sum(a * a for a in va))
	norm_b = math.sqrt(sum(b * b for b in vb))
	if norm_a == 0.0 or norm_b == 0.0:
		return 0.0
	return dot / (norm_a * norm_b)


def main() -> int:
	parser = argparse.ArgumentParser(description="Ollama bge-m3 向量生成演示")
	parser.add_argument("--text", default="这是一个用于测试 bge-m3 嵌入向量的示例句子。", help="要生成向量的文本")
	parser.add_argument("--text-b", default="", help="可选：第二段文本，如提供则计算与 --text 的余弦相似度")
	parser.add_argument("--config", default=str(Path.cwd() / "config.ini"), help="配置文件路径，默认当前目录下 config.ini")
	parser.add_argument("--debug", action="store_true", help="打印原始返回 JSON")
	args = parser.parse_args()

	config_path = Path(args.config)
	try:
		base_url, model = load_embedding_config(config_path)
		print(f"使用服务: {base_url} | 模型: {model}")
		resp_json_a = call_ollama_embeddings(base_url, model, args.text)
		if args.debug:
			print("原始返回 A:")
			print(json.dumps(resp_json_a, ensure_ascii=False, indent=2))
		vector_a = _extract_embedding(resp_json_a)
		length_a = len(vector_a)
		preview_a = ", ".join(f"{v:.4f}" for v in vector_a[:8])
		print(f"文本 A: {args.text}")
		print(f"向量维度 A: {length_a}")
		print(f"前8个值 A: [{preview_a}] ...")
		if length_a == 0:
			print("提示: 返回的向量长度为 0。请检查 Ollama 端 bge-m3 模型是否可用，或将 --debug 输出提供给我以便进一步分析。")

		if args.text_b:
			resp_json_b = call_ollama_embeddings(base_url, model, args.text_b)
			if args.debug:
				print("原始返回 B:")
				print(json.dumps(resp_json_b, ensure_ascii=False, indent=2))
			vector_b = _extract_embedding(resp_json_b)
			length_b = len(vector_b)
			print(f"文本 B: {args.text_b}")
			print(f"向量维度 B: {length_b}")
			sim = _cosine_similarity(vector_a, vector_b)
			print(f"余弦相似度: {sim:.6f}")
		return 0
	except Exception as e:
		print(f"错误: {e}", file=sys.stderr)
		return 1


if __name__ == "__main__":
	sys.exit(main())


