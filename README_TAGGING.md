# TagSnapCLI 标签分析功能使用说明

## 功能概述

TagSnapCLI 现已升级为智能标签分析工具，整合了以下核心功能：

1. **向量数据库初始化** - 基于 `init_tag_lab.json` 构建初始向量数据库
2. **智能标签检索** - 使用语义相似度找到最相关的10个标签
3. **AI 标签分析** - 通过 Gemini 进行精准的标签匹配和补充
4. **自动数据库更新** - 将新发现的补充标签自动添加到向量数据库

## 工作流程

```
用户输入文本 → 向量检索相似标签(10个) → Gemini分析匹配度 → 
输出匹配标签+补充标签 → 更新向量数据库
```

## 使用步骤

### 1. 初始化配置

```bash
python main.py init
```

这会创建：
- `config.ini` - 包含 Gemini API 和 Ollama 配置
- `prompts/segmenter.ini` - 原有的语义分割提示词

### 2. 确保依赖文件存在

- ✅ `init_tag_lab.json` - 初始标签词库
- ✅ `prompts/tag.ini` - 标签分析提示词
- ✅ `ollama_demo.py` - 向量嵌入功能
- ✅ Ollama 服务运行在本地（默认 http://127.0.0.1:11434）

### 3. 配置文件示例

**config.ini**:
```ini
[gemini]
api_key = YOUR_GEMINI_API_KEY
model = gemini-1.5-flash

[embedding]
ip_addr = http://127.0.0.1:11434
model = bge-m3

[proxy]
http = 
https = 
```

### 4. 启动标签分析

```bash
python main.py run
```

### 5. 使用原有语义分割功能

```bash
python main.py segment
```

## 输出格式

程序会显示：

1. **⚡ 检索到的相似标签** - 从向量数据库中找到的最相似的10个标签
2. **✅ 匹配的标签** - AI 认为与文档高度匹配的标签
3. **✨ 补充标签** - AI 发现的新的相关标签
4. **📝 分析说明** - AI 对标签选择的解释

## 技术架构

- **向量数据库**: ChromaDB (持久化存储在 `chroma_db_cosine/`)
- **嵌入模型**: Ollama + bge-m3
- **标签分析**: Google Gemini
- **界面**: Rich + prompt_toolkit

## 数据流

1. 首次运行时，从 `init_tag_lab.json` 加载初始标签并生成向量
2. 用户输入文本后，生成文本向量并检索最相似的10个标签
3. 将文本和候选标签按 `tag.ini` 格式提交给 Gemini
4. Gemini 返回匹配标签和补充标签
5. 补充标签自动添加到向量数据库以供未来使用

## 注意事项

- 确保 Ollama 服务正在运行并已加载 bge-m3 模型
- 首次运行会创建向量数据库，可能需要几分钟时间
- 向量数据库会自动增长，无需手动维护
- 支持中文标签和文本分析
