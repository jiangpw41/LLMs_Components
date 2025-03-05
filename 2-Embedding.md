<!-- JPW的Markdown笔记模板 v1, 其中的href需要视情更改上级目录href="../../format.css -->
<link rel="stylesheet" type="text/css" href="../../format.css">


<h1>LLMs系列进阶：Embedding</h1>

💡 分词后，自然语言文本就变成了token的整数序列，但这种序列（或one-hot）只是唯一性编码，并不蕴含语义信息。因此需要对每个token进行语义嵌入embedding，根据嵌入向量的稀疏性可以分为Dense Embedding和Sparse Embedding。

# 1 Embedding应用四大场景
embedding 模型的核心功能是将文本（或其他模态的数据）转换为向量表示，从而支持各种 NLP 任务。主要支持以下四大类下游任务：
- **信息检索（Information Retrieval, IR）**：计算 query（查询） 和 document（文档） 之间的向量相似度，匹配最相关的文档
  - 搜索引擎（Google Search、企业内部搜索）
  - 推荐系统（个性化内容推荐）
  - 问答系统（QA）支持（如 RAG：检索增强生成）
  - 代码搜索、法律文档检索
- **无监督：文本聚类（Text Clustering）**：将语义相似的文本组织在一起
  - 主题检测（新闻、社交媒体分析）
  - 自动标签生成（文档分类、论文分类）
  - 异常检测（垃圾邮件检测、舆情监控）
- **有监督：文本分类（Text Classification）**，用 embedding 作为特征，对文本进行分类学习
  - 情感分析（Sentiment Analysis）（好评/差评分类）
  - 垃圾邮件检测（Spam Detection）
  - 法律/医学文档分类
  - 社交媒体监控（如假新闻检测）
- **自监督Seq2Seq生成任务**：序列到序列任务，输入一个文本序列，基于 embedding 进行序列生成
  - 机器翻译（MT）（如 Google Translate）
  - 文本摘要（Summarization）
  - 代码生成（Code Generation）
  - 对话系统（Chatbots）

# 2 常用embedding库

主流的embedding库有如下几种
- **Transformers**：Hugging Face，广泛支持 BERT, RoBERTa, GPT, T5, BART 等模型
- **OpenAI Embeddings**：text-embedding-ada-002，基于 OpenAI GPT 训练，1536 维嵌入，语义能力强，但收费需调用 OpenAI API 获取
- **SentenceTransformers**：主流，依赖于Hugging Face Transformers，可加载 BERT、RoBERTa、MiniLM、DistilBERT 等模型。可以用于句子、文本和图像嵌入的Python库。该框架基于 PyTorch 和 Transformers。由 UKPLab（德国达姆施塔特工业大学，Technical University of Darmstadt） 开发和维护的。
- **Cohere Embeddings**：cohere.ai，支持 1024 维的高精度 embedding，类似 OpenAI embedding，提供 API 访问

# 3 常用embedding模型
主流的Embedding模型主要分为传统的静态词向量模型、传统基于LSTM的动态词向量模型和目前主流的基于Transformers的动态词向量模型三种类型。传统的Word2Vec由于其内存占用低在特定的实时搜索场景还在使用。主流的模型则包括

- Google
  - BERT及其衍生如RoBERTa：Encoder-only
  - T5系列：Encoder-Decoder，将各种下游NLP任务都统一转化成Text-to-Text格式
- Meta(FaceBook)
  - BART：Encoder-Decoder，文本理解与RoBERTa持平，兼顾生成式任务，模型体积仅比BERT大10%，性价比极高。
- OpenAI
  - text-embedding-ada-002: Encoder-only基于GPT-3的Ada架构,
- 微软
  - MiniLM: 对BERT模型进行蒸馏，体量通常只有几十MB。

## 3.1 传统非Transformer词向量
- 基于词袋模型的静态词向量
  - Word2Vec（Google，2012）: 词向量维度50-300, 滑动窗口（通常5-10词）
  - GloVe（Stanford，2014）: 
  - FastText（Meta，2016）
- 基于LSTM的动态词向量
  -  ELMo（Allen AI2，2018）
  -  CoVe（McCann，2017）
  -  ULMFit（Fast.ai+DeepMind，2018）

## 3.2 Transformer词向量

### 3.2.1 Encoder-only模型
代表是BERT系列及其各种改进如RoBERTa
- Google BERT系列
  - BERT-base：110M, 512 tokens，例如bert-base-chinese
  - BERT-large: 340M, 512 tokens
  - ALBERT: 12M (base) / 18M (large), 512 tokens
  - ELECTRA: 110M (base) / 335M (large), 512 tokens
 
- Meta(Facebook)
  - **RoBERTa**：125M (base) / 355M (large), 512 tokens, 更鲁棒的语义表示，适用于问答和文本推理。例如使用roberta-large进行相似度评估。
  - XLM-RoBERTa (XLM-R): 270M (base) / 550M (large), 512 tokens, 100+语言

### 3.2.2 生成器改进的Embedding
主要包含使用Encoder-Decoder架构的模型
  - OpenAI text-embedding-ada-002: 高质量语义嵌入，1536 维，8192 tokens, 适用于各种任务
  - Google T5 (Text-To-Text Transfer Transformer): 220M (small) 至 11B, 512 tokens
  - Meta (Facebook) BART: 140M (base) / 400M (large), 1024 tokens

### 3.2.3 蒸馏轻量化模型
主要是对主流大型词向量模型的蒸馏
- 微软MiniLM系列：对BERT进行蒸馏
  - MiniLM-L12-H384
  - MiniLM-L6-H384
  - all-MiniLM-L6-v2：权重约90MB，轻量高效，适用于大规模数据
  - ms-marco-MiniLM-L-6-v2: 权重约90MB，语义检索（特别是 MS MARCO 数据集优化的
- HuggingFace DistillBERT：对BERT进行蒸馏
  - distilbert-base-uncased: 适用于小型分类任务

### 3.2.4 其他：多语言与专用模型
- LaBSE (Language-agnostic BERT Sentence Embedding): 109种语言的跨语言语义匹配。
- Universal Sentence Encoder (USE)
- LASER (Language-Agnostic SEntence Representations): 93种语言的零样本迁移学习

# 4 代码调用
主要使用HF的transformers库或者sentence_transformers库指定模型类型进行调用。

```python
from sentence_transformers import SentenceTransformer
sentences = ["This is an example sentence", "Each sentence is converted"]

model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
embeddings = model.encode(sentences)
print(embeddings)

# 或者使用transformers
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
```