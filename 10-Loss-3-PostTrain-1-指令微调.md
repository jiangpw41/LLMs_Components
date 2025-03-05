<!-- JPW的Markdown笔记模板 v1, 其中的href需要视情更改上级目录href="../../format.css -->
<link rel="stylesheet" type="text/css" href="../../format.css">


<h1>LLMs系列进阶：指令微调</h1>

💡 预训练是在大规模数据集上进行自回归无监督/自监督训练，其损失函数遵循next token条件概率最大化优化目标（多元交叉熵损失最小）。而在有监督微调SFT (Supervised Fine-Tuning)阶段，用户给出一对输入输出(input,output)，向模型指名**正确输出**是什么，以进行单轮任务型微调，属于**确定性任务（单步映射）**。

损失函数方面，预训练阶段逐一自回归预测，对每个token进行一次反向传播（实际上可能会采取梯度积累的方式）。在SFT有监督微调阶段，往往提供一对文本对(input, output)，其中output作为监督标签，可以是一句话，也可以是分类标签等。损失函数大体还是一样，但仅对 output 部分的文本计算平均损失，而不是对整个 (input, output) 序列计算损失，如下：

$$
L = -E[\log P(output|input, \theta)] + \lambda R
$$

总体而言，SFT的目标是通过指定输入和期望的输出，基于交叉熵损失优化模型输出和目标输出之间的差异。此外，在SFT阶段，KL 散度（Kullback-Leibler Divergence）通常不会作为正则化项使用。这是因为这个阶段的目标是通过监督学习让模型拟合特定的任务数据，而不是约束模型的输出分布与某个先验分布一致。

# 1 常用SFT数据集格式
Alpaca 和 ShareGPT 格式是当前主流的大模型训练数据格式，尤其在指令微调（Instruction Tuning）和对话训练（Chat Fine-tuning）场景中被广泛使用。它们的设计思路不同，但都适用于监督微调（SFT）和强化学习训练（RLHF）。前者适用于单轮任务型指令微调，后者适用于多轮对话微调。

## 1.1 Alpaca  
/ælˈpækə/
- 预训练格式：只有text字段
    ```python
    [
    {"text": "document"},
    {"text": "document"}
    ]
    ```
- 单轮指令微调格式：instruction和output必需，此外可以加input和history（instruction-output pairs list）
    ```python
    [
        {
            "instruction": "human instruction (required)",
            "input": "human input (optional)",
            "output": "model response (required)",
            "system": "system prompt (optional)",
            "history": [
            ["human instruction in the first round (optional)", "model response in the first round (optional)"],
            ["human instruction in the second round (optional)", "model response in the second round (optional)"]
            ]
        }
    ]
    ```

## 1.2 ShareGPT
ShareGPT 格式最初由 ShareGPT 社区贡献，主要用于多轮对话（Conversational Fine-tuning）。数据通常包含一个角色标注的聊天历史，适用于对话式 LLM 训练。多轮指令微调数据集格式如下：核心是一个conversations列表，列表为from role/ value text字典。
```python
[
  {
    "conversations": [
      {
        "from": "human",
        "value": "human instruction"
      },
      {
        "from": "function_call",
        "value": "tool arguments"
      },
      {
        "from": "observation",
        "value": "tool result"
      },
      {
        "from": "gpt",
        "value": "model response"
      }
    ],
    "system": "system prompt (optional)",
    "tools": "tool description (optional)"
  }
]
```

# 2 参数高效微调PEFT技术
微调的概念最早出现于2003年到2006年间由Hiton等人发现的**预训练+微调**范式对于扩展深度网络规模的作用。最开始微调主要有两种方式
- 全量微调：全部参数继续训练，训练的成本高，会出现灾难性遗忘。
- 只微调部分层：冻结大部分参数，只微调后几层，避免微调代价过大，但效果不够好。

但较长一段时间内由于模型体量没有很大，因此这些缺点相对能忍受，直到2018年后，语言模型规模越来越大，于是出现了一批参数高效微调方式，可以分为两大类：

## 2.1 非提示Promptless微调

### 2.1.1 Adapter
Parameter-Efficient Transfer Learning for NLP，参数高效微调PEFT开山之作，2019，Google.从实验结果来看，该方法能够在只额外对增加的3.6%参数规模（相比原来预训练模型的参数量）的情况下取得和Full-finetuning接近的效果（GLUE指标在0.4%以内）

### 2.1.2 LoRA低秩自适应
LORA: LOW-RANK ADAPTATION OF LARGE LANGUAGE MODEL，2021，MicroSoft. 基本原理：冻结预训练参数，拷贝一份进行增量训练；

（1）基于权重参数矩阵不满秩假设，减少拷贝增量的中间层维度，从而减少微调过程中参训参数量，降低微调成本。

（2）手动选择离散的秩r，其选择会直接改变模型结构，在选择最佳的秩上，不够灵活。清华最近提出SoRA (Sparse Low-Rank Adaption)，引入自适应秩选择机制，

## 2.2 基于提示Prompt-based微调
Fixed-LM/prompt  or Prompt+LM

### 2.2.1 Tuning-free Prompting非训练微调：零样本学习
2020年，Scaling Law生效，智能涌现，大模型出现零样本学习能力，可以不用微调仅用提示词使模型适配下游任务。提示推理不更新参数：0-shot/1-shot/few-shots，Prompt能力的证明，2020.5，GPT-3论文
### 2.2.2 Fixed-prompt LM Tuning: 文字硬/离散模板方法PET

2020.1，人工离散pattern(任务特定)+微调模型本体(迭代iPET)。微调模型(部分)，提示词不变。人精心设计任务特定的多套pattern。小样本迭代扩大数据集。

### 2.2.3 Fixed-LM Prompt Tuning：vector软/连续模板方法
2020.5，不微调模型或Prompt，GPT-3论文(证明Prompt直觉在LLMs中的可行性)
#### 2.2.3.1 Prefix前缀微调
2021.1.1，Stanford，从提示词获得灵感，可学习前缀token，占用输入长度。模型参数不变，在每layer输入前插，训练层数*Prefix token数的embedding参数(shape=n_prefix * emb_dim)微调

#### 2.2.3.2 Prompt-Tuning提示微调
Google，Prefix的简化，Prompt tuning，仅微调提示词部分，为每个任务训练一个固定的Prompt。使用静态的、可训练的虚拟标记嵌入。在输入序列前插，简化Prefix-tuning，对embedding 层中存存的100个token对应的参数进行微调(更少)

与Prefix-tuning这项工作几乎同时完成，都是对传统离散pattern的改进，仅仅调整输入token（相比Prefix作用于每一层的激活更简单），并证明仅仅提示微调就能与模型微调匹敌。核心如下2点：
（1）只要模型规模够大，简单加入 Prompt tokens 进行微调，就能取得很好的效果，其他的一切都不是问题。作者也做实验说明随着预训练模型参数量的增加，Prompt Tuning 的方法会逼近 Fine-tune 的结果。
（2）是 Prefix Tuning 的简化版本，只在输入层加入 prompt tokens（可学习的“提示”），并不需要加入 MLP 进行调整来解决难训练的问题，主要在 T5 预训练模型上做实验。

### 2.2.4 Prompt+LM Fine-tuning: P-tuning v1, v2
2021.3.18（v1）和2022.5（v2），Tsinghua，Prefix的简化，仅在输入前加入可学习的embedding，Prompt-base tuning。
- v1: 在输入序列前插，用MLP+LSTM的方式，仅更新embedding层中virtual token的部分参数
- v2: 对P-tuning v1改进，在每layer输入前插，用MLP+LSTM的方式来对Prompt Embedding进行处理。P-Tuning v2的改进在于，将只在第一层插入连续提示修改为在许多层都插入连续提示，而不仅仅是输入层


# 3 LoRA原理

<p align="center">
  <img src="images/LoRA.PNG" width=60%><br>
</p>

LoRA基于这样一种事实：经验表明，常见的预训练模型具有非常低的内在维度；换句话说，存在与全参数空间一样有效的用于微调的低维重新参数化。

如上图，LoRA的原理是通过对目标权重矩阵学习一对低秩矩阵作为近似和增量。具体的，对于输入X，先进行下投影到r维（远远小于d维，甚至可以为4和8）。

其中矩阵A高斯初始化，B零初始化。训练时损失函数还是交叉熵损失，训练完毕后将A和B加到W上。增量公式为：

$$
W = W+ \frac{\alpha}{r}AB
$$
其中，$r$是秩，默认为8，$\alpha$为缩放系数，默认为16，用于控制LoRA矩阵的影响力。

缺点：
- 手动选择离散的秩r，其选择会直接改变模型结构，在选择最佳的秩上，不够灵活。清华最近提出SoRA (Sparse Low-Rank Adaption)，引入自适应秩选择机制。


# 4 LLaMa-Factory SFT
目前主流的微调工具主要有Kiln、Unsloth、LLaMA-Factory、PEFT和hf的transformers五种。下面主要展示用LLaMA-Factory进行LoRA SFT的流程，主要从Web UI界面进行参数配置：


- **学习率lr**：初始5e-5默认
- **steps**: 步长，一个epoch中进行评估或保存的节点，通常一个step = GPUs_number*per_device_train_batch_size（GPU数量和每个GPU上的batch_size相乘）。step的概念主要体现在梯度更新上，最原始的梯度更新是每个step更新一次，也可以指定多个step进行梯度累积。
- **学习率调节器lr_scheduler_type**：cosine，还有线性、多项式、常数、余弦重启、预热与衰减等
- **epoch**：对于7B模型和千级别的SFT数据集，一般3epoch就可以，5个epoch大概率会收敛。
- **precision**：一般采用bf16，而非fp16或fp32，因为bf16在保持16位数据时，与fp32有相同的指数位（8）位，fp16是5位，但尾数位只有（7），所以bf16可以表示与fp32的范围但计算更高效，但精度不及fp16。
- **最大梯度范数**：在反向传播中防止梯度爆炸的阈值，当梯度范数（即梯度向量的‘长度’）超过这个值后，就进行缩放。默认为1.0，无特殊需要保持即可。
- **批次大小per_device_train_batch_size**: 单个GPU上每一个step采用的样本批次大小，24GB训练6-7B模型一般选择1-4以内。
- **梯度累积gradient_accumulation_steps**: 为了应对内存限制和批次大小（batch size）不足的问题。通过累积多个小批次（mini-batch）的梯度，来模拟大批次训练的效果，从而提高训练效率并避免内存溢出。
- **warmup_steps**: 在warm up阶段平滑地增加学习率，而不是一开始就使用一个较大的学习率。这种策略可以防止模型在训练初期因学习率过高而导致参数更新不稳定，尤其是在微调时，预训练模型的参数通常已经接近最优，直接使用较大的学习率可能会破坏这些已学到的知识。
- **lora_target**: 可以使用低秩增量的模型模块，主要包括Attention模块和MLP模块中的几个投影矩阵，如q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj

- **秩r**：默认为8，
- **$\alpha$**：为缩放系数，默认为16，用于控制LoRA矩阵的影响力