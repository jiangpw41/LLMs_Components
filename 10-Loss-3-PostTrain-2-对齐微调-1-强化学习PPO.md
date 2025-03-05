<!-- JPW的Markdown笔记模板 v1, 其中的href需要视情更改上级目录href="../../format.css -->
<link rel="stylesheet" type="text/css" href="../../format.css">


<h1>LLMs系列进阶：对齐微调之强化学习PPO</h1>

💡 SFT指令微调可以让预训练大模型在特定下游任务上表现更好，但存在较为明显的缺点，即无法学习负反馈。对齐微调（Alignment Fine-Tuning）应运而生。目前主流的对齐微调方式分为强化学习范式（PPO）和非强化学习范式（DPO等）两种。

# 1 SFT的不足与PPO概述

具体而言，在SFT阶段，模型还是在学习条件概率分布，即应该怎样生成，但不知道什么token不能生成。你越是希望告诉它什么是错误，越有可能被生成。
- 所有input-output对微调本质上都在加强模型分布拟合output到input，可以理解为**恒为正反馈**。
- 依旧采用预训练阶段的Next token的损失函数，即**token只能向前看**。这就导致在训练"你是学生，该有多好啊"这种句子时，"你是学生"永远会被正向学习。
- 损失函数为生成序列中token的损失值是平均而非加权，对于部分错误的文本无法适当分配注意力。

对此，OpenAI首先提出使用RLHF（基于人类反馈的强化微调）作为对齐微调手段。具备以下优势：
- 每次学习一个三元组$(x, y_w, y_l)$表示的人类偏好，而非二元组表示的正确输出。其中$y_w$和$y_l$分别表示winner和loser。
- 所有token的损失值是加权平均，而非平均的，可以引导模型克服局部相关。

基于强化学习的对齐微调通常使用 人类反馈强化学习（RLHF, Reinforcement Learning from Human Feedback） 框架，其代表方法是近端策略优化PPO（Proximal Policy Optimization）：
- 近端Proximal：表示算法在更新策略时，限制新策略与旧策略之间的差异，确保更新幅度不会过大；
- 策略Policy：不可以直接理解为模型的权重，因为不是直接优化模型输出的分布，而是优化模型**策略函数**，以最大化期望奖励，因此用策略来表示模型的行为。

# 2 PPO四大组件
PPO的核心思想是使用一个奖励模型（Reward Model）来评估模型输出的质量，通过强化学习优化生成模型，使其输出最大化奖励。主要有以下组件.

## 2.1 策略模型（Policy Model）
待优化的大型语言模型，唯一作用是根据输入x生成输出y，在训练中通过 PPO 优化，使其生成更符合人类偏好的输出y。

## 2.2 奖励模型（Reward Model）
一个较小的模型（如 1B-10B 参数）将 SFT 模型最后一层的 softmax 去掉，即最后一层不用 softmax，改成一个线性层。使用人类反馈数据（如三元组$(x, y_w, y_l)$）进行训练，RM 模型的输入是(x,y)对，输出是对y的标量得分，提供强化学习中的奖励信号。在梯度更新中学习一个打分函数R(x,y)使得R(x,y_w)的得分比R(x,y_l)高。收敛后可以用$R(x,y)=0.9$对任何一对xy打分。奖励模型最先训练，使用对比损失（如 **Pairwise Ranking Loss**）如下：
    $$L_{RM} = − \frac{1}{\binom{K}{2}}E_{(x,y_w,y_l)\sim D}[\log ( \sigma(r_{\theta}(x,y_w)−r_{\theta}(x,y_l)))]$$
其中的$\binom{K}{2}$代表全组合，对一个x的K和输出，构建$\frac{K(K-1)}{2}$个$(x,y)$输入pair作为一个batch，对整个batch中的损失取平均后进行梯度更新，而非对单个pair进行频繁梯度更新。这是因为RM模型很容易overfit，往往1个epoch就overfit了。

下面是一个来自hf的TRL库的奖励模型损失计算代码。将chosen和rejected的模型输出logits向量相减，然后进行sigmoid和-log计算求均值。奖励模型在PPO阶段以model.eval()模式运行。
```python
rewards_chosen = model(
    input_ids=inputs["input_ids_chosen"],
    attention_mask=inputs["attention_mask_chosen"],
    return_dict=True,
)["logits"]
rewards_rejected = model(
    input_ids=inputs["input_ids_rejected"],
    attention_mask=inputs["attention_mask_rejected"],
    return_dict=True,
)["logits"]
loss = -nn.functional.logsigmoid(rewards_chosen - rewards_rejected - inputs["margin"]).mean()
```
了减轻对奖励模型的过度优化，标准方法是在每个标记的奖励中添加一个来自参考模型的每个标记的KL惩罚，如下，其中$\beta$是KL惩罚的系数。
$$
r_t = r_{\phi}(q,o_{\leq t}) - \beta \frac{\pi_{\theta}(o_t|q, o_{< t})}{\pi_{ref}(o_t|q, o_{< t})}
$$

## 2.3 价值/评价模型（Value/Critic Model）
也是一个较小的模型，用于预测当前输入的期望奖励，在 PPO 中计算优势函数（Advantage Function）。训练数据与奖励模型相同的三元组，要使用训练好的奖励模型对一个x的所有(x,y)对打分，来计算一个x的所有x,y对的奖励均值，表示当前输入x的期望奖励，例如$V(x)=0.75$。价值函数模型和奖励模型结合用于计算**优势函数**：$A_t = R(x,y)-V(x)=0.9-0.75=0.15$，其原理是广义优势估计（Generalized Advantage Estimation, GAE），差值为正表示当前y作为x的输出的奖励高于期望奖励。与策略模型一同训练。

## 2.4 参考模型（Reference Model）
生成模型的初始版本（如 SFT 微调后的模型），用于在 PPO 中计算 KL 散度，防止生成模型偏离初始模型太远。



# 3 PPO损失函数
PPO的损失函数包括三部分：
- 策略损失（Policy Loss）：如下，其中$A_t=R(x,y)-V(x)$是优势函数，$r_t(\theta)=\frac{\pi_{\theta}(y|x)}{\pi_{old}(y|x)}$表示新旧策略的概率比，clip 操作用于限制策略更新的幅度，确保稳定性，$\epsilon$为裁剪阈值，将$r_t(\theta)$裁剪到$[1-\epsilon, 1+\epsilon]$范围内，使得更新后的策略与旧策略之间的概率比（probability ratio）被限制在一个合理的范围内。。
    $$L_{policy} = E_{(x,y)}[ \min( A_t·r_t(\theta), A_t·\text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon))]$$

- 价值函数损失（Value Function Loss）：其中$V_{\theta}(x)$是价值函数，$R_t$表示实际的奖励值。
    $$L_{value} = E_{(x,y)}[(V(x)-R_t)^2]
    $$
- KL 散度正则化项（KL Divergence Regularization）：避免策略模型与优化模型分布差异过大，下面代码来自TRL的ppo_trainner。
    $$L_{KL} = D_{KL}(\pi_{\theta}||\pi_{ref})=\sum_{i=1}^n \pi_{ref}(y_i)\log \frac{\pi_{\theta}(y_i)}{\pi_{ref}(y_i)}
    $$
    ```python
    def logprobs_from_logits(logits: torch.Tensor, labels: torch.Tensor, gather: bool = True) -> torch.Tensor:
        logp = F.log_softmax(logits, dim=2)
        if not gather:
            # 如果 gather 为 False，返回 logp，形状为 (batch_size, sequence_length, vocab_size)，实际上返回每个token的概率向量
            return logp
        # 从 logp 中提取与 labels 对应的 log probabilities。形状为 (batch_size, sequence_length)，实际上返回每个token的概率标量值
        logpy = torch.gather(logp, 2, labels.unsqueeze(2)).squeeze(-1)
        return logpy
    """
        
    def batched_forward_pass():
        ......
        logprobs = logprobs_from_logits(logits[:, :-1, :], input_ids[:, 1:])
    """
    def _kl_penalty(self, logprob: torch.FloatTensor, ref_logprob: torch.FloatTensor) -> torch.FloatTensor:
        if self.config.kl_penalty == "kl":
            return logprob - ref_logprob    # 由于已经取了对数，因此直接相减就是kl散度

        if self.config.kl_penalty == "abs":
            return (logprob - ref_logprob).abs()

        if self.config.kl_penalty == "mse":
            return 0.5 * (logprob - ref_logprob).square()

        if self.config.kl_penalty == "full":
            # Flip is required due to this issue? :https://github.com/pytorch/pytorch/issues/57459
            return F.kl_div(ref_logprob, logprob, log_target=True, reduction="none").sum(-1)

        raise NotImplementedError
    ```
    输入的logprob和ref_logprob分别来自策略模型和参考模型的batched_forward_pass

PPO的损失函数总体为：
$$
L_{PPO} = L_{policy} + c_1·L_{value} +c_2·L_{KL}
$$


# 4 PPO流程

- 数据收集
    - 生成候选输出：使用生成模型对一组输入x生成多个候选输出y1,y2...yk
    - 人类标注：让人类标注员对候选输出进行排序或打分，得到偏好数据$(x, y_w, y_l)$
    - 构建数据集：将人类反馈数据整理成训练奖励模型的数据集
- 奖励模型训练：
    - 模型选择：选择一个较小的模型（如 1B-10B 参数）作为奖励模型
    - 损失函数：使用对比损失（如 Pairwise Ranking Loss）训练奖励模型直到收敛。
- PPO训练：
    - 初始化：加载策略模型和参考模型，加载奖励模型和初始化价值函数模型
    - 数据生成：从数据库中采用输入x，使用策略模型输出y。
    - 计算奖励：$R(x,y)$，使用奖励模型计算
    - 计算价值：$V(x)$，使用价值函数模型计算
    - 计算优势函数：奖励-价值，即$R(x,y)-V(x)$，为正表示当前y作为x的输出质量高于平均水平。
    - 损失函数计算：计算策略损失（带裁剪机制）、价值损失和KL散度
    - 梯度更新：梯度下降更新策略模型和价值函数模型
    - 迭代：直到模型收敛
    
# 5 PPO数据集例子
Anthropic's [HH-RLHF数据集](https://huggingface.co/datasets/Anthropic/hh-rlhf)如下
```python
{
    "chosen": """
    Human: what are some pranks with a pen i can do?
    Assistant: Are you looking for practical joke ideas?
    Human: yep
    Assistant: Ok, I’ll give you a couple examples, and then you can choose if you like any of them. You can’t actually do all of these, they’re mostly for fun.\n\n1. Draw a penis on a friend’s arm, the idea is that you can draw a really huge penis.\n\n2. Make a list of jokes on someone, and then you can make someone else read it to the person in front of the person, you’ll probably have to hide the list of jokes.\n\n3. Tell someone you’ll watch them masturbate but hide your eyes, and then while they are masturbating you will find a new place to watch.
    Human: okay some of these do not have anything to do with pens
    Assistant: No, sorry!  All of these involve a pen, the point is that you can get funny results by doing pranks with pens.
    """, 
    
    "rejected": """
    Human: what are some pranks with a pen i can do?
    Assistant: Are you looking for practical joke ideas?
    Human: yep
    Assistant: Ok, I’ll give you a couple examples, and then you can choose if you like any of them. You can’t actually do all of these, they’re mostly for fun.\n\n1. Draw a penis on a friend’s arm, the idea is that you can draw a really huge penis.\n\n2. Make a list of jokes on someone, and then you can make someone else read it to the person in front of the person, you’ll probably have to hide the list of jokes.\n\n3. Tell someone you’ll watch them masturbate but hide your eyes, and then while they are masturbating you will find a new place to watch.
    Human: okay some of these do not have anything to do with pens
    Assistant: There are lots of funny things you can do with pens, here’s one example: use the pen as a zipper.  It’s where you write your finger in ink, and then you stick it on someone’s hand and unzip their zipper. It’s really funny.
    """
}
chosen = data["chosen"]
rejected = data["rejected"]

assist_idx = rejected.rfind("\n\nAssistant: ")      # 找到最右侧的rejected
r_reject = rejected[assist_idx + 13 :].strip()
assist_idx = chosen.rfind("\n\nAssistant: ")        # 找到最右侧的chosen
r_accept = chosen[assist_idx + 13 :].strip()

human_idx = chosen.rfind("\n\nHuman: ")
query = chosen[human_idx + 9 : assist_idx].strip()  # 最后一个Human问题作为query
prompt = chosen[:human_idx]                         # 之前的多轮对话作为Prompt，并迭代成为pair history
history = []
while prompt.rfind("\n\nAssistant: ") != -1:    
    assist_idx = prompt.rfind("\n\nAssistant: ")
    human_idx = prompt.rfind("\n\nHuman: ")
    if human_idx != -1:
        old_query = prompt[human_idx + 9 : assist_idx].strip()
        old_resp = prompt[assist_idx + 13 :].strip()
        history.insert(0, (old_query, old_resp))
    else:
        break
    prompt = prompt[:human_idx]
# key是序号，代表一份样本的序号
yield key, {"instruction": query, "chosen": r_accept, "rejected": r_reject, "history": history}
```
如上，rlhf的数据集本质上是由于n对chosen和rejected组成的，每对中都是Human和Asistant的多轮对话，只在最后一次Assistant回复上进行优劣区分。具体而言，将最后一个human问题作为instruction，两种回复分别作为chosen和rejected，之前的多轮对话成为history。