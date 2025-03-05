<!-- JPW的Markdown笔记模板 v1, 其中的href需要视情更改上级目录href="../../format.css -->
<link rel="stylesheet" type="text/css" href="../../format.css">


<h1>LLMs系列进阶：对齐微调之非强化学习微调</h1>


非强化学习的对齐微调直接使用对比学习或偏好优化方法，无需奖励模型和强化学习框架。代表方法包括：
- 直接偏好优化DPO（Direct Preference Optimization）
- KTO（Kahneman-Tversky Optimization）
- ORPO（Odds Ratio Preference Optimization）


# 1 DPO直接偏好优化
无需显式奖励模型，直接优化模型，使其对高质量输出的概率高于低质量输出，损失函数如下，其中$\beta$是温度参数，控制偏好强度：
$$
L_{DPO} = -E_{(x,y_w,y_l)}[ \log \sigma(\beta(\log \frac{\pi_{\theta}(y_w|x)}{\pi_{ref}(y_w|x)}-\log \frac{\pi_{\theta}(y_l|x)}{\pi_{ref}(y_l|x)}))]
$$

上式中，DPO的损失函数中包含了一个对数概率的差异，是基于Bradley-Terry模型$P(i胜过j) = \frac{\alpha_i}{\alpha_i+\alpha_j}$，其中其中$\alpha_i$表示i的能力值，推导如下：

$$P(y_1>y_2)=\frac{exp(r(x,y_1))}{exp(r(x,y_1))+exp(r(x,y_2))}$$

其中$r(x,y)$表示某个y对于x的能力值，即奖励函数。接下来将上面这个概率公式转换为可优化的损失函数形式, 步骤如下,主要是将上述概率转换为负对数( 0-1概率加log后小于0, 取负后大于0, 概率越大越接近1, 则负对数越小), 核心是这个P(y_1>y_2)的概率要尽可能大(说明模型区分好坏的能力强), 这等价于其负对数尽可能小, 损失函数如下：

$$
\text{Loss}= -E_{(x, y_w, y_l)\sim D}[\ln\frac{exp(r(x,y_w))}{exp(r(x,y_w))+exp(r(x,y_l))}] =\\
-E_{(x, y_w, y_l)\sim D}[\ln\frac{1}{1+exp(r(x,y_l)-r(x,y_w))}] = \\
-E_{(x, y_w, y_l)\sim D}[\ln\sigma(r(x,y_l)-r(x,y_w))]
$$

上式中，价值函数$r(x,y)$为$\log \frac{\pi_{\theta}(y_l|x)}{\pi_{ref}(y_l|x)}$形式，隐含地处理了KL散度作为正则项。从优化的对偶形式看，**DPO的目标等价于在奖励最大化与KL散度最小化之间权衡**，但通过参数化合并为一个监督学习问题。其中:
- **优化目标**：奖励最大化（w-l最大化），本质上是最大化差异$E_{(x,y_w,y_l)}[ \log \pi_{\theta}(y_w|x)-\log \pi_{\theta}(y_l|x)]$。
- **奖励参数化**：DPO将奖励函数参数化为策略与参考策略的对数概率比$r(x,y)=\beta\log \frac{\pi_{\theta}(y_w|x)}{\pi_{ref}(y_w|x)}$，从而将强化学习目标转化为监督学习问题，无需显式训练奖励模型。这一参数化直接建立了奖励与策略差异的联系。此时，**最大化奖励**等价于**最大化策略模型与参考模型的对数概率比**，同时通过$\beta$参数控制差异幅度。
- **隐式KL散度约束**：通过奖励参数化的策略/参考比值形式$\beta\log \frac{\pi_{\theta}(y_w|x)}{\pi_{ref}(y_w|x)}$中直接编码了策略差异，从而隐含反向KL散度，避免了显式KL约束的复杂调参，将强化学习问题转化为更高效的监督学习形式。

DPO训练数据集：LLaMa-Factory dpo_zh_demo.json
```python
[
    {
        "conversations": [
            {
                "from": "human",
                "value": "国会的转发\n美国国会由众议院和参议院组成，每两年换届一次（参议员任期为6年，但..."
            }
        ],
        "chosen": {
            "from": "gpt",
            "value": "计算推文的政党边际概率，我们可以使用以下代码：\n```python\ntweets['party'] = tweets['party'].replace(0, 'Democrat')\ntweets['party'] = tweets['party'].replace(1, 'Republican')\ntweets['party'] = tweets['party'].replace(2, 'Independent')\n\nparty_counts = tweets.groupby('party').size().sort_values(ascending=False)..."
        },
        "rejected": {
            "from": "gpt",
            "value": "回答问题的第（a）部分，您可以通过首先使用groupby（）函数按政党对数据进行分组，然后计算每个组的转发数来计算政党的边际概率。具体来说，您可以使用以下代码：\n```scss\nparty_count = tweets.groupby('party').count()..."
        }
    },
]
```
如上，DPO的数据集就是conversations对话列表和chosen、rejected示例。

# 2 KTO
基于前景理论（Prospect Theory），直接优化模型对高质量输出的偏好。损失函数如下，其中$f(x,y)$是模型对输出的打分函数：
$$
L_{DPO} = -E_{(x,y_w,y_l)}[ \log \sigma(\beta(f(x,y_w)-f(x,y_l)))]
$$
真实数据集：LLaMa-Factory kto_en_demo.json
```python
[
    {
        "messages": [
            {
                "content": "The Federal Trade Commission is going after spyware, bits of computer code that install themselves on the computers of Internet users to track or disrupt their online activities.\nQ: Which is the best summary of this article?\nPick your answer from:\n(A). World\n(B). Sports\n(C). Business\n(D). Science/Tech\nI think the answer is",
                "role": "user"
            },
            {
                "content": "Ooh ooh ah ah! *giggle* Let me think... Ooh ooh ah ah! *giggle* The best summary of this article is... *drumroll*... D! Science/Tech! Yay! *confetti* The Federal Trade Commission is talking about spyware, which is like a little computer bug that can watch what you do on your computer without you knowing. It's like a secret agent, but not a nice one! *gasp* The FTC wants to stop the spyware from doing its thing, so they're going after it! *cheer* Woohoo!",
                "role": "assistant"
            }
        ],
        "label": true
  },
]
```
如上，KTO的数据集是messages多轮对话加一个Bool标签（判断输出是否可取）。

# 3 ORPO胜率偏好优化
通过优化胜率（Odds Ratio）来对齐模型输出。损失函数如下：
$$
L_{ORPO} = -E_{(x,y_w,y_l)}[ \log \frac{\pi_{\theta}(y_w|x)}{\pi_{\theta}(y_l|x)}]
$$