<!-- JPW的Markdown笔记模板 v1, 其中的href需要视情更改上级目录href="../../format.css -->
<link rel="stylesheet" type="text/css" href="../../format.css">


<h1>LLMs系列进阶：前馈神经网络FFN</h1>

💡 大模型是n个Transformer Block的堆叠，每个Block内由Attention模块和全连接模块组成。前者负责注意力捕捉（不含激活），后者负责增强表示，常常使用MLP（含激活函数）。


# 1. MLP作为FFN
```python
import torch
import torch.nn as nn
import torch.nn.functional as F
class MLP( nn.Module ):
    """两层，理论上MLP层扩展维度4倍，但实际不一样"""
    def __init__(self, config, device = None):
        super().__init__()
        self.add_bias = config.add_bias_linear
        self.dense_h_to_4h = nn.Linear(
            config.hidden_size,
            config.ffn_hidden_size * 2,
            bias = self.add_bias,
            device = device,
        )
        
        def swiglu(x):
            x = torch.chunk(x, 2, dim = -1)
            return F.silu( x[0] ) * x[1]

        self.activation_func = swiglu

        self.dense_4h_to_h = nn.Linear( 
            config.ffn_hidden_size,
            config.hidden_size,
            bias = self.add_bias,
            device = device,
        )
    
    def forward( self, hidden_states ):
        intermediate = self.dense_h_to_4h( hidden_states )
        intermediate = self.activation_func( intermediate )
        output = self.dense_4h_to_h( intermediate )
        return output
```

# 2. MoE作为FFN
Mixture of Experts (MoE) 混合专家模型，它通过动态选择子模型（专家）进行计算，从而在提高模型容量的同时，加快训练和推理速度，降低了计算成本，但对微调带来了较大挑战。

<p align="center">
  <img src="images/moe.PNG" width=100%><br>
</p>
作为一种基于 Transformer 架构的模型，混合专家模型主要由两个关键部分组成:
- 稀疏 MoE 层: 这些层代替了传统 Transformer 模型中的前馈网络 (FFN) 层。MoE 层包含若干“专家”(例如 8 个)，每个专家本身是一个独立的神经网络。在实际应用中，这些专家通常是前馈网络 (FFN)，但它们也可以是更复杂的网络结构，甚至可以是 MoE 层本身，从而形成层级式的 MoE 结构。
- 门控网络或路由: 这个部分用于决定哪些令牌 (token) 被发送到哪个专家。例如，在下图中，“More”这个令牌可能被发送到第二个专家，而“Parameters”这个令牌被发送到第一个专家。有时，一个令牌甚至可以被发送到多个专家。令牌的路由方式是 MoE 使用中的一个关键点，因为路由器由学习的参数组成，并且与网络的其他部分一同进行预训练。

优点：
- 模型参数量可以非常大
- 计算效率高，每次只选择一部分专家进行计算
- 自适应性强，门控网络使得模型可以根据输入数据自动选择不同专家，在不同的任务场景适应能力更强

挑战：
- 负载不均衡：部分专家被频繁选择，其他专家很少激活
- 专家的训练难题：门控机制在训练中会存在部分专家训练不足，参数更新可能不平衡。
- 门控网络设计关键且富有挑战

```python
# 非严格MoE: 这个设计模式比较特别，通过将信号分为“门控”信号和“放大”信号的方式，能够灵活地调整网络的表达能力，类似于专家模型中的加权选择过程（不过它并不是严格的 MoE 模型）。这种结构可能有助于网络在不同维度的特征之间进行更细粒度的调整，从而提高模型的表示能力和泛化能力。
import torch
import torch.nn as nn
class MistralMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size                   # 4096
        self.intermediate_size = config.intermediate_size       # 14336
        self.gate_proj = nn.Linear( self.hidden_size, self.intermediate_size, bias = False)     # 生成激活后的门控信号，与up逐位乘后输入down
        self.up_proj = nn.Linear( self.hidden_size, self.intermediate_size, bias = False)       # 生成放大信号
        self.down_proj = nn.Linear( self.intermediate_size, self.hidden_size, bias = False)     # 映射回原空间
        self.act_fn = ACT2FN[config.hidden_act]     # "silu"
    
    def forward( self, x):
        gate = self.gate_proj( x )
        gate = self.act_fn( gate )
        up = self.up_proj( x )
        down = self.down_proj( gate*up )
        
        return down * up
```