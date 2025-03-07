Github主页: https://github.com/deepseek-ai
API文档: https://api-docs.deepseek.com/zh-cn/guides/json_mode
集成工具 (integration): https://github.com/deepseek-ai/awesome-deepseek-integration


开源五项基础训练工具
- FlashMLA：基于 Hopper GPU 的高效 MLA 解码内核
- DeepEP：专为 MoE 和 EP 设计的高效通信库
- DeepGEMM：高效矩阵乘法（GEMM）库
- DualPipe：双向流水线并行算法
	- EPLB：自动平衡 GPU 负载
	- profile-data：训练和推理框架的分析数据
- 3FS：高性能分布式文件系统

DeepSeek 系列模型的主要发展历程和关键技术创新：
1. DeepSeek LLM (V1基础版, 2024-01))：与Llama类似, 但优化如下
	- 多阶段学习率调度器而非Llama的余弦学习率调度器
	- 分组查询注意力机制（GQA）替代了传统的多头注意力机制
	- 采用更深结构而不是更宽的结构
2. DeepSeekMath: 与DeepSeek LLM相同的模型架构
3. DeepSeek-V2: 2024-05
	- Attention：MLA替代原来的GQA
	- FFN：采用了DeepSeekMoE体系结构，目的是为了实现最终的专家专业化
4. DeepSeek-V3: 2024-12
	- 无辅助损失的负载均衡策略
	- 多词元预测训练目标：将预测范围扩展到每个位置的多个未来token
5. DeepSeek-R1(-Zero): 基于DeepSeek-V3-Base模型
6. DeepSeek-R1-Distill 