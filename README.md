<p align="center">
  <img src="pic/common/pr/banner.jpg" >
</p>

<p> 
<a href="https://github.com/luhengshiwo/LLMForEverybody/stargazers">
<img src="pic/common/svg/github.svg" > </a>
<a href="https://mp.weixin.qq.com/s/cV6v7yGmwYa2WwNDZjLPiQ"> <img src="pic/common/svg/wechat.svg" > </a>
<a href="https://www.zhihu.com/people/luhengshiwo"> <img src="pic/common/svg/zhihu.svg"> </a>
<a href="https://blog.csdn.net/qq_25295605?spm=1011.2415.3001.5343"> <img src="pic/common/svg/csdn.svg"> </a>
<a href="https://juejin.cn/user/3824524390049531"> <img src="pic/common/svg/juejin.svg"> </a>
</p> 

## 目录

- 🐳[序-AGI之路](#序-AGI之路)
- 🐱[第一章-大模型之Pre-Training](#第一章-大模型之Pre-Training)
  - 🐼[架构](#架构)
  - 🐹[Optimizer](#Optimizer)
  - 🐰[激活函数](#激活函数)
  - 🐭[Attention](#Attention机制)
  - 🐯[位置编码](#位置编码)
- 🐶[第二章-大模型之部署与推理](#第二章-大模型之部署与推理)
- 🐯[第三章-大模型微调](#第三章-大模型微调)
- 🐻[第四章-大模型量化](#第四章-大模型量化)
- 🐼[第五章-显卡与大模型并行](#第五章-显卡与大模型并行)
- 🐨[第六章-Prompt-Engineering](#第六章-Prompt-Engineering)
- 🦁[第七章-Agent](#第七章-Agent)
  - 🐷[RAG](#RAG)
- 🐘[第八章-大模型企业落地](#第八章-大模型企业落地)
- 🐰[第九章-大模型评估指标](#第九章-大模型评估指标)
- 🐷[第十章-热点](#第十章-热点)

## 序-AGI之路

**[⬆ 一键返回目录](#目录)** 

[大家都在谈的Scaling_Law是什么](00-序-AGI之路/大家都在谈的ScalingLaw是什么.md)

[aatest](00-序-AGI之路/aatest.md)

[智能涌现和AGI的起源](00-序-AGI之路/智能涌现和AGI的起源.md)

[什么是perplexity](https://mp.weixin.qq.com/s?__biz=MzkyOTY4Mjc4MQ==&mid=2247483766&idx=1&sn=56563281557b6f58feacb935eb6a872a&chksm=c2048544f5730c52cf2bf4c9ed60ac0a21793bacdddc4d63b481d4aa887bc6a838fecf0b6cc7&token=607452854&lang=zh_CN#rd)

[Pre-Training预训练Llama-3.1 405B超大杯，需要多少算力资源？](https://mp.weixin.qq.com/s?__biz=MzkyOTY4Mjc4MQ==&mid=2247483839&idx=1&sn=3f35dfe8ed2c87bf4c0b4ac7bfa3e6a9&chksm=c204858df5730c9b8a152a0330dee0183467a063c25aadd0da7cc47d9d5b2f97347fab22708d&token=607452854&lang=zh_CN#rd)

## 第一章-大模型之Pre-Training

**[⬆ 一键返回目录](#目录)** 

### 架构

[10分钟搞清楚为什么Transformer中使用LayerNorm而不是BatchNorm](01-第一章-预训练/10分钟搞清楚为什么Transformer中使用LayerNorm而不是BatchNorm.md)

[混合专家模型 (MoE) 详解（节选）](<01-第一章-预训练/混合专家模型 (MoE) 详解（节选）.md>)

[最简单的方式理解Mamba（中文翻译）](01-第一章-预训练/最简单的方式理解Mamba（中文翻译）.md)

[10分钟了解什么是多模态大模型（Multimodal LLMs）](<01-第一章-预训练/10分钟了解什么是多模态大模型（Multimodal LLMs）.md>)

### Optimizer

[全网最全的神经网络优化器optimizer总结](01-第一章-预训练/全网最全的神经网络优化器optimizer总结.md)

[神经网络的优化器（一）综述](01-第一章-预训练/神经网络的优化器（一）概述.md)

[神经网络的优化器（二）SGD](01-第一章-预训练/神经网络的优化器（二）SGD.md)

[神经网络的优化器（三）Momentum](01-第一章-预训练/神经网络的优化器（三）Momentum.md)

[神经网络的优化器（四）ASGD](01-第一章-预训练/神经网络的优化器（四）ASGD.md)

[神经网络的优化器（五）Rprop](01-第一章-预训练/神经网络的优化器（五）Rprop.md)

[神经网络的优化器（六）AdaGrad](01-第一章-预训练/神经网络的优化器（六）AdaGrad.md)

[神经网络的优化器（七）AdaDeleta](01-第一章-预训练/神经网络的优化器（七）AdaDeleta.md)

[神经网络的优化器（八）RMSprop](01-第一章-预训练/神经网络的优化器（八）RMSprop.md)

[神经网络的优化器（九）Adam](01-第一章-预训练/神经网络的优化器（九）Adam.md)

[神经网络的优化器（十）Nadam](01-第一章-预训练/神经网络的优化器（十）Nadam.md)

[神经网络的优化器（十一）AdamW](01-第一章-预训练/神经网络的优化器（十一）AdamW.md)

[神经网络的优化器（十二）RAdam](01-第一章-预训练/神经网络的优化器（十二）RAdam.md)

### 激活函数

[为什么大型语言模型都在使用 SwiGLU 作为激活函数？](<01-第一章-预训练/为什么大型语言模型都在使用 SwiGLU 作为激活函数？.md>)

### Attention机制

[看懂FlashAttention需要的数学储备是？高考数学最后一道大题](01-第一章-预训练/看懂FlashAttention需要的数学储备是？高考数学最后一道大题！.md)

[FlashAttention v2相比于v1有哪些更新？](<01-第一章-预训练/FlashAttention v2相比于v1有哪些更新？.md>)

[为什么会发展出Multi Query Attention和Group Query Attention](<01-第一章-预训练/为什么会发展出Multi Query Attention和Group Query Attention.md>)

### 位置编码

[什么是大模型的位置编码Position Encoding](<01-第一章-预训练/什么是大模型的位置编码Position Encoding.md>)

[复变函数在大模型位置编码中的应用](01-第一章-预训练/复变函数在大模型位置编码中的应用.md)

[最美的数学公式-欧拉公式](01-第一章-预训练/最美的数学公式-欧拉公式.md)

[从欧拉公式的美到旋转位置编码RoPE](01-第一章-预训练/从欧拉公式的美到旋转位置编码RoPE.md)

## 第二章-大模型之部署与推理

**[⬆ 一键返回目录](#目录)**

[10分钟私有化部署大模型到本地](02-第二章-部署与推理/10分钟私有化部署大模型到本地.md)

[大模型output token为什么比input token贵](<02-第二章-部署与推理/大模型output token为什么比input token贵？.md>)

[如何评判大模型的输出速度？首Token延迟和其余Token延迟有什么不同？](02-第二章-部署与推理/如何评判大模型的输出速度？首Token延迟和其余Token延迟有什么不同？.md)

[大模型的latency（延迟）和throughput（吞吐量）有什么区别](02-第二章-部署与推理/大模型的latency（延迟）和throughput（吞吐量）有什么区别.md)

[vLLM 使用PagedAttention轻松、快速且廉价地提供LLM服务（中文版翻译）](<02-第二章-部署与推理/vLLM 使用PagedAttention轻松、快速且廉价地提供LLM服务（中文版翻译）.md>)

[DevOps, AIOps, MLOps, LLMOps，这些Ops都是什么？](<02-第二章-部署与推理/DevOps, AIOps, MLOps, LLMOps，这些Ops都是什么？.md>)

## 第三章-大模型微调

**[⬆ 一键返回目录](#目录)**

[10分钟教你套壳（不是）Llama-3，小白也能上手](https://mp.weixin.qq.com/s?__biz=MzkyOTY4Mjc4MQ==&mid=2247483895&idx=1&sn=72e9ca9874aeb4fd51a076c14341242f&chksm=c20485c5f5730cd38f43cf32cc851ade15286d5bd14c8107906449f8c52db9d3bfd72cfc40c8&token=607452854&lang=zh_CN#rd)

[大模型的参数高效微调（PEFT），LoRA微调以及其它](03-第三章-微调/大模型的参数高效微调（PEFT），LoRA微调以及其它.md)

## 第四章-大模型量化

**[⬆ 一键返回目录](#目录)**

[10分钟理解大模型的量化](04-第四章-量化/10分钟理解大模型的量化.md)

## 第五章-显卡与大模型并行

**[⬆ 一键返回目录](#目录)**

[AGI时代人人都可以看懂的显卡知识](https://mp.weixin.qq.com/s?__biz=MzkyOTY4Mjc4MQ==&mid=2247484001&idx=1&sn=5a178a9006cc308f2e84b5a0db6994ff&chksm=c2048653f5730f45b3b08af03023aee24969d89ad5586e4e25c68b09393bf5a8abfd9670a6f3&token=607452854&lang=zh_CN#rd)

[Transformer架构的GPU并行和之前的NLP算法有什么不同？](05-第五章-显卡与并行/Transformer架构的GPU并行和之前的NLP算法有什么不同？.md)

## 第六章-Prompt-Engineering

**[⬆ 一键返回目录](#目录)**

[过去式就能越狱大模型？一文了解大模型安全攻防战！](<06-第六章-Prompt Engineering/过去式就能越狱大模型？一文了解大模型安全攻防战！.md>)

[万字长文 Prompt Engineering-解锁大模型的力量](<06-第六章-Prompt Engineering/万字长文 Prompt Engineering-解锁大模型的力量.md>)

[COT思维链，TOT思维树，GOT思维图，这些都是什么？](<06-第六章-Prompt Engineering/COT思维链，TOT思维树，GOT思维图，这些都是什么？.md>)

## 第七章-Agent

**[⬆ 一键返回目录](#目录)**

[开发大模型or使用大模型](07-第七章-Agent/开发大模型or使用大模型.md)

[Agent设计范式与常见框架](07-第七章-Agent/Agent设计范式与常见框架.md)

[langchain向左coze向右](07-第七章-Agent/langchain向左coze向右.md)

### RAG

[向量数据库拥抱大模型](07-第七章-Agent/向量数据库拥抱大模型.md)

[搭配Knowledge-Graph的RAG架构](<07-第七章-Agent/搭配Knowledge Graph的RAG架构.md>)

[GraphRAG：解锁大模型对叙述性私人数据的检索能力（中文翻译）](<07-第七章-Agent/GraphRAG 解锁大模型对叙述性私人数据的检索能力（中文翻译）.md>)

[干货： 落地企业级RAG的实践指南](<07-第七章-Agent/干货： 落地企业级RAG的实践指南.md>)

## 第八章-大模型企业落地

**[⬆ 一键返回目录](#目录)**

[CRUD-ETL工程师的末日从NL2SQL到ChatBI](08-第八章-大模型企业落地/CRUDETL工程师的末日从NL2SQL到ChatBI.md)

[大模型落地难点之幻觉](08-第八章-大模型企业落地/大模型落地难点之幻觉.md)

[大模型落地难点之输出的不确定性](08-第八章-大模型企业落地/大模型落地难点之输出的不确定性.md)

[大模型落地难点之结构化输出](08-第八章-大模型企业落地/大模型落地难点之结构化输出.md)

## 第九章-大模型评估指标

[大模型有哪些评估指标？](09-第九章-评估指标/大模型有哪些评估指标？.md)

[大模型性能评测之大海捞针(Needle In A Haystack)](09-第九章-评估指标/大模型性能评测之大海捞针.md)

[评估指标/大模型性能评测之数星星](09-第九章-评估指标/大模型性能评测之数星星.md)

## 第十章-热点

**[⬆ 一键返回目录](#目录)**

[Llama 3.1 405B 为什么这么大？](https://mp.weixin.qq.com/s?__biz=MzkyOTY4Mjc4MQ==&mid=2247483782&idx=1&sn=3a14a0cde14eb6643beaeb5b472ffa26&chksm=c20485b4f5730ca2d7b002a29e617a75c08d004a1b3da891ab352cbe31ca37541a546e29abc7&token=607452854&lang=zh_CN#rd)

[9.11大于9.9？大模型怎么又翻车了？](https://mp.weixin.qq.com/s?__biz=MzkyOTY4Mjc4MQ==&mid=2247483800&idx=1&sn=48b326352c37d686f7f46ee5df9f00b4&chksm=c20485aaf5730cbca8f0dfcb9746830229b8f07eec092e0e124bc558d1073ee32e3f55716221&token=607452854&lang=zh_CN#rd)

[韩国“N 号房”事件因Deep Fake再现，探究背后的技术和应对方法](<10-第十章-热点/韩国“N 号房”事件因Deep Fake再现，探究背后的技术和应对方法.md>)

[我是怎么通过2022下半年软考高级：系统架构设计师考试的](10-第十章-热点/我是怎么通过2022下半年软考高级：系统架构设计师考试的.md)

[用Exploit and Explore解决不知道吃什么的选择困难症](<10-第十章-热点/用Exploit and Explore解决不知道吃什么的选择困难症.md>)