所有人都在谈Agents，但大部分人都不知道Agents设计之道

-00

2023年AI圈的主角无疑是大模型，如火如荼的百模大战是让人印象深刻，2024年站在AI圈C位的显然是AI助手，Kimi、豆包、文小言、元宝等产品更是战成一团。那么2025年AI圈新的主角，则几乎一定是智能体（AI Agent）.

- 0

目前没有一个统一的Agent设计范式，但是有一些常见的设计模式，我们这边选择吴恩达文章中提到的几种设计范式:

- Reflection: 自我反思
- Tool use： 使用工具
- Planning：通过多步骤任务规划进行推理
- Multi-agent collaboration 多Agent协作

2025年，在你jump into Agent相关开发之前，你还需要了解Agents的设计之道:

一些人将 agents 定义为一个完全自主的系统，它们能够长期独立运行，使用各种工具来完成复杂的任务；另一些则把 agents 当作更符合规范性的并遵循预定义的工作流。

在Anthropic，他们将所有这些不同的形式都归类为 agentic systems（智能系统），但在 workflows（工作流）和 agents（智能体）之间做出了一个重要的架构区分：

- Workflows 是通过预先定义好的代码路径来编排大模型和工具的系统；

- Agents 是由大模型动态规划自身处理流程和工具使用，并能够自主控制如何完成任务的系统.

这两种系统之间存在共性和区别，并在使用中不同的组织对术语的边界定义有所不同。

下面，我们对通用意义上的智能体进行一些讨论。

## 智能体的4个层级

### level 1 - 简单处理器

描述：系统只简单输出大模型的输出

代码：print_llm_output(llm_response)

典型应用：ChatGPT，kimi

### level 2 - 路由

描述：系统根据大模型的输出，选择不同的路径

代码：if llm_decision(): path_a() else: path_b()

典型应用：基于大模型的意图识别（一般用于智能客服）

### level 3 - 使用工具

描述：大模型使用外部工具来完成任务

代码：run_function(llm_chosen_tool, llm_chosen_args)

典型应用：Perplexity（调用搜索工具等）

### level 4 - 多步骤Agents

注意：这是一个新的层级，在这个层级，智能体会分化为：Workflow 和 Agents两种形式。

#### Workflow

描述：是通过预先定义好的代码路径来编排大模型和工具

典型应用：扣子等Agent编排工具

#### Agents

描述：由大模型动态规划自身处理流程和工具使用，并能够自主控制如何完成任务

典型应用：AutoGen等

- 1

## 智能体的终态：完全自主的Agent

描述：系统根据用户的要求，自主的完成任务

代码：create_and_run_code(user_request)

典型应用：ASI/AGI


## 参考

[OpenAI率先打样，今年AI圈的主角是智能体](https://36kr.com/p/3128052415404292)

[AI Agents Are Here. What Now?](https://huggingface.co/blog/ethics-soc-7)
