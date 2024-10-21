GraphRAG: 解锁大模型对叙述性私有数据的检索能力（中文翻译）

https://www.microsoft.com/en-us/research/blog/graphrag-unlocking-llm-discovery-on-narrative-private-data/

## 写在最前面
在前一段时间，微软开源的GraphRAG引起了一些轰动，我看了很多资料，其中最有价值应该是这篇微软自己的博客：https://www.microsoft.com/en-us/research/blog/graphrag-unlocking-llm-discovery-on-narrative-private-data/

文章内容详实，因此尝试将其翻译成中文，希望这能帮助到更多需要相关信息的人。

由于译者水平有限，翻译过程中难免会有错误，还请大家多多包涵。如果有任何问题，欢迎在评论区指出，我会尽快修改。

对于标题中的Narrative Private Data，这里翻译成了叙述性私有数据，如果有更好的翻译，欢迎指出。

阅读提示：文中使用的数据集包含敏感主题，译者只做翻译，译文不代表译者立场。

## 正文开始：

LLM 面临的最大挑战（也是最大的机遇）或许是将其强大的能力扩展到解决训练数据之外的问题，并使用 LLM 从未见过的数据获得可比的结果。这为数据调查开辟了新的可能性，例如根据上下文和数据集识别主题和语义概念。在本文中，我们介绍了微软研究院创建的 GraphRAG，这是增强 LLM 能力的一项重大进步。

检索增强生成 (RAG) 是一种根据用户查询搜索信息并提供结果作为生成 AI 答案的参考的技术。该技术是大多数基于 LLM 的工具的重要组成部分，大多数 RAG 方法都使用向量相似性作为搜索技术。GraphRAG 使用 LLM 生成的知识图谱，在对复杂信息进行文档分析时显著提高问答性能。这建立在我们最近的[研究基础](https://www.microsoft.com/en-us/research/publication/can-generalist-foundation-models-outcompete-special-purpose-tuning-case-study-in-medicine/)之上，该研究指出在私有数据集上执行发现时提示词增强（prompt augmentation）的能力。（这句话由于英文的表述习惯，翻译起来有点拗口，意思大概是在私有数据上做检索的时候，如果你很好的做prompt，那会得到比较好的检索结果，并因此带来好的RAG结果,译者注）在这里，我们将私有数据集定义为 LLM 未经过训练且从未见过的数据，例如企业的专有研究、商业文档或通信。Baseline RAG 是为了帮助解决这个问题而创建的，但我们观察到 Baseline RAG 表现非常差的情况。例如：

- Baseline RAG 很难将各个点连接起来。当回答问题需要通过共享属性遍历不同的信息片段以提供新的综合见解时，就会发生这种情况。
- 当被要求全面理解大型数据集甚至单个大型文档中的总结语义概念时，Baseline RAG 表现不佳。

为了解决这一问题，技术社区正在努力开发扩展和增强 RAG 的方法（例如 [LlamaIndex](https://www.llamaindex.ai/)）。微软研究院的新方法 GraphRAG 使用 LLM 创建基于私有数据集的知识图谱。然后，此图谱与图机器学习一起使用，在查询时执行提示词增强。GraphRAG 在回答上述两类问题方面表现出显著的进步，表现出的智能或掌握程度优于之前应用于私有数据集的其他方法。

## 将 RAG 应用于私有数据集

为了证明 GraphRAG 的有效性，让我们首先使用新闻文章中的暴力事件信息 (VIINA) [数据集](https://github.com/zhukovyuri/VIINA) 进行调查。之所以选择此数据集，是因为它的复杂性以及存在不同意见和部分信息。这是一个混乱的真实世界测试案例，而且时间还很短，因此没有被纳入 LLM 基础模型的训练中。

在本研究中，我们使用了2023年6月来自俄罗斯和乌克兰新闻来源的数千篇新闻文章（翻译成英文），以创建一个私有数据集，我们将在该数据集上执行基于 LLM 的检索。该数据集太大，无法放入 LLM 上下文窗口，因此需要采用 RAG 方法。（技术的软实力可见一斑，你可以不动声色的夹带私货并向全世界传播，译者注）

我们从一个探索性查询开始，该查询针对基线 RAG 系统和我们的新方法 GraphRAG 提出：

Query: “What is Novorossiya?” 此部分不做翻译

|Baseline RAG|GraphRAG|
| :--- |:--- |
|The term “Novorossiya” is not mentioned in the provided context. However, to provide some information, Novorossiya, or New Russia, is a historical term referring to a region in the south of Ukraine that was conquered by the Russian Empire in the 18th century. In recent years, the term has been revived by Russian-backed separatists in Eastern Ukraine to refer to the territories they claim, but it is not officially recognized as a separate entity by international bodies. |<ol><li>Novorossiya, also known as New Russia, is a term with historical and contemporary significance. Historically, it referred to a region in the southern part of the Russian Empire that is now part of Ukraine. In the contemporary context, it has been used by pro-Russian separatists to refer to certain areas of Ukraine [Data: Entities (6494, 912, 6493)].</li><li>The term Novorossiya has been used in the context of a political movement in Ukraine, particularly by Russian nationalists. This movement has been associated with efforts to destabilize Ukraine and create a so-called “Novorossiya” [Data: Entities (912); Relationships (15211, 15206)]. The term has also been used to denote a separatist state in Ukraine, known as the Federal Republic of Novorossiya (FRN) [Data: Entities (7690)].  </li><ol>|

在这些结果中，我们可以看到两个系统都表现良好 - 突出显示了基线 RAG 表现良好的一类查询。让我们尝试一个需要连接点的查询：

Query: “What has Novorossiya done?”

|Baseline RAG|GraphRAG|
| :--- |:---|
|The text does not provide specific information on what Novorossiya has done.| <ol><li>Novorossiya, a political movement in Ukraine, has been involved in a series of destructive activities, particularly targeting various entities in Ukraine [Entities (6494, 912)]. The movement has been linked to plans to destroy properties of several Ukrainian entities, including Rosen, the Odessa Canning Factory, the Odessa Regional Radio Television Transmission Center, and the National Television Company of Ukraine [Relationships (15207, 15208, 15209, 15210)]. In addition to these, Novorossiya has also been associated with plans to destroy properties of Roshen, the Odessa Conservatory Factory, the Odesa Regional Radio Television Transmission Center, and the National Telecommunications Company of Ukraine [Relationships (8329, 8330, 8331, 8332)]. Furthermore, Novorossiya has been implicated in plans to blow up ATMs of PrivatBank, the largest state bank in Ukraine[Relationships (8335, 8335)].</li><li>The Office of the General Prosecutor in Ukraine has reported on the creation of Novorossiya, indicating the government’s awareness and potential concern over the activities of this movement [Relationships (8264, 8264)]. The destructive activities of Novorossiya have had significant impacts on the targeted entities and have likely contributed to the ongoing tensions and conflicts in the region. /  In summary, Novorossiya has been involved in a series of destructive activities targeting various entities in Ukraine. These activities have had significant impacts on the targeted entities and have likely contributed to the ongoing tensions and conflicts in the region. The Ukrainian government, through the Office of the General Prosecutor, has acknowledged the existence and activities of Novorossiya, indicating a level of concern over the movement’s actions.</li><ol>|

基线 RAG 无法回答这个问题。查看插入上下文窗口的源文档（见下图），没有任何文本片段讨‘Novorossiya’，导致此问题。

![alt text](<assest/GraphRAG 解锁大模型对叙述性私人数据的检索能力（中文翻译）/0.png>)

相比之下，GraphRAG 方法在查询 Novorossiya 中发现了一个实体。这使得 LLM 能够在图中立足，并通过指向原始支持文本的链接得出包含出处的更优答案。例如，下图 2 显示了 LLM 用于 LLM 生成的声明“Novorossiya has been implicated in plans to blow up ATMs”的确切内容。我们看到了原始源文档（经过英文翻译）中的片段，LLM 使用该片段通过图中两个实体之间存在的关系来支持这一断言。

![alt text](<assest/GraphRAG 解锁大模型对叙述性私人数据的检索能力（中文翻译）/1.png>)

通过使用 LLM 生成的知识图谱，GraphRAG 极大地改进了 RAG 的“检索”部分，用更高相关性的内容填充上下文窗口，从而得到更好的答案并捕获证据来源。

能够信任和验证 LLM 生成的结果始终很重要。我们关心的是结果是否正确、连贯，并准确代表源材料中的内容。GraphRAG 在生成每个响应时提供出处或来源基础信息。它表明答案是基于数据集的。每个断言的引用来源随时可用，也使人类用户能够快速准确地直接对照原始源材料审核 LLM 的输出。

然而，这并不是使用 GraphRAG 所能实现的全部。

## 全数据集推理

Baseline RAG 很难处理需要汇总整个数据集的信息才能得出答案的查询。诸如“数据中的前 5 个主题是什么？”之类的查询表现不佳，因为 Baseline RAG 依赖于对数据集内语义相似的文本内容进行向量搜索。查询中没有任何内容可以将其引导至正确的信息。

但是，使用 GraphRAG，我们可以回答这些问题，因为 LLM 生成的知识图谱的结构告诉我们整个数据集的结构（以及主题）。这允许将私有数据集组织成预先汇总的有意义的语义集群。在响应用户查询时，LLM 使用这些集群来总结这些主题。

我们通过向这两个系统提出以下问题来说明全数据集推理能力：

Query: “What are the top 5 themes in the data?“

|Baseline RAG|GraphRAG|
| :--- |:--- |
|<ol><li>Improving the quality of life in cities and urban development.</li><li>The meeting between Vladimir Putin and the President of the United Arab Emirates, Mohammed bin Zayed Al Nahyan.</li><li>The National Rating of the Investment Climate in the Regions of Russia.</li><li>The state of the Russian economy and the growth of domestic brands.</li><li>The discussion on the meaning of life and the role of science in understanding it. </li></ol> |<ol>The data presents a wide range of themes, but the top five most prevalent themes can be identified as follows: <li>Conflict and Military Activity: A significant portion of the data revolves around the ongoing conflict in Ukraine, with numerous reports detailing military activities, occupations, and attacks by various entities. This theme includes the actions of the Ukrainian military, Russian forces, and various other military units and entities [Data: Reports (513, 241, 151, 493, 343, +more)].</li><li>Political and Governmental Entities: Many reports focus on political figures and governmental entities, highlighting their roles, relationships, and actions in the context of the current situation. This includes entities like the Kremlin, the President of Ukraine, the US State Department, and various regional administrations [Data: Reports (98, 493, 265, 129, 196, +more)]. </li><li>Infrastructure and Environmental Concerns: Several reports highlight potential environmental impacts and infrastructure threats. These include potential environmental damage due to industrial activities, flooding crises, and threats to infrastructure due to military conflicts [Data: Reports (308, 154, 201, 348, 216, +more)].</li><li>Community Analysis and Threat Assessment: Several reports provide detailed analyses of specific communities, often centered around a particular location or entity. These analyses typically discuss the key entities within the community, their relationships, and any associated threats. Many reports also include assessments of potential threats associated with the entities or situations they discuss [Data: Reports (323, 596, 518, 583, 179, +more)]. </li><li>Health, Humanitarian Concerns, and Emergency Situations: Some reports focus on health and humanitarian issues, such as the Shereshevsky-Turner Syndrome community and the potential risk of intestinal diseases in the Odessa region. The data also covers emergency situations, including natural disasters, accidents, and attacks [Data: Reports (204, 149, 372, 154, 532, +more)]. <li>These themes provide a comprehensive overview of the current situation, highlighting the complexity and interconnectedness of the issues at hand.</li><ol>

查看基线 RAG 的结果，我们发现列出的主题与两国之间的战争没有太大关系。正如预期的那样，向量搜索检索到不相关的文本，这些文本被插入到 LLM 的上下文窗口中。所包含的结果很可能与“主题”一词有关，导致对数据集中发生的事情的评估不太有用。

通过观察 GraphRAG 的结果，我们可以清楚地看到，结果与整个数据集的情况更加一致。答案提供了在数据集中观察到的五个主要主题以及支持细节。参考报告由 LLM 为 GraphRAG 中的每个语义集群预先生成，进而提供对原始源材料的出处。

## 创建由 LLM 生成的知识图谱

我们注意到 GraphRAG 所依据的基本流程，它建立在我们之前使用图形机器学习的[研究](https://www.microsoft.com/en-us/worklab/patterns-hidden-inside-the-org-chart)和[GitHub仓库](https://github.com/graspologic-org/graspologic)之上：（面试时会考,译者注）

- LLM 处理整个私有数据集，创建对源数据中所有实体和关系的引用，然后使用这些引用创建 LLM 生成的知识图谱。
- 然后使用此图谱创建自下而上的聚类，将数据分层组织成语义聚类（下图中用不同颜色表示）。这种分区允许预先总结语义概念和主题，这有助于全面理解数据集。
- 在查询时，这两种结构都用于在回答问题时为 LLM 上下文窗口提供材料。

![alt text](<assest/GraphRAG 解锁大模型对叙述性私人数据的检索能力（中文翻译）/2.jpg>)

上图显示了该Graph的一个可视化示例。每个圆圈代表一个实体（例如，一个人、一个地方或一个组织），实体大小代表该实体拥有的关系数量，颜色代表相似实体的分组。颜色分区是一种建立在图形结构之上的自下而上的聚类方法，它使我们能够回答不同抽象层次的问题。

## 结果指标

上述示例代表了 GraphRAG 在不同主题领域的多个数据集上的持续改进。我们使用 LLM 评分器进行评估，以确定 GraphRAG 和基线 RAG 之间的成对优胜者，从而评估这一改进。我们使用一组定性指标，包括全面性（在问题隐含背景框架内的完整性）、人类赋权（提供支持性源材料或其他背景信息）和多样性（对提出的问题提供不同的观点或角度）。初步结果表明，GraphRAG 在这些指标上的表现始终优于基线 RAG。

除了相对比较之外，我们还使用 [SelfCheckGPT](https://arxiv.org/pdf/2303.08896) 对忠实度进行绝对测量，以帮助确保结果基于源材料，真实、连贯。结果表明，GraphRAG 的忠实度与基线 RAG 相似。我们目前正在开发一个评估框架来衡量上述问题类别的表现。这将包括更强大的机制来生成问答测试集以及其他指标，例如准确性和上下文相关性。

## 下一步

通过结合 LLM 生成的知识图谱和图机器学习，GraphRAG 使我们能够回答仅使用基线 RAG 无法解决的重要问题。在将这项技术应用于社交媒体、新闻文章、工作场所生产力和化学等各种场景后，我们已经看到了令人鼓舞的结果。展望未来，我们计划在继续应用这项技术的同时，在各种新领域与客户密切合作，同时致力于指标和稳健评估。随着研究的继续，我们期待分享更多信息。


## 注释

[1] 在本次比较中，我们使用 LangChain 的[Q&A](https://python.langchain.com/v0.1/docs/use_cases/question_answering/)作为基线 RAG，这是当今广泛使用的此类 RAG 工具的著名代表性示例。

[2] 此数据集包含敏感主题。选择此数据集的唯一目的是展示数据分析工具，这些工具可显示所有相关信息，包括其来源。这些工具以该数据集信息为基础，使人类用户能够根据来自乌克兰语 (unian) 和俄语 (ria) 新闻文章的对立观点，更快地得出明智的结论，这些新闻文章均以他们的母语为来源。这些工具突出显示了每条声明的来源，可用于识别信息的来源。

## 正文结束


## 欢迎关注我的GitHub和微信公众号，来不及解释了，快上船！

[GitHub: LLMForEverybody](https://github.com/luhengshiwo/LLMForEverybody)

仓库上有原始的Markdown文件，完全开源，欢迎大家Star和Fork！