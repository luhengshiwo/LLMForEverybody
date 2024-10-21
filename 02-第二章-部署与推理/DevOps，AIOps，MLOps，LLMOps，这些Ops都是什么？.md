DevOps, AIOps, MLOps, LLMOps，这些Ops都是什么？

也许你会在很多地方看到这些词，或许AIOps和MLOps还会搞混淆，下面我们来一一解释这些Ops的含义，在开始之前，我们先了解一下CI/CD.

## 0. CI/CD

CI/CD 是持续集成（Continuous Integration）和持续交付（Continuous Delivery）的缩写。CI/CD 是一种软件开发实践，旨在通过自动化软件构建、测试和部署流程来提高开发团队的效率和质量。

![alt text](<assest/DevOps, AIOps, MLOps, LLMOps，这些Ops都是什么？/00.png>)

***持续集成（CI）***：

- 目的：频繁地将代码变更集成到主分支；
- 实践：开发者经常（通常是每天多次）将代码变更合并到共享仓库中。每次提交都通过自动化构建和自动化测试来验证，以确保变更不会破坏现有的功能；
- 工具：通常使用版本控制系统（如Git）、构建工具（如Maven、Gradle）、自动化测试工具（如JUnit、NUnit）和持续集成服务器（如Jenkins、Travis CI、GitLab CI）

***持续交付/部署（CD）***：

- 目的：确保软件可以随时部署到生产环境中；
- 实践：在持续集成的基础上，持续交付增加了将软件自动部署到测试、暂存或生产环境的步骤。这包括自动化部署流程，但不一定意味着每次变更都会立即发布到生产环境；
- 工具：除了持续集成的工具外，还包括部署工具（如Ansible、Chef、Puppet）和配置管理工具（如Terraform、CloudFormation）

## 1. DevOps

DevOps 的历史可以追溯到2007年左右，当时软件开发和 IT 运营社区开始担忧传统的软件开发模式。在这种模式下，编写代码的开发人员与部署和支持代码的运营人员会独立工作。DevOps 这一术语由“开发”和“运营”两个词构成，它反映了将这些领域整合为一个持续流程的过程。DevOps 运动在 2007 到 2008 年间开始盛行，Patrick Debois 在比利时举办的 DevOpsDays 是 DevOps 概念的重要里程碑之一，这个会议将 DevOps 理念传播到了全球.

![alt text](<assest/DevOps, AIOps, MLOps, LLMOps，这些Ops都是什么？/01.png>)

DevOps 的核心价值包括：

- 自动化：通过自动化工具和流程来减少手动操作，提高效率和减少错误。
- 持续集成/持续部署（CI/CD）：频繁地将代码变更集成到主分支，并通过自动化测试和部署流程快速发布到生产环境。
- 敏捷开发：采用敏捷方法论，如Scrum或Kanban，以快速响应变化和持续交付价值。
- 监控和反馈：实时监控系统性能和用户反馈，以便快速识别和解决问题。
- 文化和沟通：鼓励团队成员之间的开放沟通和协作，打破传统的部门壁垒。

## 2. AIOps

AIOps 是人工智能运维（Artificial Intelligence for IT Operations）的缩写，是一种利用人工智能和机器学习技术来改进 IT 运维的实践。AIOps 旨在通过自动化和智能化来提高 IT 运维的效率和质量。

![alt text](<assest/DevOps, AIOps, MLOps, LLMOps，这些Ops都是什么？/02.png>)

AIOps的核心目标是从海量的运维数据中提取有价值的信息，实现故障的快速发现、准确诊断和自动修复，从而提高IT系统的可靠性和运维效率。

AIOps的关键组成部分包括数据收集、存储和分析，以及基于这些数据的智能决策和自动化响应。它通常涉及到以下几个方面：

- 数据源：AIOps平台需要从各种IT基础设施组件、应用程序、性能监控工具和服务凭单系统中收集数据;

- 大数据分析：利用大数据技术处理和分析收集到的海量数据，以识别和预测潜在的问题;

- 机器学习：应用机器学习算法来提高对数据的理解和分析能力，从而实现更准确的故障预测和根因分析;

- 自动化：基于分析结果自动触发响应措施，减少人工干预，提高问题处理的速度和效率;

- 可视化和报告：通过可视化工具展示分析结果和运维状态，帮助IT团队更好地理解系统性能和做出决策.

AIOps的应用可以帮助企业在数字化转型的过程中，更好地管理和维护复杂的IT环境，提高服务质量，降低运营成本，并增强对业务变化的适应能力

## 3. MLOps

MLOps即机器学习运维（Machine Learning Operations），是一组工作流实践，旨在简化机器学习（ML）模型的部署和维护过程。它结合了 DevOps 和 GitOps 的原则，通过自动化和标准化流程，将机器学习模型集成到软件开发过程中。MLOps 的目标是提高模型的质量和准确性，简化管理流程，避免数据漂移，提高数据科学家的效率，从而使整个团队获益。

> tips：在工业界，人们先想到AI可以辅助运维，然后才意识到AI本身也需要运维，所以‘AIOps’的名称被先指代为AI运维，而后来AI模型的运维只能使用‘MLOps’这个名字了。

![alt text](<assest/DevOps, AIOps, MLOps, LLMOps，这些Ops都是什么？/03.png>)

MLOps 的关键组成部分包括：

- 版本控制：跟踪机器学习资产中的更改，以便重现结果并在必要时回滚到以前的版本。
- 自动化：自动执行机器学习管道中的各个阶段，确保可重复性、一致性和可扩展性。
- 连续 X：包括持续集成（CI）、持续交付（CD）、持续训练（CT）和持续监控（CM），以实现模型的持续改进和部署。
- 模型治理：管理机器学习系统的各个方面以提高效率，包括促进团队协作、确保数据安全和合规性。

MLOps 的优势包括缩短上市时间、提高工作效率、高效的模型部署和节省成本。它允许数据科学家、工程师和 IT 团队紧密合作，快速迭代和部署模型，同时确保模型在生产环境中的性能和可靠性。

至此，算法工程师的职责边界已经被拓展到了模型的部署和维护。

## 4. LLMOps

随着大模型的兴起，LLMOps（Large Language Model Operations）也逐渐成为了一个新的热门话题。LLMOps 是指大型语言模型的运维，旨在简化大模型的部署和维护过程，提高模型的质量和效率，其对标的概念是MLOps。

![alt text](<assest/DevOps, AIOps, MLOps, LLMOps，这些Ops都是什么？/04.png>)

首先，我们对比下传统软件应用（包含机器学习）和大模型应用的区别：

| | 传统软件应用| 大模型应用|
| :---: |:----:| :----: |
| 特性|预设的规则|概率和预测|
| 输出 |判别式-相同的输入，相同的输出| 非判别式-相同的输入，很多可能的输出|
| 测试|1个输出，一个准确的输出|一个输入，很多个准确（或者不准确）的输出|
| 验收标准 |验证：非对即错|验证：准确性，质量，一致性，偏见，毒性。。。|

其次，我们再看下MLOps和LLMOps的区别：

|features | MLOps| LLMOps|
| :---: |:----:| :----: |
|范围|ML模型的整个生命周期|LLM(应用)整个的生命周期|
|重点|数据准备、模型训练、模型部署、模型监控、模型再训练|数据准备、模型部署、监控、可观测性、安全性|
|工具|MLOps 平台、数据准备工具、模型训练框架、模型部署工具、监控工具|LLMOps 平台、数据准备工具、模型部署工具、监控工具、可观测性工具、安全工具|

![alt text](<assest/DevOps, AIOps, MLOps, LLMOps，这些Ops都是什么？/05.png>)

>注意： LLMOps,一般指的是大模型应用的运维，而不是大模型的训练，因为大模型的训练是一个离线的过程，而大模型的应用是一个在线的过程。

## 参考

<div id="refer-anchor-1"></div>

[1] [What is CI/CD?](https://www.mabl.com/blog/what-is-cicd)

[2] [What is DevOps and where is it applied?](https://shalb.com/blog/what-is-devops-and-where-is-it-applied/)

[3] [What is AIOps and What are Top 10 AIOps Use Cases](https://cloudfabrix.com/blog/what-is-aiops-top-10-common-use-cases/)

[4] [MLOps](https://www.databricks.com/glossary/mlops)

[5] [LLMOps: What Is It and How To Implement Best Practices](https://spotintelligence.com/2024/01/08/llmops/)

[6] [deeplearning.ai](https://www.deeplearning.ai/short-courses/automated-testing-llmops/)

## 欢迎关注我的GitHub和微信公众号，来不及解释了，快上船！

[GitHub: LLMForEverybody](https://github.com/luhengshiwo/LLMForEverybody)

仓库上有原始的Markdown文件，完全开源，欢迎大家Star和Fork！