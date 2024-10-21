干货：落地企业级RAG的实践指南

对于企业级数据，很多来自多种文档类型，例如 PDF、Word 文档、电子邮件和网页, 我们需要关注以下两个阶段：Load & Process，Split/Chunking

## 1. 什么是RAG？
检索增强生成（Retrieval-Augmented Generation，简称 RAG）通过结合大型语言模型（LLM）和信息检索系统来提高生成文本的准确性和相关性.这种方法允许模型在生成回答之前，先从权威知识库中检索相关信息，从而确保输出内容的时效性和专业性，无需对模型本身进行重新训练.

RAG技术之所以重要，是因为它解决了LLM面临的一些关键挑战，例如虚假信息的提供、过时信息的生成、非权威来源的依赖以及由于术语混淆导致的不准确响应.通过引入RAG，可以从权威且预先确定的知识来源中检索信息，增强了对生成文本的控制，同时提高了用户对AI解决方案的信任度.

![alt text](<assest/干货： 落地企业级RAG的实践指南/01.png>)

## 2.企业级RAG落地难点

对于企业级数据，很多来自多种文档类型，例如 PDF、Word 文档、电子邮件和网页, 我们需要关注以下两个阶段：

- Load & Process，即上图中的A，是指加载数据的过程.在实际应用中，数据的格式和结构各不相同.因此，如何高效地加载和处理这些数据是一个非常具有挑战性的问题.

- Split/Chunking，即上图中的B，是指将数据分割成多个部分的过程.在实际应用中，数据通常是非结构化的，需要进行小心的分割和处理，以便模型能够更好地理解和处理.

## 3. 需要load的数据信息
除了获取文档上的文字信息，其它的信息如文件名，页码等都是重要的结构信息.在RAG的实践中，我们需要将这些信息都提取出来，以便更好地理解和处理数据.

**文档元素**：指文档的基本构成要素,可用于各种 RAG 任务，例如过滤和分块:
- 标题
- 叙述文本
- 列表项
- 表格
- 图像

**元素元数据**: 有关元素的附加信息,可用于在混合搜索中进行筛选以及识别回答来源:
- 文件名
- 文件类型
- 页码
- 章节

> 注：如果你不明白什么是混合搜索，没关系，我们后面会详细介绍.


## 4. 对数据的处理
对数据进行处理是必要但困难的，主要因为：
- 内容提示：不同的文档类型对元素类型（视觉、markdown）有不同的提示；
- 标准化需求：要处理来自不同文档类型的内容，需要对其进行标准化；
- 提取方式不一样：不同的文档格式可能需要不同的提取方法；
- 元数据洞察：在许多情况下，提取元数据需要了解文档结构

我们需要把不同的文档类型（PDF,Word,EPUB,MarkDown等）转换成统一的格式，以便模型能够更好地理解和处理.一个简单有效的方式是将其转化为Json格式.

Json格式有如下的特点：

- 结构常见且易于理解
- 是标准的 HTTP 响应
- 能够用于多种编程语言
- 可以转换为 JSONL 用于流式传输用例

下面提供一个转化好的Json示例：
```Json
[
    {
        "element_id":"bff1fd0ec25e78f1224ad7309a1e79c4",
        "metadata":{
        "filename": "CoT.pdf",
        "filetype":"application/pdf",
        "languages":[
            "eng"
        ],
        "page_number":1,
        },
        "text":"B All Experimental Results",
        "type": "Title"
    },
    {
        "element_id":"ebf8dfb149bcbbd8c4b7f9a7046900a9",
        "metadata":{
        "filename": "CoT.pdf",
        "filetype":"application/pdf",
        "languages":[
            "eng"
        ],
        "page_number":1,
        },
        "text": "This section contains tables for experimental results for varying models and model sizes, on all benchmarks, for standard prompting vs. chain-of-thought prompting.",
        "type": "NarrativeText"
    }
]
```

对于开发者而言，就是需要找到一个框架，能够处理不同的文档类型，将其转化为Json格式.
一些文档类型（例如 HTML、Word Docs 和 Markdown）包含格式信息，可以使用基于规则的解析器进行预处理；但是对于 PDF 或者图像文档，需要使用其它技术进行处理.这些技术一般不是开源的，需要购买或者自己开发.


## 6. 语义搜索和混合搜索
对于绝大多数人来说，语义搜索Semantic Search 并不陌生，语义搜索的目标是给定一个输入文本，从文档语料库中查找语义相似的内容以用于加入到Prompt中.
但语义搜索并不是万能的，它有一些局限性，比如：
- 搜索结果过多：在有大量文档的情况下，语义相似的匹配结果太多了；
- 最新信息：用户可能想要最新的信息，而不仅仅是语义最相似的信息；
- 重要信息丢失：丢失了文档中与搜索相关的重要结构信息，例如标题、页码等.

混合策略：混合搜索是一种将语义搜索与其他信息检索技术（如过滤和关键字搜索）相结合的搜索策略.过滤选项来自文档的元数据.


## 7. 分块Chunking

向量数据库需要将文档分割成块，以便检索和生成提示.根据文档的分块方式，相同的查询将返回不同的内容.

**均等大小的块**：最简单的方法是将文档分割成大小大致均等的块.这会导致相似的内容被分割成多个块.

**按原子元素分块**：通过识别原子元素，可以通过组合元素而不是分割原始文本来分块.这样可以产生更连贯的块

***步骤***：

1. 分区：首先，将文档分解为原子元素；
2. 将元素组合成块：向块中添加文档元素，直到达到字符或标记阈值.
3. 应用中断条件：应用开始新块的条件，例如当我们到达新的标题元素（表示新部分）、元素元数据更改（表示新页面或部分）或内容相似性超过阈值时.
4. 组合较小的块：可选地，组合小块，以便块足够大以进行有效的语义搜索.

***要点***：
- 连贯的块：将来自同一文档元素的内容保持在一起，从而产生更连贯的块.
- 结构化块：允许分块方法利用文档结构，而传统的分块技术则不是这样，它们根据标记或字符数进行拆分.

![alt text](<assest/干货： 落地企业级RAG的实践指南/17.png>)

## 8. 处理数据中图片的方案

对于其他文档，例如PDF和图像，信息是视觉化的.我们需要Document lmage Analysis (DlA) 从文档的原始图像中提取格式信息和文本.
目前，DIA有两个主要的方法：

- Document Layout Detection (DLD) 使用目标检测模型在文档图像上绘制和标记布局元素周围的边界框
- VisionTransformer (ViT) 模型将文档图像作为输入，并生成结构化输出（如 JSON）的文本表示作为输出.

![alt text](<assest/干货： 落地企业级RAG的实践指南/2.png>)

具体的，DLD的步骤是1）视觉检测：使用计算机视觉模型（例如 YOLOX 或 Detectron2）识别和分类边界框.2）文本提取：必要时使用对象字符识别（OCR）从边界框中提取文本.
注意：对于某些文档（例如 PDF），可以直接从文档中提取文本，而无需使用OCR.

![alt text](<assest/干货： 落地企业级RAG的实践指南/22.png>)

而ViT指的是文档图像传入编码器，由解码器生成文本输出，其中Document Understanding Transformer(DONUT)是一种常见的架构，它不需要 OCR 而是将图像输入直接转换为文本，甚至可以训练模型使用直接输出有效的 JSON 字符串！

## 9. 处理数据中表格的方案

大多数 RAG 用例都侧重于文档中的文本内容，与此同时，一些行业（例如金融、保险）大量处理嵌入在非结构化文档中的结构化数据.为了支持表格问答等用例，我们需要从文档中提取表格.

业界目前有三种技术：
- Table Transformer：识别表格单元格边界框并将输出转换为 HTML;
- Vision Transformer：使用上一节（预处理 PDF 和图像）中的视觉转换器模型，但以 HTML 作为输出;
- OCR + Table Parser：使用 OCR 提取表格，然后使用表格解析器将其转换为结构化数据. 

## 参考

[1] [deeplearning.ai](https://www.deeplearning.ai/short-courses/preprocessing-unstructured-data-for-llm-applications/)

[2] [unstructured.io](https://unstructured.io/)

[3] [github:unstructured](https://github.com/Unstructured-IO/unstructured)

## 欢迎关注我的GitHub和微信公众号，来不及解释了，快上船！

[GitHub: LLMForEverybody](https://github.com/luhengshiwo/LLMForEverybody)

仓库上有原始的Markdown文件，完全开源，欢迎大家Star和Fork！