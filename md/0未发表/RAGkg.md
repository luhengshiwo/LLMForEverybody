### Populate the vector index
- Calculate vector representation for each movie tagline using OpenAI
- Add vector to the `Movie` node as `taglineEmbedding` 标语嵌入property、


### Similarity search
- Calculate embedding for question
- Identify matching movies based on similarity of question and `taglineEmbedding` vectors


## 1. 导入


## 2. 不同的公司使用的术语不同：
- 
>

## 3. 理论的大模型推理速度

> 1


***太慢了！***

| 策略 | 难度| 数据要求|准确性提升|
| :--- |:----:| :----: |---: |
| Prompt engineering|低|无| 26%   |
| Self-reflection |低| 无|26-40% |
| Few-shot learning (with RAG)|中|少量|50% |
| Instruction Fine-tuning |高|中等|40-60%|

## 参考

<div id="refer-anchor-1"></div>

## 欢迎关注我的GitHub和微信公众号：

[GitHub: LLMForEverybody](https://github.com/luhengshiwo/LLMForEverybody)




