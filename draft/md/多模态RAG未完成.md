## lesson 2

bridgetower model 是一种图片-文字 pair embedding model

umap 是什么？

Notes:
- While we presented the two metrics: Cosine Similarity and
Euclidean Distance, and one visualization technique for embeddings:
UMAP above to demonstrate the meaning of embeddings,
you can also use other metrics (e.g., Cosine Distance and
Minkowski Distance) and other visualization techniques (e.g., t-SNE)
to verify the embeddings.
- There are other multimodal embedding models that can compute
the embeddings for images and texts like BridgeTower does. For example,
CLIP for image embedding and Sentence Transformer for text embedding.

## lesson 3 process videos

将 video 转化为 图片-文字 pair

## lesson 4 multimodal RAG from vector database

Notes:
- We observe that the transcripts of frames extracted from video1 are usually fragmented and even an incomplete sentence. E.g., four more was just icing on the cake for a. Thus, such transcripts are not meaningful and are not helpful for retrieval. In addition, a long transcript that includes many information is also not helpful in retrieval. A naive solution to this issue is to augment such a transcript with the transcripts of n neighboring frames. It is advised that we should pick an individual n for each video such that the updated transcripts say one or two meaningful facts.
- It is ok to have updated transcripts of neighboring frames overlapped with each other.
- Changing the transcriptions which will be ingested into vector store along with their corresponding frames will affect directly the performance. It is advised that one needs to do diligent to experiment with one's data to get the best performance.


## lesson 5 lvlm，用来回答问题

LLaVA
LLaVA (Large Language-and-Vision Assistant), an end-to-end trained large multimodal model that connects a vision encoder and LLM for general-purpose visual and language understanding.
LLaVA doesn't just see images but understands them, reads the text embedded in them, and reasons about their context—all while conversing with you in a way that feels almost natural.
LLaVA is adept at tasks like explaining a complex chart, recognizing and reading text in photos, or identifying intricate details in high-resolution images.



<p style="background-color:#fff1d7; padding:15px; "> <b>Note:</b>
<br>
* The first user's message of a conversation must include a text prompt and a base64-encoded image.
<br>  
* The follow-up user's or assistant's messages of a conversation must include a text prompt only.
</p>

QnA on Visual Cues on Images

QnA on Textual Cues on Images

QnA on Caption/Transcript Associated with Images



## lesson 6

retreive 的东西，加上query 生成 文字， 加上检索召回的图片，一起生成回答





This course, developed in partnership with Intel, teaches you to build an interactive system for querying video content using multimodal AI. You’ll create a sophisticated question-answering system that processes, understands, and interacts with video. 

Increasingly, language models and AI applications have added the capability to process images, audio, and video. In this course, you will learn more about these models and applications by implementing a multimodal RAG system. You will understand and use a multimodal embedding model to embed images and captions in a multimodal semantic space. Using that common space, you will build and use a retrieval system that returns images using text prompts. You will use a Large Vision Language Model (LVLM) to generate a response using the images and text from the retrieval.

By the end of this course, you’ll have the expertise to create AI systems that can intelligently interact with video content. This skill set opens up possibilities for developing advanced search engines that understand visual context, creating AI assistants capable of discussing video content, and building automated systems for video content analysis and summarization. Whether you’re looking to enhance content management systems, improve accessibility features, or push the boundaries of human-AI interaction, the techniques learned in this course will provide a solid foundation for innovation in multimodal AI applications.

In this course, you will make API calls to access multimodal models hosted by Prediction Guard on Intel’s cloud.

What you’ll learn in this course
Introduction to Multimodal RAG Systems: Understand the architecture of multimodal RAG systems and interact with a Gradio app demonstrating multimodal video chat capabilities.
Multimodal Embedding with BridgeTower: Explore the BridgeTower model to create joint embeddings for image-caption pairs, measure similarities, and visualize high-dimensional embeddings.
Video Pre-processing for Multimodal RAG: Learn to extract frames and transcripts from videos, generate transcriptions using the Whisper model, and create captions using Large Vision Language Models (LVLMs).
Building a Multimodal Vector Database: Implement multimodal retrieval using LanceDB and LangChain, performing similarity searches on multimodal data.
Leveraging Large Vision Language Models (LVLMs): Understand the architecture of LVLMs like LLaVA and implement image captioning, visual question answering, and multi-turn conversations.
Key technologies and concepts
Multimodal Embedding Models: BridgeTower for creating joint embeddings of image-caption pairs
Video Processing: Whisper model for transcription, LVLMs for captioning
Vector Stores: LanceDB for efficient storage and retrieval of high-dimensional vectors
Retrieval Systems: LangChain for building a retrieval pipeline 
Large Vision Language Models (LVLMs): LLaVA 1.5 for advanced visual-textual understanding
APIs and Cloud Infrastructure: PredictionGuard APIs, Intel Gaudi AI accelerators, Intel Developer Cloud
Hands-on project
Throughout the course, you’ll build a complete multimodal RAG system that:

Processes and embeds video content (frames, transcripts, and captions)
Stores multimodal data in a vector database
Retrieves relevant video segments given text queries
Generates contextual responses using LVLMs
Maintains multi-turn conversations about video content


## 参考

<div id="refer-anchor-1"></div>

[1] [multimodal-embeddings](https://learn.deeplearning.ai/courses/multimodal-rag-chat-with-videos/lesson/3/multimodal-embeddings)

## 欢迎关注我的GitHub和微信公众号，来不及解释了，快上船！

[GitHub: LLMForEverybody](https://github.com/luhengshiwo/LLMForEverybody)

仓库上有原始的Markdown文件，完全开源，欢迎大家Star和Fork！