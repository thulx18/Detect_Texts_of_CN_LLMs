# Detect_Texts_of_CN_LLMs

This GitHub repository mainly consolidates my work on the text tracing task.

---

这个仓库主要汇总了我在文本溯源任务上的工作
由于以前项目的原因，尝试了许多不同方式，检测一段文本是哪个中文大模型生成的
涉及大模型：

- Baichuan
- ChatGLM
- AquilaChat
- Qwen

使用的数据：

- Wiki
- THUCNews
- weibo
- 某商用大模型的真实对话记录
- 使用开源中文大模型生成的文本

结果总结：

- LLMDet，五分类78.83%
- 微调Roberta，三分类90.04%，五分类87.93%
- SGDclassifier，真实商用文本三分类81.29%

总结：

- 文本长度对分类结果影响很大
- LLMDet本身统计的特征维度有限，即使扩大了ngram的统计量，也只提升一点点，甚至降低
- 即使在生成文本上效果比较好，真实文本可能效果不佳

补充说明：

- 由于是把几个我的尝试汇总起来，如需使用可能需要修改数据路径
