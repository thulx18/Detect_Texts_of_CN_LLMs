该方法使用了[LLMDet](https://github.com/TrustedLLM/LLMDet)在中文上的效果，代码见原仓库

主要流程说明：

1. 使用不同开源大模型生成尽可能多的文本

2. 统计ngrams字典

3. 使用字典计算ppl

4. 训练分类器

在中文上，最好的一次效果为：

<img title="" src="../img/混淆矩阵-LLMDet-7883.png" alt="混淆矩阵-LLMDet-7883.png" data-align="center">


