# NLPCC 2022 Shared Task 6 - Dialogue Text Analysis, Topic Extraction Demo
Topic Extraction baseline for Dialogue Text Analysis Task of nlpcc 2022 

## DTA 主题分析代码

### 数据处理和生成

- DTA_original_data：比赛提供数据(http://tcci.ccf.org.cn/conference/2022/cfpt.php)
- 词向量文件夹：搜狗词向量链接：https://pan.baidu.com/s/14k-9jsspp43ZhMxqPmsWMQ


### 模型训练步骤

- 标注数据和词向量放于DTA_original_data和related_data/word_vectors文件夹内

- 运行脚本：

  ```python
  python run.py
  ```
生成数据：生成的所有数据放于related_data文件夹内,saved_model存储最优模型

### 模型评估

训练完毕后，运行脚本：

```python
python evaluate.py
```
最终提交结果存储在related_data/submit中
