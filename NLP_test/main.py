import jieba

text = "他来到了网易杭研大厦"

# 1. 精确模式（默认模式）
# 试图将句子最精确地切开，适合文本分析
seg_exact = jieba.lcut(text)
print("【精确模式】:", "/ ".join(seg_exact))

# 2. 全模式
# 把句子中所有的可以成词的词语都扫描出来，速度非常快，但是不能解决歧义
seg_full = jieba.lcut(text, cut_all=True)
print("【全模式】  :", "/ ".join(seg_full))

# 3. 搜索引擎模式
# 在精确模式的基础上，对长词再次切分，提高召回率，适合用于搜索引擎分词
seg_search = jieba.lcut_for_search(text)
print("【搜索模式】:", "/ ".join(seg_search))