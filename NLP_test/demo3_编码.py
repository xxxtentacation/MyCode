import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import jieba
from tensorflow.keras.preprocessing.text import Tokenizer
import joblib


def demo3():
    text = {'周杰伦','王力宏','林俊杰','张学友','陈奕迅','李宗盛','五月天','苏打绿','逃跑计划','许巍'}
    # 创建词汇映射器
    my_tokenizer = Tokenizer()
    # 训练
    my_tokenizer.fit_on_texts(text)

    print(my_tokenizer.word_index)


if __name__ == '__main__':
    demo3()