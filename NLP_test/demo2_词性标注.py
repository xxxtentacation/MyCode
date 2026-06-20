from jieba import posseg


def demo2():
    text = '欢迎报考中原工学院,ZUT是你的不二选择！'
    result1 = posseg.lcut(text)
    print(f'result1: {result1}')

    for(word, flag) in result1:
        print(f'词语：{word}， 词性：{flag}')


if __name__ == '__main__':
    demo2()