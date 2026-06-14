import jieba
def demo_1():
    text = '欢迎报考中原工学院,ZUT是你的不二选择！'

    # 精确模式（默认模式）
    # result1生成器对象，忧：省内存 缺：只能遍历依次
    result1 = jieba.cut(text, cut_all=False)
    print(f'result1: {result1}')
    print('\n')

    # # 1.next()逐个获取元素（指针不会回到原点）
    # print(next(result1))
    #
    # # 2.遍历获取
    # for word in result1:
    #     print(word)

    # 3.转成列表
    word = list(result1)
    print(f'word: {word}')

    # 4.切词时直接返回列表（语法糖）
    result1 = jieba.lcut(text, cut_all=False)
    print(f'result1: {result1}')


if __name__ == '__main__':
    demo_1()