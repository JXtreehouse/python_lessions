text = 'the clown ran after the car and the car ran into the tent and the tent fell down on the clown and the car'
words = text.split()
print(words)

for word in words:# 初始化空列表
    print(word)


#步骤一：获得单词列表  相当于去重
unique_words = list()
for word in words:
   if(word not in unique_words):# 使用in判断某个元素是否在列表里
       unique_words.append(word)
print(unique_words)


#步骤二：初始化词频列表

# [e]*n 快速初始化
counts = [0] * len(unique_words)
print(counts)

# 步骤三：统计词频
for word in words:
    index = unique_words.index(word)

    counts[index] = counts[index] + 1
    print(counts[index])
print(counts)
# 步骤四：找出最高词频和其对应的单词
bigcount = None #None 为空，初始化bigcount
bigword = None

for i in range(len(counts)):
    if bigcount is None or counts[i] > bigcount:
        bigword = unique_words[i]
        bigcount = counts[i]
print(bigword,bigcount)