# 案例回顾：找出一个文本中最高词频的单词

text = '''the clown ran after the car and the car ran into the tent 
        and the tent fell down on the clown and the car'''
words = text.split() # 获取单词的列表

# 使用字典可以极大简化步骤
# 获取单词-词频字典
counts = dict() # 初始化一个空字典
for word in words:
    counts[word] = counts.get(word, 0) + 1  # 构造字典。注意get方法需要设定默认返回值0（当单词第一次出现时，词频为1）
print(counts)

# 在字典中查找最高词频的单词
bigcount = None
bigword = None
for word,count in counts.items():
    if bigcount is None or count > bigcount:
        bigword = word
        bigcount = count

print(bigword, bigcount)