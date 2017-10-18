#encoding=utf-8 
import numpy as np 
def main():
    lst = [[1,3,5],[2,4,6]]
    print(type(lst))
    np_lst = np.array(lst)
    print(type(np_lst))
    # 同一种numpy.array中只能有一种数据类型
    # 定义np的数据类型
    # 数据类型有：bool int int8 int16 int32 int64 int128 uint8 uint16 uint32 uint64 uint128 float16/32/64 complex64/128
    np_lst = np.array(lst,dtype=np.float)

    print(np_lst.shape)
    print(np_lst.ndim)#数据的维度
    print(np_lst.dtype)#数据类型
    print(np_lst.itemsize) #每个元素的大小
    print(np_lst.size)#数据大小 几个元素

    # numpy array
    print(np.zeros([2,4]))# 生成2行4列都是0的数组
    print(np.ones([3,5]))

    print("---------随机数Rand-------") 
    print(np.random.rand(2,4))# rand用于产生0～1之间的随机数 2*4的数组
    print(np.random.rand())
    print("---------随机数RandInt-------")
    print(np.random.randint(1,10)) # 1~10之间的随机整数
    print(np.random.randint(1,10,3))# 3个1～10之间的随机整数
    print("---------随机数Randn 标准正太分布-------")
    print(np.random.randn(2,4)) # 2行4列的标准正太分布的随机整数
    print("---------随机数Choice-------")
    print(np.random.choice([10,20,30]))# 指定在10 20 30 里面选一个随机数生成
    print("---------分布Distribute-------")
    print(np.random.beta(1,10,100))# 生成beta分布
if __name__ == "__main__":
    main()
