import pandas as pd
import numpy as np

# 任意的多组列表
lng = np.random.normal(117,0.20,1000)

lat = np.random.normal(32.00,0.20,1000)

# 字典中的key值即为csv中列名
dataframe = pd.DataFrame({'lng':lng,'lat':lat})


#将DataFrame存储为csv,index表示是否显示行名，default=True
dataframe.to_csv('data/lng-lat.csv',index = False, sep=',' )
