import json

json_dict = json.loads('./1052.json')
print(json_dict)
city = ['北京市','上海市','广州市','深圳市','成都市','杭州市','武汉市','重庆市','南京市','天津市','苏州市','西安市','长沙市','沈阳市','青岛市','郑州市','大连市','东莞市','宁波市','厦门市','福州市','无锡市','合肥市','昆明市','哈尔滨市','济南市','长春市','温州市','石家庄市','南宁市','常州市','南昌市','贵阳市','太原市','烟台市','南通市','珠海市','徐州市','海口市']
l = []
for line in lines:
    linet = line.split(',')[2].split(':')[1].split('"')[1]
    print(linet)
    if linet in city:
        l_s = line.split(',')
        l.append(l_s)
    df=DataFrame(l,columns=['parent','level','name','lng','lat','people_count_2010','adcode'])
    DataFrame.to_csv(df,'10523.csv')