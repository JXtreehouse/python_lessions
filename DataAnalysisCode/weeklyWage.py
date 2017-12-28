# 使用input()函数，从键盘读取输入的文本
# a = input('请输入文本:')
# print('您输入的内容是：',a)

def salary_calculator(): #没有参数的函数
    user = str #初始化user为字符串变量
    print("----工资计算器----")

    while True:
        user = input("\n请输入你的名字，或者输入0来结束报告: ")

        if user == "0":
            print("结束报告")
            break
        else:
            hours = float(input("请输入你的工作小时数："))
            payrate =float(input("请输入你的单位时间工资： ￥"))

            if hours <= 40:
                print("员工姓名:",user)
                print("加班小时数：0")
                print("加班费：￥0.00")
                regularpay = round(hours * payrate,2) # round函数保留小数点后两位
                print("税前工资:￥" + str(regularpay))


            elif hours > 40:

                overtimehours = round(hours - 40, 2)

                print("员工姓名: " + user)

                print("加班小时数: " + str(overtimehours))

                regularpay = round(40 * payrate, 2)

                overtimerate = round(payrate * 1.5, 2)

                overtimepay = round(overtimehours * overtimerate)

                grosspay = round(regularpay + overtimepay, 2)

                print("常规工资: ￥" + str(regularpay))

                print("加班费: ￥" + str(overtimepay))

                print("税前工资: ￥" + str(grosspay))

#调用 salary_calculator

salary_calculator()