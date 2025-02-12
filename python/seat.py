# 加载excel表格，生成座次图

# 先生成一个座次图的模板，然后根据excel表格的内容，填充模板，生成最终的座次图

# 1. 生成座次图模板
# 每一排16位， 共8排， 左边四个位置， 中间16个位置，右边4个位置
# 每一排的位置按照15，13，11，9，7，5，3，1，2，4，6，8，10，12，14，16的顺序排列

def generate_seat_template():
    seat_template = []
    for i in range(8):
        row = []
        for j in range(16):
            row.append(0)
        seat_template.append(row)
    return seat_template

def show_seat_template(seat_template):
    for row in seat_template:
        print(row)