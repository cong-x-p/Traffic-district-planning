import csv
import random


# 遗传算法
class GeneticAlgorithm:
    # -----------初始数据定义---------------------
    # 定义一个的二维数组表示配送中心(编号为0)与客户之间，以及客户相互之间的距离d[i][j]
    '''
    example :
    0 2 5
    2 0 3
    5 3 0
    '''
    d = [[]]

    # 客户分别需要的货物的需求量，第0位为配送中心自己
    '''
    example:
    0 1 3
    '''
    q = []

    # 定义一些遗传算法需要的参数
    JCL = 0.9  # 遗传时的交叉率
    BYL = 0.09  # 遗传时的变异率
    JYHW = 5  # 变异时的基因换位次数
    PSCS = 20  # 爬山算法时的迭代次数

    def __init__(self, rows, times, mans, cars, tons, distance, PW):
        self.rows = rows  # 排列个数
        self.times = times  # 迭代次数
        self.mans = mans  # 客户数量
        self.cars = cars  # 车辆总数
        self.tons = tons  # 车辆载重
        self.distance = distance  # 车辆一次行驶的最大距离
        self.PW = PW  # 当生成一个不可行路线时的惩罚因子

    # 初始化族群序列
    def initGroup(self):
        users = [i for i in range(1, self.mans + 1)]
        V = []

        for _ in range(self.rows):
            random.shuffle(users)
            V.append(users.copy())

        return V

    # -------------遗传函数开始执行---------------------
    def run(self):
        # 路线数组
        lines = self.initGroup()

        # 适应度
        fit = [self.calFitness(i, False) for i in lines]

        # 迭代次数
        t = 0

        while t < self.times:
            # 适应度
            newLines = [[0 for i in range(self.mans)] for i in range(self.rows)]
            nextFit = [0 for i in range(self.rows)]
            randomFit = [0 for i in range(self.rows)]
            tmpFit = 0

            # 计算总的适应度
            totalFit = sum(fit)

            # 通过适应度占总适应度的比例生成随机适应度
            for i in range(self.rows):
                randomFit[i] = tmpFit + fit[i] / totalFit
                tmpFit += randomFit[i]

            # 上一代中的最优直接遗传到下一代
            m = fit[0]
            ml = 0

            for i in range(self.rows):
                if m < fit[i]:
                    m = fit[i]
                    ml = i

            for i in range(self.mans):
                newLines[0][i] = lines[ml][i]

            nextFit[0] = fit[ml]

            # 对最优解使用爬山算法促使其自我进化
            self.clMountain(newLines[0])

            # 开始遗传
            nl = 1
            while nl < self.rows:
                # 根据概率选取排列
                r = int(self.randomSelect(randomFit))

                # 判断是否需要交叉，不能越界
                if random.random() < self.JCL and nl + 1 < self.rows:
                    fline = [0 for x in range(self.mans)]
                    nline = [0 for x in range(self.mans)]

                    # 获取交叉排列
                    rn = int(self.randomSelect(randomFit))

                    f = int(random.uniform(0, self.mans))
                    l = int(random.uniform(0, self.mans))

                    min = 0
                    max = 0
                    fpo = 0
                    npo = 0

                    if f < l:
                        min = f
                        max = l
                    else:
                        min = l
                        max = f

                    # 将截取的段加入新生成的基因
                    # 除排在第一位的最优个体外,另N - 1 个个体要按交叉概率Pc 进行配对交叉重组。
                    while min < max:
                        fline[fpo] = lines[rn][min]
                        nline[npo] = lines[r][min]

                        min += 1
                        fpo += 1
                        npo += 1

                    for i in range(self.mans):
                        if self.isHas(fline, lines[r][i]) == False:
                            fline[fpo] = lines[r][i]
                            fpo += 1

                        if self.isHas(nline, lines[rn][i]) == False:
                            nline[npo] = lines[rn][i]
                            npo += 1

                    # 基因变异
                    self.change(fline)
                    self.change(nline)

                    # 交叉并且变异后的结果加入下一代
                    for i in range(self.mans):
                        newLines[nl][i] = fline[i]
                        newLines[nl + 1][i] = nline[i]

                    nextFit[nl] = self.calFitness(fline, False)
                    nextFit[nl + 1] = self.calFitness(nline, False)

                    nl += 2
                else:
                    # 不需要交叉的，直接变异，然后遗传到下一代

                    line = [0 for i in range(self.mans)]
                    i = 0
                    while i < self.mans:
                        line[i] = lines[r][i]
                        i += 1

                    # 基因变异
                    self.change(line)

                    # 加入下一代
                    i = 0
                    while i < self.mans:
                        newLines[nl][i] = line[i]
                        i += 1

                    nextFit[nl] = self.calFitness(line, False)
                    nl += 1

            # print(新的一代覆盖上一代 当前是第 %d 代" %(t))

            for i in range(self.rows):
                for h in range(self.mans):
                    lines[i][h] = newLines[i][h]
                fit[i] = nextFit[i]
            t += 1

        # 上代中最优的为适应函数最小的
        m = fit[0]
        ml = 0

        for i in range(self.rows):
            if m < fit[i]:
                m = fit[i]
                ml = i

        # 输出结果:
        self.calFitness(lines[ml], True)

        print("最优权值为: %f" % (m))
        print("最优结果为:")
        for i in range(self.mans):
            print("%d" % (lines[ml][i]), end=',')
        print('')

    # -----------------遗传函数执行完成--------------------

    # -----------------各种辅助计算函数--------------------
    # 线路中是否包含当前的客户
    def isHas(self, line, num):
        for i in range(0, self.mans):
            if line[i] == num:
                return True
        return False

    # 计算适应度,适应度计算的规则为每条配送路径要满足题设条件，并且目标函数即车辆行驶的总里程越小，适应度越高
    def calFitness(self, line, isShow):
        carTon = 0  # 当前车辆的载重
        carDis = 0  # 当前车辆行驶的总距离
        totalDis = 0  # 所有车辆行驶的总距离

        r = 0  # 表示当前需要车辆数
        fore = 0  # 表示正在运送的客户编号
        M = 0  # 表示当前的路径规划所需要的总车辆和总共拥有的车辆之间的差，如果大于0，表示是一个失败的规划，乘以一个很大的惩罚因子用来降低适应度

        splitPoint = []

        # 遍历每个客户点
        i = 0
        while i < self.mans:
            # 行驶的距离
            newDis = carDis + self.d[fore][line[i]]

            # 当前车辆的载重
            newTon = carTon + self.q[line[i]]

            # 如果已经超过最大行驶距离或者超过车辆的最大载重，切换到下一辆车
            if newDis + self.d[line[i]][0] > self.distance or newTon > self.tons:
                # 下一辆车
                totalDis += carDis + self.d[fore][0]  # 后面加这个d[fore][0]表示需要从当前客户处返程的距离
                r += 1
                fore = 0
                i -= 1  # 表示当前这个点的配送还没有完成
                carTon = 0
                carDis = 0

                splitPoint.append(line[i])
            else:
                carDis = newDis
                carTon = newTon
                fore = line[i]
            i += 1
        # 加上最后一辆车的距离和返程的距离
        totalDis += carDis + self.d[fore][0]

        if isShow:
            print("总行驶里程为: %.1fkm" % (totalDis))
            print("截断点为:" + ','.join(str(i) for i in splitPoint))
        else:
            # print("中间过程尝试规划的总行驶里程为: %.1fkm" %(totalDis))
            pass

        # 判断路径是否可用，所使用的车辆数量不能大于总车辆数量
        if r - self.cars + 1 > 0:
            M = r - self.cars + 1

        # 目标函数，表示一个路径规划行驶的总距离的倒数越小越好
        result = 1 / (totalDis + M * self.PW)

        return result

    # 爬山法
    def clMountain(self, line):
        oldFit = self.calFitness(line, False)
        i = 0
        while i < self.PSCS:
            f = random.uniform(0, self.mans)
            n = random.uniform(0, self.mans)
            self.doChange(line, f, n)
            newFit = self.calFitness(line, False)
            if newFit < oldFit:
                self.doChange(line, f, n)
            i += 1

    # 基因变异
    # 当满足变异率的条件时，随机的两个因子发生多次交换，交换次数为变异迭代次数规定的次数
    def change(self, line):
        if random.random() < self.BYL:
            i = 0
            while i < self.JYHW:
                f = random.uniform(0, self.mans)
                n = random.uniform(0, self.mans)
                self.doChange(line, f, n)
                i += 1

    # 将线路中的两个因子执行交换
    def doChange(self, line, f, n):
        line[int(f)], line[int(n)] = line[int(n)], line[int(f)]

    # 根据概率随机选择的序列
    def randomSelect(self, ranFit):
        ran = random.random()
        for i in range(self.rows):
            if ran < ranFit[i]:
                return i


# 读取数据函数
def read_distance(filename='distance.csv'):
    data = []
    with open(filename) as f:
        rows = csv.reader(f)
        for i in rows:
            data.append(i)

    data = data[1:]
    for i in range(len(data)):
        data[i] = data[i][1:]
        for j in range(len(data[i])):
            data[i][j] = float(data[i][j])

    return data


def read_need(filename='need.csv'):
    data = []
    with open(filename) as f:
        rows = csv.reader(f)
        for i in rows:
            data.append(i)

    data = [float(i) for i in data[0]]
    return data


"""
输入参数的的意义依次为
        self.rows = rows                            #排列个数
        self.times = times                          #迭代次数
        self.mans = mans                            #客户数量
        self.cars = cars                            #车辆总数
        self.tons = tons                            #车辆载重
        self.distance = distance                    #车辆一次行驶的最大距离
        self.PW = PW                                #当生成一个不可行路线时的惩罚因子
"""
if __name__ == '__main__':
    ga = GeneticAlgorithm(rows=20, times=25, mans=8, cars=2, tons=8, distance=50, PW=100)

    ga.d = read_distance()
    ga.q = read_need()

    for i in range(50):
        print("第 %d 次训练：" % (i + 1))
        ga.run()
