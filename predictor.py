
# coding: utf-8



def predict_vm(ecs_lines, input_lines):
    # Do your work from here# 
    import random
    import math
        
# 解析输入数据
    phy = input_lines[0].strip('\n').split(' ')
    n_flavor = int(input_lines[2].strip('\n'))
    flavor_need = {}
    for i in range(n_flavor):
        cur_flavor = input_lines[3+i].strip('\n').split(' ')
        flavor_need[cur_flavor[0]] = [int(k) for k in cur_flavor[1:]]
    optimize = input_lines[4 + n_flavor].strip('\n')
    start_predict = input_lines[6 + n_flavor].split(' ')[0].split('-')
    end_predict = input_lines[7 + n_flavor].split(' ')[0].split('-')
    
    
    
    start_train = ecs_lines[0].split(' ')[0].split('\t')[2].split('-')
    end_train = ecs_lines[-1].split(' ')[0].split('\t')[2].split('-')
    
    def Datetime(time_1):  ##写成def Datetime(time_1，Datetime_1 = [])程序
        Datetime_1 = []
        for i in time_1:
            Datetime_1.append(i.split(' ')[0])
        return Datetime_1


    def count_flavor(a,flavor):
    #读入文件并删除用户名  
        a_1 = []
        for line in a:
            res = line.split('\t')[1:3]
            a_1.append(res)
    #将规格名称与时间分隔开
        col_1 = []
        col_2 = []
        for i in a_1:
            col_1.append(i[0])
            col_2.append(i[1])
    #按规格与日期对数据进行统计
        dict_1 = {}
    # 只保留输入的flavor那部分 
        for flavor_ in flavor:
            dict_1[flavor_] = []
        for i in range(len(col_1)):
            if col_1[i] in dict_1.keys():
                dict_1[col_1[i]].append(col_2[i])
        result = {}
        for favor in dict_1:
            a = Datetime(dict_1[favor])
            b = []
            c = []
            for i in a:
                if i not in c:
                    b.append([i,a.count(i)])
                    c.append(i)
            result[favor] = b
        return result


    def runnian(year):
        if year % 4 == 0:
            month_1 = {'01':31,
               '02':29,
               '03':31,
               '04':30,
               '05':31,
               '06':30,
               '07':31,
               '08':31,
               '09':30,
               '10':31,
               '11':30,
               '12':31}
        else:
            month_1 = {'01':31,
               '02':28,
               '03':31,
               '04':30,
               '05':31,
               '06':30,
               '07':31,
               '08':31,
               '09':30,
               '10':31,
               '11':30,
               '12':31}
        return month_1

    def full_year(year):
        month_1 = runnian(year)
        year_output = []
        for i in range(12):
            if i+1 < 10:
                for j in range(month_1[str(0)+str(i+1)]):
                    year_output.append(str(year)+'-'+(str(0)+str(i+1))+'-'+str(j+1))
            else:
                for j in range(month_1[str(i+1)]):
                    year_output.append(str(year)+'-'+(str(i+1))+'-'+str(j+1))
        return year_output
    
    def contiYMD(start, end):  ##例如start = ['2015','03','11'], end = ['2017', '04', '04']
        start_year = int(start[0])
        end_year = int(end[0])
        entire_day = []
        half = full_year(end_year)
        month_1 = runnian(int(start[0]))
        front = sum([month_1[i] for i in month_1 if int(i) < int(start[1])])+int(start[2])
        back = sum([month_1[i] for i in month_1 if int(i) < int(end[1])])+int(end[2])
        if start_year != end_year:
            for i in range(start_year,end_year):
                a = full_year(i)
                entire_day.extend(a)
            all_day = entire_day[front-1:]
            all_day.extend(half[:back])
        else:
            all_day = half[front-1:back]
        all_day_1 = []
        for i in all_day:
            if int(i.split('-')[2]) < 10:
                all_day_1.append(i.split('-')[0]+'-'+i.split('-')[1]+'-'+str(0)+i.split('-')[2])
            else:
                all_day_1.append(i)
        return all_day_1


    # 对缺失数据进行填充（输入[['2015-03-11',2],['2015-03-12',7]...['2015-04-11',1]],对每一种规格）
    def fill_zeros(date_count,start,end):
        full_day = contiYMD(start,end)
        dict_1 = {}
        for day in full_day:
            dict_1[day] = 0
        for day in date_count:
            if day[0] in dict_1:
                dict_1[day[0]] = day[1]
        return dict_1


    # date_count 是对于每种规格按时间的统计，与fill_zeros输入相同
    # （输入[['2015-03-11',2],['2015-03-12',7]...['2015-04-11',1]],对每一种规格）
    def divide_week(date_count,start,end):
        full_day = contiYMD(start,end)
        full_day_count = fill_zeros(date_count,start,end)
        n = len(full_day_count)//7
        diff = len(full_day_count) - n*7
        count_by_week = []
        for i in range(n):
            one_week = []
            for j in range(i*7+diff,(i+1)*7+diff):
                one_week.append(full_day_count[full_day[j]])
            count_by_week.append(one_week)
        return count_by_week    
    
    # 矩阵点乘
    def dot(A, B):
        res = [[0] * len(B[0]) for i in range(len(A))]
        for i in range(len(A)):
            for j in range(len(B[0])):
                for k in range(len(B)):
                    res[i][j] += A[i][k] * B[k][j]
        return res
    
    # 对应位置的元素相乘
    def multiply(A,B):
        res = [[0] * len(A[0]) for i in range(len(A))]
        for i in range(len(A)):
            for j in range(len(A[0])):
                res[i][j] = A[i][j]*B[i][j]
        return res
    
    # 求平均值
    def mean(A):
        return sum(A)/len(A)
    
    
    # 创建一个规定大小的全零列表
    def zeros(n,c):
        return [[0] * c for i in range(n)]
    
    # 列表求负值
    def neg(A):
        for i in range(len(A)):
            for j in range(len(A[0])):
                A[i][j] = 0 - A[i][j]
        return A
    
    # 将溢出值转化为正无穷
    def attempt_exp(x):
        try:
            return math.exp(x)
        except:
            return float('inf')
    # 定义激活函数
    def activation_function(A):
        B = [[0]*len(A[0]) for i in range(len(A))]
        for i in range(len(B)):
            for j in range(len(B[0])):
                B[i][j] = 1/(1+(attempt_exp(0-A[i][j])))
        return B
    
    
    # 列表相减（二维）
    def subtract(A,B):
        for i in range(len(A)):
            for j in range(len(A[0])):
                A[i][j] = A[i][j] - B[i][j]
        return A
    
    # 列表相加
    def sum_list(A,B):
        res = [[0] * len(A[0]) for i in range(len(A))]
        for i in range(len(A)):
            for j in range(len(A[0])):
                res[i][j] = A[i][j] + B[i][j]
        return res 
    
    
    # 计算平方和
    def sum_of_squares(A,B):
        C = [0 for i in range(len(A))]
        for i in range(len(A)):
            C[i] = (A[i] - B[i])**2
        res = sum(C)
        return res

    # 标准差
    def standard(A):
        B = [mean(A) for i in range(len(A))]
        S = math.sqrt(sum_of_squares(A, B)/(len(A)))
        return S

    
    # 创建一个规定大小的随机数列表
    def r_u(n,c):
        zero = [[0] * c for i in range(n)]
        for i in range(n):
            for j in range(c):
                zero[i][j] = 0.8
        return zero
    
    #对每个flavor单独赋初始值
    def r_u_1(n,c,flavor):
        a = {'flavor1':0.8,'flavor2':0.8,'flavor3':0.8,'flavor4':0.8,'flavor5':0.8,'flavor6':0.8,'flavor7':0.8,
             'flavor8':0.8,'flavor9':0.8,'flavor10':0.8,'flavor11':0.8,'flavor12':0.8,'flavor13':0.8,'flavor14':0.8,
             'flavor15':0.8}
        zero = [[0]*c for i in range(n)]
        for i in range(n):
            for j in range(c):
                zero[i][j] = a[flavor]
        return zero
    
    
    # 转置
    def trans(A):
        a = [[0] * len(A) for i in range(len(A[0]))]
        for i in range(len(a)):
            for j in range(len(a[0])):
                a[i][j] = A[j][i]
        return a
    
    
   # 1-某一个值
    def sub_1(A):
        for i in range(len(A)):
            for j in range(len(A[0])):
                A[i][j] = 1 - A[i][j]
        return A 
    
    
    # 常数乘一个列表
    def mul_cons(c,A):
        for i in range(len(A)):
            for j in range(len(A[0])):
                A[i][j] = c * A[i][j]
        return A
    
    
    # 列表除一个常数
    def div_cons(A,c):
        for i in range(len(A)):
            for j in range(len(A[0])):
                A[i][j] = A[i][j]/c
        return A
    
    
    # MSE
    def MSE(A,B):
        C = subtract(A,B)
        for i in range(len(C)):
            C[i] = C[i]**2
        return mean(C)

    # 用平均值代替离群值
    def outlier_find(pre_data,flavor):
        a = {'flavor1':[2.0,1.0],'flavor2':[2.0,1.0],'flavor3':[2.0,1.0],'flavor4':[2.0,1.0],'flavor5':[2.0,1.0],
             'flavor6':[2.0,1.0],'flavor7':[2.0,1.0],'flavor8':[2.0,1.0],'flavor9':[2.0,1.0],'flavor10':[2.0,1.0],
             'flavor11':[2.0,1.0],'flavor12':[2.0,1.0],'flavor13':[2.0,1.0],'flavor14':[2.0,1.0],'flavor15':[2.0,1.0]}
        for i in pre_data:
            mean_list = mean(i)
            standard_list = standard(i)
            for k,j in enumerate(i):
                if (j < (mean_list - a[flavor][0]* standard_list) or j > (mean_list + a[flavor][1]* standard_list)):
                    i[k] = mean_list
        return pre_data
    
    
    def flavor_sort_CPU(list): ##这里是以CPU作为基准
        FLAVOR = [[],[],[],[],[],[],[],[],[],[],[],[],[],[],[]]
        for i in list:      
            FLAVOR[15 - int(i[0][6:])].extend(i)
        
    ##删除FLAVOR中的空白[]
        demond_3 = []    
        for j in FLAVOR:
            if len(j):
                demond_3.append(j)
    
        return demond_3 
    
    def flavor_sort_MEM(list):
        MEM = [['flavor15'],['flavor14'], ['flavor12'],['flavor13'],['flavor11'],
       ['flavor9'],['flavor10'],['flavor8'],['flavor6'],['flavor7'],
       ['flavor5'],['flavor3'],['flavor4'],['flavor2'],['flavor1']] 
        for i in list:
            MEM[MEM.index([i[0]])].append(i[1])
        
        demond_3 = []    
        for j in MEM:
            if len(j) > 1:
                demond_3.append(j)
            
        return demond_3
    
# 亚雯版分配空间，计算空间浪费率
    def space_allocate(demond_3_list):
        CPU = {'flavor1':1,
       'flavor2':1,
       'flavor3':1,
       'flavor4':2,
       'flavor5':2,
       'flavor6':2,
       'flavor7':4,
       'flavor8':4,
       'flavor9':4,
       'flavor10':8,
       'flavor11':8,
       'flavor12':8,
       'flavor13':16,
       'flavor14':16,
       'flavor15':16
      }
        RAM = {'flavor1':1,
       'flavor2':2,
       'flavor3':4,
       'flavor4':2,
       'flavor5':4,
       'flavor6':8,
       'flavor7':4,
       'flavor8':8,
       'flavor9':16,
       'flavor10':8,
       'flavor11':16,
       'flavor12':32,
       'flavor13':16,
       'flavor14':32,
       'flavor15':64
      }
        C = int(phy[0])
        R = int(phy[1])
        M = [[C, R]]
        content = [[]]
        for i in demond_3_list:
            a = 0
            for p, k in enumerate(M):
                if CPU[i] <= k[0] and RAM[i] <= k[1]:
                    k[0] = k[0] - CPU[i]
                    k[1] = k[1] - RAM[i]
                    content[p].append(i)
                    a = 1
                    break
            if a!= 1:
                M.append([C,R])
                content.append([])
                M[-1][0] = C - CPU[i]
                M[-1][1] = R - RAM[i]
                content[-1].append(i)
        if optimize == 'CPU':
            s = len(content) - 1 + M[-1][0]/C
        else:
            s = len(content) - 1 + M[-1][1]/R
        return [s,content]


    
    def random_exchange(demond_3_list):
        demond_3_list_new = []
        a_1 = random.randint(0,len(demond_3_list)-1)
        a_2 = random.randint(0,len(demond_3_list)-1)
        t = 'huangweiyawen'
        t = demond_3_list[a_1]
        demond_3_list[a_1] = demond_3_list[a_2]
        demond_3_list[a_2] = t
        for i in demond_3_list:
            demond_3_list_new.append(i)
        return demond_3_list_new    
    
    
    def stimulate_annealing(demond_3_list):
        init_temp = 10
        end_temp = 0.1
        coefficient = 0.99
      
        while init_temp > end_temp:   
            s_0 = space_allocate(demond_3_list)[0]
            demond_3_list_new  = random_exchange(demond_3_list)
            s_1 = space_allocate(demond_3_list_new)[0]
            if s_1 < s_0:
                demond_3_list = []
                for i in demond_3_list_new:
                    demond_3_list.append(i)
            else:
                if math.exp((s_0 - s_1)/ init_temp) > random.random():
                    demond_3_list = []
                    for i in demond_3_list_new:
                        demond_3_list.append(i)
            init_temp  = coefficient * init_temp
        
        return demond_3_list
    
    
    
    
    # 神经网络
    class NeuralNetwork(object):
        def __init__(self, input_nodes, hidden_nodes, output_nodes, learning_rate,flavor):
        # Set number of nodes in input, hidden and output layers.
            self.input_nodes = input_nodes
            self.hidden_nodes = hidden_nodes
            self.output_nodes = output_nodes

        # Initialize weights
            self.weights_input_to_hidden = r_u_1(self.input_nodes,self.hidden_nodes,flavor)
                                       

            self.weights_hidden_to_output = r_u_1(self.hidden_nodes,self.output_nodes,flavor)
                                       
            self.lr = learning_rate
    
                    
    
        def train(self, features, targets):

            n_records = len(features)
            delta_weights_i_h = zeros(self.input_nodes,self.hidden_nodes)
            delta_weights_h_o = zeros(self.hidden_nodes,self.output_nodes)
            for X, y in zip(features, targets):
            #### Implement the forward pass here ####
            ### Forward pass ###
            # TODO: Hidden layer - Replace these values with your calculations.
                hidden_inputs = dot(X,self.weights_input_to_hidden) # signals into hidden layer
                hidden_outputs = activation_function(hidden_inputs) # signals from hidden layer

            # TODO: Output layer - Replace these values with your calculations.
                final_inputs = dot(hidden_outputs,self.weights_hidden_to_output) # signals into final output layer
                final_outputs = final_inputs # signals from final output layer
            
            #### Implement the backward pass here ####
            ### Backward pass ###

            # TODO: Output error - Replace this value with your calculations.
 
                error = subtract([y],final_outputs)
             # Output layer error is the difference between desired target and actual output.
            
            # TODO: Calculate the hidden layer's contribution to the error
                hidden_error = dot(self.weights_hidden_to_output,trans(error))
            
            # TODO: Backpropagated error terms - Replace these values with your calculations.
                output_error_term = error
        
                hidden_error_term = multiply(trans(hidden_error),dot(hidden_outputs,sub_1(hidden_outputs)))
            

            # Weight step (input to hidden)
                delta_weights_i_h = sum_list(delta_weights_i_h,dot(trans(X),hidden_error_term))
            
            
            # Weight step (hidden to output)
                delta_weights_h_o = sum_list(delta_weights_h_o,dot(trans(hidden_outputs),output_error_term))
        
            

        # TODO: Update the weights - Replace these values with your calculations.

            self.weights_hidden_to_output = sum_list(self.weights_hidden_to_output,div_cons(mul_cons(self.lr,delta_weights_h_o),n_records))
        
         # update hidden-to-output weights with gradient descent step
            self.weights_input_to_hidden = sum_list(self.weights_input_to_hidden,div_cons(mul_cons(self.lr,delta_weights_i_h),n_records))

         # update input-to-hidden weights with gradient descent step
 
        def run(self, features):
        
        #### Implement the forward pass here ####
        # TODO: Hidden layer - replace these values with the appropriate calculations.
            hidden_inputs = dot(features,self.weights_input_to_hidden) # signals into hidden layer
            hidden_outputs = activation_function(hidden_inputs) # signals from hidden layer
        
        # TODO: Output layer - Replace these values with the appropriate calculations.
        
            final_inputs = dot(hidden_outputs,self.weights_hidden_to_output) # signals into final output layer
            final_outputs = final_inputs # signals from final output layer 
        
            return final_outputs
       
    
#  训练数据预处理   
    all_flavor_count = count_flavor(ecs_lines,flavor_need)
    
    all_flavor_divide_week = {}
    for flavor in all_flavor_count:
        flavor_divide_week = divide_week(all_flavor_count[flavor],start_train,end_train)
        all_flavor_divide_week[flavor] = flavor_divide_week

# 去除离群点
    for flavor in all_flavor_divide_week:
        all_flavor_divide_week[flavor] = trans(outlier_find(trans(all_flavor_divide_week[flavor]),flavor))
        
# 将数据分成特征和标签
    all_flavor_train = {}
    all_flavor_target = {}
    for flavor in all_flavor_divide_week:
        train = all_flavor_divide_week[flavor]
        train_1 = []
        target_1 = []
        for a in range(len(train)-5):
            cur_record = []
            cur_record1 = []
            for i in range(a,a+5):
                cur_record.extend(train[i])
            for i in range(a+5,a+6):
                cur_record1.extend(train[i])
            train_1.append([cur_record])
            target_1.append(cur_record1)
        all_flavor_train[flavor] = train_1
        all_flavor_target[flavor] = target_1
        
        
    
# 运行神经网络进行训练和测试
    iterations =  {'flavor1':200,'flavor2':200,'flavor3':200,'flavor4':200,'flavor5':200,'flavor6':200,
                  'flavor7':200,'flavor8':200,'flavor9':200,'flavor10':200,'flavor11':200,'flavor12':200,'flavor13':200,
                   'flavor14':200,'flavor15':200}
    learning_rate = {'flavor1':0.9,'flavor2':0.9,'flavor3':0.9,'flavor4':0.9,'flavor5':0.9,'flavor6':0.9,'flavor7':0.9,
                    'flavor8':1.1,'flavor9':0.9,'flavor10':0.9,'flavor11':0.9,'flavor12':0.9,'flavor13':0.9,
                     'flavor14':0.9,'flavor15':0.9}
    hidden_nodes = {'flavor1':10,'flavor2':10,'flavor3':10,'flavor4':10,'flavor5':10,'flavor6':10,'flavor7':10,
                'flavor8':10,'flavor9':10,'flavor10':10,'flavor11':10,'flavor12':10,'flavor13':10,
                    'flavor14':10,'flavor15':10}
    N_i = 35
    output_nodes = 7
    network = {}
    for flavor in all_flavor_target:
        network[flavor] = NeuralNetwork(N_i,hidden_nodes[flavor],output_nodes,learning_rate[flavor],flavor)


# 输入数据：最后6周的训练数据
    full_day_count = {}
    for flavor1 in flavor_need:
        full_day_count_1 = fill_zeros(all_flavor_count[flavor1],start_train,end_train)
        full_day_count[flavor1] = full_day_count_1
    input_1 = {}
    for flavor in full_day_count:
        i_list = []
        for i in full_day_count[flavor]:
            i_list.append(full_day_count[flavor][i])
        input_1[flavor] = i_list
    input_2 = {}
    for flavor in input_1:
        i_list2 = [input_1[flavor][-35:]]
        input_2[flavor] = i_list2
            

    
    all_flavor_prediction = {}
    for flavor in all_flavor_target:
        X = all_flavor_train[flavor]
        y = all_flavor_target[flavor]
        for ii in range(iterations[flavor]):
            network[flavor].train(X,y)
        
        pre_1 = network[flavor].run(input_2[flavor])
        input_3 = [[0]*len(input_2[flavor][0]) for i in range(len(input_2[flavor]))]
        input_3[0][:-7] = input_2[flavor][0][:-7]
        input_3[0][-7:] = pre_1[0]
        pre_2 = network[flavor].run(input_3)
        pre_1[0].extend(pre_2[0])   
        all_flavor_prediction[flavor] = pre_1[0]

    
# 放置部分
    
    start_day = len(contiYMD(end_train, input_lines[-2].split(' ')[0].split('-'))) - 2
    
    end_day = len(contiYMD(end_train, input_lines[-1].split(' ')[0].split('-'))) - 2
    
    all_flavor_prediction_1 = {}
    for i in all_flavor_prediction:
        all_flavor_prediction_1[i] = int(abs(round(sum(all_flavor_prediction[i][start_day: end_day]))))
    
    
    
    list_all = []
    for i in all_flavor_prediction_1:
        list_all.append([i, all_flavor_prediction_1[i]])
    
    
    end = int(input_lines[2].split('\n')[0]) + 3
        
    ##判断是优化CPU还是MEM：
    if input_lines[end + 1].split('\n')[0] == 'CPU':
        demond_3 = flavor_sort_CPU(list_all)    
    else:
        demond_3 = flavor_sort_MEM(list_all)
        
    CPU = {'flavor1':1,
       'flavor2':1,
       'flavor3':1,
       'flavor4':2,
       'flavor5':2,
       'flavor6':2,
       'flavor7':4,
       'flavor8':4,
       'flavor9':4,
       'flavor10':8,
       'flavor11':8,
       'flavor12':8,
       'flavor13':16,
       'flavor14':16,
       'flavor15':16
      }
    RAM = {'flavor1':1,
       'flavor2':2,
       'flavor3':4,
       'flavor4':2,
       'flavor5':4,
       'flavor6':8,
       'flavor7':4,
       'flavor8':8,
       'flavor9':16,
       'flavor10':8,
       'flavor11':16,
       'flavor12':32,
       'flavor13':16,
       'flavor14':32,
       'flavor15':64
      }


    C = int(phy[0])
    R = int(phy[1])
    M = [[C, R]]
    content = [[]]
    for i in demond_3:
        for j in range(i[1]):
            a = 0
            for p, k in enumerate(M):                
                if CPU[i[0]]<= k[0] and RAM[i[0]]<= k[1]: ##注意：这里只能使用and而不可以是&
                    k[0] = k[0] - CPU[i[0]]
                    k[1] = k[1] - RAM[i[0]]
                    content[p].append(i[0])
                    a = 1
                    break   ##continue是跳出单次循环的剩余语句；break是跳出剩余循环；
            if a != 1:
                M.append([C,R])
                content.append([])
                M[-1][0] = C - CPU[i[0]]
                M[-1][1] = R - RAM[i[0]]
                content[-1].append(i[0])
    
    
    demond_4 = []
    for k,i in enumerate(content):
        for j in set(i):
            demond_4.append([k+1, j, i.count(j)])
            
    #获得第一行
    p = 0
    for i in demond_4:
        p = p + i[-1]
    phy_1 = []
    phy_1.append(str(p))

    for i in demond_3:
        phy_1.append(i[0]+ ' '+str(i[1]))
        
    num = demond_4[-1][0]    
    phy_1.append(str(num))


    for i in range(num):
        cur_demond = []
        for demond in demond_4:
            if demond[0] == (i + 1):
                cur_demond.extend(demond[1:])
        cur_str = str(str((i+1)))
        for a in cur_demond:
            cur_str += str(' ')
            cur_str += str(a)            
        phy_1.append(str(cur_str))
            
    phy_1[-(num+2)] = phy_1[-(num+2)]+ '\n'
    result = phy_1 
    
    if ecs_lines is None:
        print ('ecs information is none')
        return result
    if input_lines is None:
        print ('input file information is none')
        return result

    return result





