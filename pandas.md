#### 重复值的处理
pandas中有两个函数是专门用来处理重复值的，分别是duplicated函数和drop_duplicates函数（看看行是否重复，也即看看对应的列值是否相等）
+ **duplicated函数：** 直接使用data.duplicated()这里有两点需要说明：第一，数据表中两个条目间==所有列==的内容都相等时duplicated才会判断为重复值。(Duplicated也可以单独对某一列进行重复值判断)。第二，duplicated支持从前向后(first)，和从后向前(last)两种重复值查找模式。默认是从前向后进行重复值的查找和判断。换句话说就是将后出现的相同条件判断为重复值,并在重复值判断中显示为True。也可指定某列判断是否有缺失值

        data.drop_duplicates(['k1']))
+ **drop_dupulicated:** 用来删除数据中的重复值，判断标准和逻辑与duplicated函数一样。使用drop_duplicates函数后，python将返回一个只包含唯一值的数据表。

#### 空值处理
Pandas中查找数据表中空值的函数有两个，一个是函数isnull，如果是空值就显示True。另一个函数notnull正好相反，如果是空值就显示False。

            loandata.isnull()
    loandata.notnull()

数据预处理中一般可以通过isnull和value_counts函数获得某一属性的控制数据量。 例子如下

        loandata['loan_amnit'].isnull().value_counts()
        
对于空值有3中处理的方法：
+ 第一种是使用fillna函数对空值进行填充，可以选择填充0、均值、中位数、众数或者插值处理（最近邻插补、回归方法）。
+ 第二种方法是使用dropna函数直接将包含空值的数据删除。
+ 第三种直接不处理

        loandata.fillna(0)
        loandata.dropna()
        # 实际用常用相关列或均值进行取代
        loandata['loan_amnt']=loandata['loan_amnt'].fillna(loandata['total_pymnt']-loandata['total_rec_int']).astype(np.int64)
        loandata['annual_inc']=loandata['annual_inc'].fillna(loandata['annual_inc'].mean())
        
#### 数据间的空格
**查看数据中的空格：** 使用value_counts()函数进行统计

        loandata['loan_status'].value_counts()
**处理数据中的空格：** Python中去除空格的方法有三种，第一种是去除数据两边的空格，第二种是单独去除左边的空格，第三种是单独去除右边的空格。

        loandata['term']=loandata['term'].map(str.strip)
    loandata['term']=loandata['term'].map(str.lstrip)
    loandata['term']=loandata['term'].map(str.rstrip)
    loandata['loan_status'] = loandata['loan_status'].map(str.strip)

#### 大小写转换
大小写转换的方法也有三种可以选择，分别为全部转换为大写，全部转换为小写，和转换为首字母大写。

        loandata['term']=loandata['term'].map(str.upper)
    loandata['term']=loandata['term'].map(str.lower)
    loandata['term']=loandata['term'].map(str.title)
    
#### 关键字段内容检查
最后我们还需要对数据表中关键字段的内容进行检查，确保关键字段中内容的统一。主要包括数据是否全部为字符，或数字。下面我们对emp_length列进行检验，此列内容由数字和字符组成，如果只包括字符，说明可能存在问题。下面的代码中我们检查该列是否全部为字符。答案全部为False。

        loandata['emp_length'].apply(lambda x: x.isalpha())

除此之外，还能检验该列的内容是否全部为字母或数字。或者是否全部为数字。
        
        loandata['emp_length'].apply(lambda x: x. isalnum ())

    loandata['emp_length'].apply(lambda x: x. isdigit ())

#### 异常值处理
**发现异常值和极端值的方法：**
+ 对数据进行简单的描述性统计进而查看哪些数据是不合理的。使用describe函数可以生成描述统计结果。其中最常用的是最大值(max)和最小值(min)情况
+ 
        loandata.describe().astype(np.int64).T
+ 箱型图分析 异常值通常被定义为小于Q_L-1.5IQR或大于Q_U+1.5IQR

**异常数据的替换：**
对于异常值数据我们这里选择使用replace函数对loan\_amnt的异常值进行替换，这里替换值选择为loan_amnt的均值。下面是具体的代码和替换结果。

        loandata.replace([100000,36],loandata['loan_amnt'].mean())
        
#### 更改数据格式
更改和规范数据格式，所使用的函数是astype。

下面是更改数据格式的代码。对loan_amnt列中的数据，由于贷款金额通常为整数，因此我们数据格式改为int64。如果是利息字段，由于会有小数，因此通常设置为float64。

    loandata['loan_amnt']=loandata['loan_amnt'].astype(np.int64)
在数据格式中还要特别注意日期型的数据。日期格式的数据需要使用to_datatime函数进行处理。下面是具体的代码和处理后的结果。

    loandata['issue_d']=pd.to_datetime(loandata['issue_d'])
格式更改后可以通过dtypes函数来查看，下面显示了每个字段的数据格式。

    loandata.dtypes

#### 数据分组
对数据进行分组处理，在数据表的open_acc字段记录了贷款用户的账户数量，这里我们可以根据账户数量的多少对用户进行分级，5个账户以下为A级，5-10个账户为B级，依次类推。下面是具体的代码和处理结果。

    bins = [0, 5, 10, 15, 20]
    group_names = ['A', 'B', 'C', 'D']
    loandata['categories'] = pd.cut(loandata['open_acc'], bins, labels=group_names)
==代码解读：== 首先设置了数据分组的依据，然后设置每组对应的名称。最后使用cut函数对数据进行分组并将分组后的名称添加到数据表中。

#### 数据分列
第四步是数据分列，这个操作和Excel中的分列功能很像，在原始数据表中grade列中包含了两个层级的用户等级信息，现在我们通过数据分列将分级信息进行拆分。数据分列操作使用的是split函数，下面是具体的代码和分列后的结果。

    grade_split = pd.DataFrame((x.split('-') for x in loandata.grade),index=loandata.index,columns=['grade','sub_grade'])

完成数据分列操作后，使用merge函数将数据匹配会原始数据表，这个操作类似Excel中的Vlookup函数的功能。通过匹配原始数据表中包括了分列后的等级信息。以下是具体的代码和匹配后的结果。

    loandata=pd.merge(loandata,grade_split,right_index=True, left_index=True)

