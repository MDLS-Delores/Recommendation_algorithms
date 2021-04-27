from tqdm import tqdm
import numpy as np
import torch
import pandas as pd
import os
from sklearn.metrics import roc_auc_score
from sklearn import preprocessing

# 指定使用第几块GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class FactorizationMachine(torch.nn.Module):
    def __init__(self,features,embedding_dim):
        '''
        features: [n1,n2,...nn], 包含所有特征的维度的list
        embedding_dim: int, 代表隐向量维度k
        '''
        super().__init__() # 表示继承父类
        self.k = embedding_dim
        # 注意这里使用 list 的时候一定要用torch.nn.ModuleLsit() 初始化，才能将 vocabulary_dict 的参数传入模型训练
        # 交叉项的词典，通过label查embedding向量
        self.embedding_dict = torch.nn.ModuleList()
        # 线性项的词典
        self.weight = torch.nn.ModuleList()
        for i in range(len(features)):
            # 使用 torch.nn.Embedding 构建 ni * k 大小的 vocabulary_dict
            self.embedding_dict.append(torch.nn.Embedding(features[i],embedding_dim))
            self.weight.append(torch.nn.Embedding(features[i],1))
        self.sigmoid = torch.nn.Sigmoid()


    def forward(self,x):
        '''
        输入 x 为 batch_size 维的特征向量，内容为每个特征的label
        '''
        sums = 0
        squared = 0
        linear = 0
        for i in range(x.size(dim=1)):
            sums += self.embedding_dict[i](x[:,i])
            squared += torch.pow(self.embedding_dict[i](x[:,i]),2)
            linear += self.weight[i](x[:,i])
        sum_squared = torch.sum(torch.pow(sums,2),dim=1) # 和的平方
        squared_sum = torch.sum(squared,dim=1) # 平方的和
        output = linear.squeeze(1) + 0.5 * (sum_squared - squared_sum)
        return self.sigmoid(output)

class LoaderData(torch.utils.data.Dataset):
    def __init__(self,dataset_path):
        super().__init__() # 表示继承父类
        data = pd.read_csv(dataset_path,sep=',',engine='c',header=None).to_numpy()
        # items 表示读到的特征，不能要最后一列的评分数据，以及dataframe第一列的序号数据
        self.items = data[:,1:-1].astype(int)
        # 对读入data的最后一列的评分数据处理
        self.targets = self.__preprocess__target(data[:,-1]).astype(np.float32)
        # 统计每个特征的类别数
        self.field_dim = np.max(self.items,axis=0) + 1


    # 获取读到的数据长度
    def __len__(self):
        return self.targets.shape[0]

    # 根据index获取一个样本对
    def __getitem__(self,index):
        return self.items[index], self.targets[index]

    # 对评分数据进行处理
    def __preprocess__target(self,target):
        target[target<3] = 0
        target[target>=3] = 1
        return target

def preprocess_data():
    '''
    处理movielens-1M数据，拿出'userid','movieid','gender','age','occupation','zipcode','rating'
    '''
    r_title = ['userid','movieid','rating','timestamp']
    r_data = pd.read_csv('/workspace/mdls_test/ml-1m/ratings.dat',sep='::',engine='python',header=None,names=r_title)
    u_title = ['userid','gender','age','occupation','zipcode']
    u_data = pd.read_csv('/workspace/mdls_test/ml-1m/users.dat',sep='::',engine='python',header=None,names=u_title)

    u_data.replace('M',0,inplace=True)
    u_data.replace('F',1,inplace=True)

    # 自己定义fit的类别
    le = preprocessing.LabelEncoder()
    le.fit([1,18,25,35,45,50,56])
    u_data['age'] = le.transform(u_data['age'])

    # 取zipcode前三位，定位大的位置
    u_data['zipcode'] = u_data['zipcode'].apply(lambda x:x[:3])

    # 合数据
    data = pd.merge(u_data,r_data,on='userid')

    data['zipcode'] = le.fit_transform(data['zipcode'])
    data2 = data[['userid','movieid','gender','age','occupation','zipcode','rating']]
    data2.to_csv('/workspace/mdls_test/ml-1m/label_encode2.dat',header=None)

    # 划分训练集和测试集
    title = ['userid','movieid','gender','age','occupation','zipcode','rating']
    all_data = pd.read_csv('/workspace/mdls_test/ml-1m/label_encode2.dat',sep=',',engine='python',header=None,names=title)

    # 随机采样
    rand_train = all_data.sample(frac=0.8)
    rand_test = all_data[~all_data.index.isin(rand_train.index)]

    rand_train.to_csv('/workspace/mdls_test/ml-1m/label_encode2.randtrain',header=None)
    rand_test.to_csv('/workspace/mdls_test/ml-1m/label_encode2.randtest',header=None)


# 读数据
train_input = LoaderData('/workspace/mdls_test/ml-1m/label_encode2.randtrain')
test_input = LoaderData('/workspace/mdls_test/ml-1m/label_encode2.randtest')
features = train_input.field_dim
embedding_dim = 10
BATCH_SIZE = 256

# 训练集分批处理
traindata = torch.utils.data.DataLoader(
	dataset = train_input, # 需要是 TensorDataset 格式
	batch_size = BATCH_SIZE, # 分批大小
	shuffle=True # 随机打乱
)

# 测试题分批处理
testdata = torch.utils.data.DataLoader(
	dataset = test_input, # 需要是 TensorDataset 格式
	batch_size = BATCH_SIZE, # 分批大小
	shuffle=True # 随机打乱
)

model = FactorizationMachine(features,embedding_dim)
model = model.to(device)
# 变成二分类问题，因此使用 BCEloss 而不是 MSEloss
lossfunc = torch.nn.BCELoss()
# SGD 下降速度很慢，可能是因为没有初始化参数，选用Adam效果变好
optimizer = torch.optim.Adam(model.parameters(), lr = 0.0005)

for epoch in range(20):
    # 训练
    model.train()
    all_train_loss = 0
    train_rmse = 0
    for step, (batch_x,targets) in enumerate(tqdm(traindata)):
        batch_x = batch_x.to(device)
        targets = targets.to(device)
        optimizer.zero_grad()
        outputs = model(batch_x)
        trainloss = lossfunc(outputs, targets)
        trainloss.backward()
        optimizer.step()
        all_train_loss += trainloss.item()
        train_rmse = train_rmse + torch.sum(torch.pow((targets-outputs),2))
    print("trainng:",epoch,"trainloss:",all_train_loss/(step+1),"train_rmse:",(train_rmse.item()/train_input.__len__()))

    # 测试
    model.eval()
    test_rmse = 0
    all_test_loss = 0
    all_targets = []
    all_outputs = []
    for step,(batch_x,targets) in enumerate(tqdm(testdata)):
        batch_x = batch_x.to(device)
        targets = targets.to(device)
        optimizer.zero_grad()
        outputs = model(batch_x)
        testloss = lossfunc(outputs, targets)
        testloss.backward()
        optimizer.step()
        all_test_loss += testloss.item()
        test_rmse = test_rmse + torch.pow((targets-outputs),2).sum()
        all_targets = np.append(all_targets,targets.cpu().detach().numpy())
        all_outputs = np.append(all_outputs,outputs.cpu().detach().numpy())
    auc = roc_auc_score(all_targets,all_outputs)
    print("testing:",epoch,"testloss:",all_test_loss/(step+1),"test_rmse:",(test_rmse.item()/test_input.__len__()),"AUC:",auc)