from Settings import *
from Util import *
from Optim import VRL as OP1
from Optim import FedProx as OP2
from Optim import FedNova as OP3
from ALA import ALA 


class Client_Sim:
    def __init__(self, Loader, Model, Lr, wdecay, epoch=1, fixlr=False, optzer="SGD", Depochs=False):
        self.TrainData = cp.deepcopy(Loader)
        self.DLen = 0
        for batch_id, (inputs, targets) in enumerate(self.TrainData):
            inputs, targets = inputs.to(device), targets.to(device)
            self.DLen += len(inputs)
        self.Model = cp.deepcopy(Model)
        self.Optzer = optzer
        self.Wdecay = wdecay
        self.Epoch = epoch
        self.Mu = 0.001
        self.Round = 0
        self.LR = Lr
        self.decay_step = 10
        self.decay_rate = 0.9
        self.GetGrad = None
        self.optimizer = None
        self.local_steps = 1
        self.optimizer = OP1.VRL(self.Model.parameters(), lr=self.LR, momentum=0.9, weight_decay=self.Wdecay, vrl=True, local=True)
        self.loss_fn = nn.CrossEntropyLoss()
        self.FixLR = fixlr
        self.gradnorm = 0
        self.trainloss = 0
        self.difloss = 0
        self.depochs = Depochs

    def reload_data(self, loader):
        self.TrainData = cp.deepcopy(loader)

    def getParas(self):
        GParas = cp.deepcopy(self.Model.state_dict())
        return GParas
        
    def getKParas(self):
        NP = []
        # 遍历GetGrad中的参数
        for ky in self.GetGrad.keys():
            if "bias" in ky or "weight" in ky:
                GNow = self.GetGrad[ky]
                NP += list(GNow.cpu().detach().numpy().reshape(-1))
        # 取参数绝对值
        NP = np.abs(NP)
        # 计算梯度的80%分位数，作为阈值
        """计算梯度的80%分位数是指将所有梯度值按升序排列，然后找到一个值，使得80%的梯度值都小于或等于该值。"""
        Cut = np.percentile(NP,80)
        
        GParas = cp.deepcopy(self.Model.state_dict())
        # 根据阈值筛选高梯度的参数
        for ky in GParas.keys():
            if "bias" in ky or "weight" in ky:
                if ky in self.GetGrad.keys():
                    GParas[ky] = GParas[ky] * (torch.abs(self.GetGrad[ky]) >= Cut)
        return GParas

    def updateParas(self, Paras):
        self.Model.load_state_dict(Paras)

    def updateLR(self, lr):
        self.LR = lr
        self.decay_rate = 1

    def getLR(self):
        return self.LR

    def compModelDelta(self, model_1, model_2):
        sd1 = model_1.state_dict()
        sd2 = model_2.state_dict()
        Res = cp.deepcopy(model_1)

        for key in sd1:
            sd1[key] = sd1[key] - sd2[key]
        Res.load_state_dict(sd1)
        return Res

    def genState(self,TL):
        Res = self.getParas()
        C = 0
        for ky in Res.keys():
            Res[ky] = TL[C]
            C += 1
        return Res

    def selftrain(self, control_local=None, control_global=None):
        self.Round += 1
        BeforeParas = self.getParas()
        if self.Round % self.decay_step == 0:
            self.LR *= self.decay_rate
        optimizer = None
        if self.Optzer == "SGD":
            optimizer = torch.optim.SGD(self.Model.parameters(), lr=self.LR, momentum=0.9, weight_decay=self.Wdecay)
        if self.Optzer == "FedProx":
            optimizer = OP2.FedProx(self.Model.parameters(), lr=self.LR, momentum=0.9, weight_decay=self.Wdecay, mu = self.Mu)
        if self.Optzer == "FedNova":
            optimizer = OP3.FedNova(self.Model.parameters(), lr=self.LR, momentum=0.9, weight_decay=self.Wdecay)
        
        self.optimizer.param_groups[0]['lr'] = self.LR
        if self.Optzer == "VRL":
            optimizer = self.optimizer
        
        self.gradnorm = 0
        self.trainloss = 0
        self.difloss = 0
        
        SLoss = []
        GNorm = []
        new_loss_fn = nn.CrossEntropyLoss()
        Init_Model = cp.deepcopy(self.Model)
        
        # model_size_mb = get_model_size(self.Model)
        # print(f"client 模型大小: {model_size_mb:.2f} MB")
        
        self.Model.train()
        Local_Steps = 0
        #cifra10
        max_entropy = np.log2(10)
        if self.depochs==True: 
            # 动态调整epochs训练次数，当数据集分布熵越大，质量越高，训练次数也越多，质量差的减少训练次数，保证全局模型的质量。
            data_entropy= evaluate_data_quality_entropy(self.TrainData)
            if data_entropy <max_entropy/3:
                self.Epoch = 1
            elif data_entropy <max_entropy*2/3:
                self.Epoch = 3
            elif data_entropy <max_entropy:
                self.Epoch = 5

        for r in range(self.Epoch):
            sum_loss = 0
            grad_norm = 0
            C = 0
            for batch_id, (inputs, targets) in enumerate(self.TrainData):
                C = C + 1
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = self.Model(inputs)
                optimizer.zero_grad()
                if self.Optzer == "VRL":
                    self.optimizer.zero_grad()
                loss = self.loss_fn(outputs, targets)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.Model.parameters(),10)
                if self.Optzer == "VRL":
                    self.optimizer.step()
                else:
                    optimizer.step()
                temp_norm = 0
                for parms in self.Model.parameters():
                    gnorm = parms.grad.detach().data.norm(2)
                    temp_norm = temp_norm + (gnorm.item()) ** 2
                if grad_norm == 0:
                    grad_norm = temp_norm
                else:
                    grad_norm = grad_norm + temp_norm
                
                newoutputs = self.Model(inputs)
                newloss = new_loss_fn(newoutputs, targets)
                self.difloss = self.difloss + loss.item() - newloss.item()

            SLoss.append(sum_loss / C)
            GNorm.append(grad_norm)
            Local_Steps = C

        self.trainloss = np.mean(SLoss)
        Lrnow = self.getLR()
        self.gradnorm = np.mean(GNorm) * Lrnow
        self.local_steps = Local_Steps * self.Epoch
        
        if self.Optzer == "VRL":
            self.optimizer.update_params()
        NVec = 1
        if self.Optzer == "FedNova":
            NVec = optimizer.local_normalizing_vec
        AfterParas = self.getParas()
        self.GetGrad = minusParas(AfterParas,BeforeParas)
        AfterParas = cp.deepcopy([])
        BeforeParas = cp.deepcopy([])
        return NVec
        
    def evaluate(self, loader=None, max_samples=100000):
 
        self.Model.eval()
        loss, correct, samples, iters = 0, 0, 0, 0
        loss_fn = nn.CrossEntropyLoss()
        if loader == None:
            loader = self.TrainData
        with torch.no_grad():
            for i, (x, y) in enumerate(loader):
                x, y = x.to(device), y.to(device)
                y_ = self.Model(x)
                _, preds = torch.max(y_.data, 1)
                correct += (preds == y).sum().item()
                # loss += loss_fn(y_, y).item()
                # loss += loss_fn(y,y_).item()*y_.shape[0]
                samples += y_.shape[0]
                # iters += 1
                # iters += y_.shape[0]
                if samples >= max_samples:
                    break
        return correct,samples
    def evaluate_trainLoss(self, max_samples=100000):
 
        self.Model.eval()
        loss, correct, samples, iters = 0, 0, 0, 0
        loss_fn = nn.CrossEntropyLoss()
        # if loader == None:
        loader = self.TrainData
        with torch.no_grad():
            for i, (x, y) in enumerate(loader):
                x, y = x.to(device), y.to(device)
                y_ = self.Model(x)
                _, preds = torch.max(y_.data, 1)
                correct += (preds == y).sum().item()
                loss += loss_fn(y_, y).item()*y_.shape[0]
                samples += y_.shape[0]
                # iters += 1
                # iters += y_.shape[0]
                if samples >= max_samples:
                    break
        return loss,samples  
    def fim(self,loader=None):
        if loader == None:
            loader = cp.deepcopy(self.TrainData)
        self.Model.eval()
        Ts = []
        K = 10000
        for i, (x,y) in enumerate(loader):
                x, y = list(x.cpu().detach().numpy()), list(y.cpu().detach().numpy())
                for j in range(len(x)):
                    Ts.append([x[j],y[j]])
                if len(Ts) >= K:
                    break

        TLoader = torch.utils.data.DataLoader(dataset=Ts, batch_size=500, shuffle=False)
        F_Diag = FIM(
            model=self.Model,
            loader=TLoader,
            representation=PMatDiag,
            n_output=10,
            variant="classif_logits",
            device="cuda"
        )
        
        Tr = F_Diag.trace().item()

        return Tr
    
# ditto
class Client_ditto_Sim:
    def __init__(self, Loader, Model, Lr, wdecay, epoch=1, fixlr=False, optzer="SGD", Depochs=False):
        self.TrainData = cp.deepcopy(Loader)
        self.DLen = 0
        for batch_id, (inputs, targets) in enumerate(self.TrainData):
            inputs, targets = inputs.to(device), targets.to(device)
            self.DLen += len(inputs)
        self.Model = cp.deepcopy(Model)
        self.Optzer = optzer
        self.Wdecay = wdecay
        self.Epoch = epoch
        self.Mu = 0.001
        self.Round = 0
        self.LR = Lr
        self.decay_step = 10
        self.decay_rate = 0.9
        self.GetGrad = None
        self.optimizer = None
        self.local_steps = 1
        self.optimizer = OP1.VRL(self.Model.parameters(), lr=self.LR, momentum=0.9, weight_decay=self.Wdecay, vrl=True, local=True)
        self.loss_fn = nn.CrossEntropyLoss()
        self.FixLR = fixlr
        self.gradnorm = 0
        self.trainloss = 0
        self.difloss = 0
        self.depochs = Depochs

    def reload_data(self, loader):
        self.TrainData = cp.deepcopy(loader)

    def getParas(self):
        GParas = cp.deepcopy(self.Model.state_dict())
        return GParas
        
    def getKParas(self):
        NP = []
        # 遍历GetGrad中的参数
        for ky in self.GetGrad.keys():
            if "bias" in ky or "weight" in ky:
                GNow = self.GetGrad[ky]
                NP += list(GNow.cpu().detach().numpy().reshape(-1))
        # 取参数绝对值
        NP = np.abs(NP)
        # 计算梯度的80%分位数，作为阈值
        """计算梯度的80%分位数是指将所有梯度值按升序排列，然后找到一个值，使得80%的梯度值都小于或等于该值。"""
        Cut = np.percentile(NP,80)
        
        GParas = cp.deepcopy(self.Model.state_dict())
        # 根据阈值筛选高梯度的参数
        for ky in GParas.keys():
            if "bias" in ky or "weight" in ky:
                if ky in self.GetGrad.keys():
                    GParas[ky] = GParas[ky] * (torch.abs(self.GetGrad[ky]) >= Cut)
        return GParas

    def updateParas(self, Paras):
        self.Model.load_state_dict(Paras)

    def updateLR(self, lr):
        self.LR = lr
        self.decay_rate = 1

    def getLR(self):
        return self.LR

    def compModelDelta(self, model_1, model_2):
        sd1 = model_1.state_dict()
        sd2 = model_2.state_dict()
        Res = cp.deepcopy(model_1)

        for key in sd1:
            sd1[key] = sd1[key] - sd2[key]
        Res.load_state_dict(sd1)
        return Res

    def genState(self,TL):
        Res = self.getParas()
        C = 0
        for ky in Res.keys():
            Res[ky] = TL[C]
            C += 1
        return Res

    def selftrain(self, control_local=None, control_global=None):
        self.Round += 1
        BeforeParas = self.getParas()
        if self.Round % self.decay_step == 0:
            self.LR *= self.decay_rate
        optimizer = None
        if self.Optzer == "SGD":
            optimizer = torch.optim.SGD(self.Model.parameters(), lr=self.LR, momentum=0.9, weight_decay=self.Wdecay)
        if self.Optzer == "FedProx":
            optimizer = OP2.FedProx(self.Model.parameters(), lr=self.LR, momentum=0.9, weight_decay=self.Wdecay, mu = self.Mu)
        if self.Optzer == "FedNova":
            optimizer = OP3.FedNova(self.Model.parameters(), lr=self.LR, momentum=0.9, weight_decay=self.Wdecay)
        if self.Optzer == "ditto":
            optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, self.Model.parameters()),
                                        lr=self.LR, momentum=self.Mu,
                                        weight_decay=self.Wdecay)
        
        self.optimizer.param_groups[0]['lr'] = self.LR
        if self.Optzer == "VRL":
            optimizer = self.optimizer
        
        self.gradnorm = 0
        self.trainloss = 0
        self.difloss = 0
        
        SLoss = []
        GNorm = []
        new_loss_fn = nn.CrossEntropyLoss()
        Init_Model = cp.deepcopy(self.Model)
        
        # model_size_mb = get_model_size(self.Model)
        # print(f"client 模型大小: {model_size_mb:.2f} MB")
        
        self.Model.train()
        Local_Steps = 0
        #cifra10
        max_entropy = np.log2(10)
        if self.depochs==True: 
            # 动态调整epochs训练次数，当数据集分布熵越大，质量越高，训练次数也越多，质量差的减少训练次数，保证全局模型的质量。
            data_entropy= evaluate_data_quality_entropy(self.TrainData)
            if data_entropy <max_entropy/3:
                self.Epoch = 1
            elif data_entropy <max_entropy*2/3:
                self.Epoch = 3
            elif data_entropy <max_entropy:
                self.Epoch = 5

        for r in range(self.Epoch):
            sum_loss = 0
            grad_norm = 0
            C = 0
            for batch_id, (inputs, targets) in enumerate(self.TrainData):
                C = C + 1
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = self.Model(inputs)
                optimizer.zero_grad()
                if self.Optzer == "VRL":
                    self.optimizer.zero_grad()
                loss = self.loss_fn(outputs, targets)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.Model.parameters(),10)
                if self.Optzer == "VRL":
                    self.optimizer.step()
                else:
                    optimizer.step()
                temp_norm = 0
                for parms in self.Model.parameters():
                    gnorm = parms.grad.detach().data.norm(2)
                    temp_norm = temp_norm + (gnorm.item()) ** 2
                if grad_norm == 0:
                    grad_norm = temp_norm
                else:
                    grad_norm = grad_norm + temp_norm
                
                newoutputs = self.Model(inputs)
                newloss = new_loss_fn(newoutputs, targets)
                self.difloss = self.difloss + loss.item() - newloss.item()

            SLoss.append(sum_loss / C)
            GNorm.append(grad_norm)
            Local_Steps = C

        self.trainloss = np.mean(SLoss)
        Lrnow = self.getLR()
        self.gradnorm = np.mean(GNorm) * Lrnow
        self.local_steps = Local_Steps * self.Epoch
        
        if self.Optzer == "VRL":
            self.optimizer.update_params()
        NVec = 1
        if self.Optzer == "FedNova":
            NVec = optimizer.local_normalizing_vec
        AfterParas = self.getParas()
        self.GetGrad = minusParas(AfterParas,BeforeParas)
        AfterParas = cp.deepcopy([])
        BeforeParas = cp.deepcopy([])
        return NVec
        
    def evaluate(self, loader=None, max_samples=100000):
 
        self.Model.eval()
        loss, correct, samples, iters = 0, 0, 0, 0
        loss_fn = nn.CrossEntropyLoss()
        if loader == None:
            loader = self.TrainData
        with torch.no_grad():
            for i, (x, y) in enumerate(loader):
                x, y = x.to(device), y.to(device)
                y_ = self.Model(x)
                _, preds = torch.max(y_.data, 1)
                correct += (preds == y).sum().item()
                # loss += loss_fn(y_, y).item()
                # loss += loss_fn(y,y_).item()*y_.shape[0]
                samples += y_.shape[0]
                # iters += 1
                # iters += y_.shape[0]
                if samples >= max_samples:
                    break
        return correct,samples
    def evaluate_trainLoss(self, max_samples=100000):
 
        self.Model.eval()
        loss, correct, samples, iters = 0, 0, 0, 0
        loss_fn = nn.CrossEntropyLoss()
        # if loader == None:
        loader = self.TrainData
        with torch.no_grad():
            for i, (x, y) in enumerate(loader):
                x, y = x.to(device), y.to(device)
                y_ = self.Model(x)
                _, preds = torch.max(y_.data, 1)
                correct += (preds == y).sum().item()
                loss += loss_fn(y_, y).item()*y_.shape[0]
                samples += y_.shape[0]
                # iters += 1
                # iters += y_.shape[0]
                if samples >= max_samples:
                    break
        return loss,samples  
    def fim(self,loader=None):
        if loader == None:
            loader = cp.deepcopy(self.TrainData)
        self.Model.eval()
        Ts = []
        K = 10000
        for i, (x,y) in enumerate(loader):
                x, y = list(x.cpu().detach().numpy()), list(y.cpu().detach().numpy())
                for j in range(len(x)):
                    Ts.append([x[j],y[j]])
                if len(Ts) >= K:
                    break

        TLoader = torch.utils.data.DataLoader(dataset=Ts, batch_size=500, shuffle=False)
        F_Diag = FIM(
            model=self.Model,
            loader=TLoader,
            representation=PMatDiag,
            n_output=10,
            variant="classif_logits",
            device="cuda"
        )
        
        Tr = F_Diag.trace().item()

        return Tr
#计算传输模型大小单位 兆M    
def get_model_size(model):
    total_params = sum(p.numel() for p in model.parameters())
    total_size = total_params * 4 / (1024 ** 2)  # Convert to MB
    return total_size

"""

- 评估数据集好坏，计算整个训练数据集中类别分布的熵。-
如果熵值较低，说明数据集中的类别分布相对集中；
而较高的熵值则表示更均匀的类别分布，即数据集更具有多样性。
您可以使用这个熵值来评估数据集的质量，决定是否值得用于训练。
对于熵的范围，它通常在 0 到 log2(N) 之间，数据集为cifra10，则log2（10）= 3.32，
接近 0：示数据集中某几个类别的样本数远远超过其他类别，分布不均匀。
中等值： 熵的中等值（例如，1到2之间）可能表示数据集中的类别分布相对均匀，模型可能需要更全面地学习各个类别的特征。
接近 log2(10)： 如果熵接近 log2(10)，表示数据集中的样本在各个类别上均匀分布，这可能需要更复杂的模型来适应多样性。
"""
def calculate_entropy(labels):
    """
    Calculate the entropy of a set of labels.

    Parameters:
    - labels: Tensor or numpy array containing class labels.

    Returns:
    - entropy: Entropy value.
    """
    unique_labels, label_counts = np.unique(labels, return_counts=True)
    label_probs = label_counts / len(labels)
    entropy = -np.sum(label_probs * np.log2(label_probs))
    return entropy

def evaluate_data_quality_entropy(train_loader):
    """
    Evaluate the quality of the training dataset using entropy.

    Parameters:
    - train_loader: DataLoader for the training dataset.

    Returns:
    - entropy: Entropy value for the class distribution.
    """
    all_labels = []

    # Iterate through the training dataset
    for _, labels in train_loader:
        all_labels.extend(labels.numpy())

    # Calculate entropy of the class distribution
    entropy = calculate_entropy(all_labels)

    return entropy



class FedALA_Client_Sim:
    def __init__(self,rand_percent,layerIndex,cid, Loader, Model, Lr, wdecay, epoch=1, fixlr=False, optzer="SGD",Depochs=False):
        self.TrainData = cp.deepcopy(Loader)
        self.DLen = 0
        for batch_id, (inputs, targets) in enumerate(self.TrainData):
            inputs, targets = inputs.to(device), targets.to(device)
            # print("targets:",targets)
            self.DLen += len(inputs)
        self.Model = cp.deepcopy(Model)
        self.Optzer = optzer
        self.Wdecay = wdecay
        self.Epoch = epoch
        self.Mu = 0.001
        self.Round = 0
        self.LR = Lr
        self.decay_step = 10
        self.decay_rate = 0.9
        self.GetGrad = None
        self.optimizer = None
        self.local_steps = 1
        self.optimizer = OP1.VRL(self.Model.parameters(), lr=self.LR, momentum=0.9, weight_decay=self.Wdecay, vrl=True, local=True)
        self.loss_fn = nn.CrossEntropyLoss()
        self.FixLR = fixlr
        self.gradnorm = 0
        self.trainloss = 0
        self.difloss = 0
        self.depochs = Depochs
        
        # 增加ALA模块
        # =====================================
        self.batch_size = 16 # 这个影响不大
        self.loss = nn.CrossEntropyLoss()
        self.rand_percent = rand_percent
        self.layer_idx = layerIndex
        self.eta = 1.0
        self.ALA = ALA(cid,self.loss, self.TrainData, self.batch_size, 
            self.rand_percent, self.layer_idx, self.eta,device)
        # =====================================


    def reload_data(self, loader):
        self.TrainData = cp.deepcopy(loader)

    def getParas(self):
        GParas = cp.deepcopy(self.Model.state_dict())
        return GParas
        
    def getKParas(self):
        NP = []
        # 遍历GetGrad中的参数
        for ky in self.GetGrad.keys():
            if "bias" in ky or "weight" in ky:
                GNow = self.GetGrad[ky]
                NP += list(GNow.cpu().detach().numpy().reshape(-1))
        # 取参数绝对值
        NP = np.abs(NP)
        # 计算梯度的80%分位数，作为阈值
        """计算梯度的80%分位数是指将所有梯度值按升序排列，然后找到一个值，使得80%的梯度值都小于或等于该值。"""
        Cut = np.percentile(NP,80)
        
        GParas = cp.deepcopy(self.Model.state_dict())
        # 根据阈值筛选高梯度的参数
        for ky in GParas.keys():
            if "bias" in ky or "weight" in ky:
                if ky in self.GetGrad.keys():
                    GParas[ky] = GParas[ky] * (torch.abs(self.GetGrad[ky]) >= Cut)
        return GParas

    def updateParas(self, Paras):
        self.Model.load_state_dict(Paras)
    
    # 全局模型参数与局部模型进行聚合
    def local_initialization(self,SCPR,received_global_model):
        self.ALA.adaptive_local_aggregation(received_global_model, self.Model)


    def updateLR(self, lr):
        self.LR = lr
        self.decay_rate = 1

    def getLR(self):
        return self.LR

    def compModelDelta(self, model_1, model_2):
        sd1 = model_1.state_dict()
        sd2 = model_2.state_dict()
        Res = cp.deepcopy(model_1)

        for key in sd1:
            sd1[key] = sd1[key] - sd2[key]
        Res.load_state_dict(sd1)
        return Res

    def genState(self,TL):
        Res = self.getParas()
        C = 0
        for ky in Res.keys():
            Res[ky] = TL[C]
            C += 1
        return Res

    def selftrain(self, control_local=None, control_global=None):
        self.Round += 1
        BeforeParas = self.getParas()
        if self.Round % self.decay_step == 0:
            self.LR *= self.decay_rate
        optimizer = None
        if self.Optzer == "SGD":
            optimizer = torch.optim.SGD(self.Model.parameters(), lr=self.LR, momentum=0.9, weight_decay=self.Wdecay)
        if self.Optzer == "FedProx":
            optimizer = OP2.FedProx(self.Model.parameters(), lr=self.LR, momentum=0.9, weight_decay=self.Wdecay, mu = self.Mu)
        if self.Optzer == "FedNova":
            optimizer = OP3.FedNova(self.Model.parameters(), lr=self.LR, momentum=0.9, weight_decay=self.Wdecay)
        
        self.optimizer.param_groups[0]['lr'] = self.LR
        if self.Optzer == "VRL":
            optimizer = self.optimizer
        
        self.gradnorm = 0
        self.trainloss = 0
        self.difloss = 0
        
        SLoss = []
        GNorm = []
        new_loss_fn = nn.CrossEntropyLoss()
        Init_Model = cp.deepcopy(self.Model)
        
        # model_size_mb = get_model_size(self.Model)
        # print(f"client 模型大小: {model_size_mb:.2f} MB")
        
        self.Model.train()
        Local_Steps = 0
                #cifra10
        max_entropy = np.log2(10)
        if self.depochs==True: 
            # 动态调整epochs训练次数，当数据集分布熵越大，质量越高，训练次数也越多，质量差的减少训练次数，保证全局模型的质量。
            data_entropy= evaluate_data_quality_entropy(self.TrainData)
            if data_entropy <max_entropy/3:
                self.Epoch = 1
            elif data_entropy <max_entropy*2/3:
                self.Epoch = 3
            elif data_entropy <max_entropy:
                self.Epoch = 5
        print(f"client local epochs:{self.Epoch}")
        for r in range(self.Epoch):
            sum_loss = 0
            grad_norm = 0
            C = 0
            for batch_id, (inputs, targets) in enumerate(self.TrainData):
                C = C + 1
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = self.Model(inputs)
                optimizer.zero_grad()
                if self.Optzer == "VRL":
                    self.optimizer.zero_grad()
                loss = self.loss_fn(outputs, targets)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.Model.parameters(),10)
                if self.Optzer == "VRL":
                    self.optimizer.step()
                else:
                    optimizer.step()
                temp_norm = 0
                for parms in self.Model.parameters():
                    gnorm = parms.grad.detach().data.norm(2)
                    temp_norm = temp_norm + (gnorm.item()) ** 2
                if grad_norm == 0:
                    grad_norm = temp_norm
                else:
                    grad_norm = grad_norm + temp_norm
                
                newoutputs = self.Model(inputs)
                newloss = new_loss_fn(newoutputs, targets)
                self.difloss = self.difloss + loss.item() - newloss.item()

            SLoss.append(sum_loss / C)
            GNorm.append(grad_norm)
            Local_Steps = C

        self.trainloss = np.mean(SLoss)
        Lrnow = self.getLR()
        self.gradnorm = np.mean(GNorm) * Lrnow
        self.local_steps = Local_Steps * self.Epoch
        
        if self.Optzer == "VRL":
            self.optimizer.update_params()
        NVec = 1
        if self.Optzer == "FedNova":
            NVec = optimizer.local_normalizing_vec
        AfterParas = self.getParas()
        self.GetGrad = minusParas(AfterParas,BeforeParas)
        AfterParas = cp.deepcopy([])
        BeforeParas = cp.deepcopy([])
        return NVec
        
    # def evaluate(self, loader=None, max_samples=100000):
 
    #     self.Model.eval()
    #     loss, correct, samples, iters = 0, 0, 0, 0
    #     loss_fn = nn.CrossEntropyLoss()
    #     if loader == None:
    #         loader = self.TrainData
    #     with torch.no_grad():
    #         for i, (x, y) in enumerate(loader):
    #             x, y = x.to(device), y.to(device)
    #             y_ = self.Model(x)
    #             _, preds = torch.max(y_.data, 1)
    #             correct += (preds == y).sum().item()
    #             loss += loss_fn(y_, y).item()
    #             samples += y_.shape[0]
    #             iters += 1
    #             if samples >= max_samples:
    #                 break
    #     return correct / samples, loss / iters
        
    def evaluate(self, loader=None, max_samples=100000):
 
        self.Model.eval()
        loss, correct, samples, iters = 0, 0, 0, 0
        loss_fn = nn.CrossEntropyLoss()
        if loader == None:
            loader = self.TrainData
        with torch.no_grad():
            for i, (x, y) in enumerate(loader):
                x, y = x.to(device), y.to(device)
                y_ = self.Model(x)
                _, preds = torch.max(y_.data, 1)
                correct += (preds == y).sum().item()
                # loss += loss_fn(y_, y).item()
                samples += y_.shape[0]
                # iters += 1
                if samples >= max_samples:
                    break
        return correct ,samples
    def evaluate_trainLoss(self, max_samples=100000):
 
        self.Model.eval()
        loss, correct, samples, iters = 0, 0, 0, 0
        loss_fn = nn.CrossEntropyLoss()
        # if loader == None:
        loader = self.TrainData
        # print("train_loader length :",len(loader))
        with torch.no_grad():
            for i, (x, y) in enumerate(loader):
                x, y = x.to(device), y.to(device)
                y_ = self.Model(x)
                _, preds = torch.max(y_.data, 1)
                correct += (preds == y).sum().item()
                loss += loss_fn(y_, y).item()*y_.shape[0]
                # loss += loss_fn(y,y_).item()*y_.shape[0]
                samples += y_.shape[0]
                # iters += 1
                # iters += y_.shape[0]
                if samples >= max_samples:
                    break
        # print("samples length :",len(sa))
        return loss,samples
    def fim(self,loader=None):
        if loader == None:
            loader = cp.deepcopy(self.TrainData)
        self.Model.eval()
        Ts = []
        K = 10000
        for i, (x,y) in enumerate(loader):
                x, y = list(x.cpu().detach().numpy()), list(y.cpu().detach().numpy())
                for j in range(len(x)):
                    Ts.append([x[j],y[j]])
                if len(Ts) >= K:
                    break

        TLoader = torch.utils.data.DataLoader(dataset=Ts, batch_size=500, shuffle=False)
        F_Diag = FIM(
            model=self.Model,
            loader=TLoader,
            representation=PMatDiag,
            n_output=10,
            variant="classif_logits",
            device="cuda"
        )
        
        Tr = F_Diag.trace().item()

        return Tr


class Server_Sim:
    def __init__(self, Loader, Model, Lr, wdecay=0, Fixlr=False, Dname="cifar10"):
        self.TrainData = cp.deepcopy(Loader)
        self.DLen = 0
        for batch_id, (inputs, targets) in enumerate(self.TrainData):
            inputs, targets = inputs.to(device), targets.to(device)
            # print("数据长度：",len(inputs))
            self.DLen += len(inputs)
        self.Model = cp.deepcopy(Model)
        self.optimizer = torch.optim.SGD(self.Model.parameters(), lr=Lr, momentum=0.9, weight_decay=wdecay)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=1, gamma=0.9)
        self.loss_fn = nn.CrossEntropyLoss()
        self.FixLr = Fixlr
        self.RecvParas = []
        self.RecvLens = []
        self.RecvScale = []
        self.RecvAs = []
        self.LStep = 0
        self.CStep = 0
        self.Eta = 0.01
        self.Beta1 = 0.5
        self.Beta2 = 0.9
        self.Tau = 0.001
        self.Vt = None
        self.Mt = None
        self.Round = 0
        # self.GetGrad = None

    def reload_data(self, loader):
        self.TestData = cp.deepcopy(loader)

    def getParas(self):
        GParas = cp.deepcopy(self.Model.state_dict())
        # 加一个全局模型
        GModel = cp.deepcopy(self.Model)
        return GParas,GModel
    
    def getKParas(self, num_layers=1):  # 设置num_layers为前几层，默认为前3层
        # 获取模型的前几层参数
        GParas = {}
        for i, (ky, val) in enumerate(self.Model.state_dict().items()):
            if i < num_layers:
                GParas[ky] = cp.deepcopy(val)

        # 创建只包含前几层的模型
        GModel = self.createPartialModel(num_layers)

        return GParas, GModel

    def createPartialModel(self, num_layers):
        # 获取模型的前几层，创建新模型
        GModel = cp.deepcopy(self.Model)
        GModel = nn.Sequential(*list(GModel.children())[:num_layers])

        return GModel
   
    
    def getLR(self):
        LR = self.optimizer.state_dict()['param_groups'][0]['lr']               
        return LR

    def updateParas(self, Paras):
        self.Model.load_state_dict(Paras)
        
    def getMinus(self,P1,P2,sign=1):
        Res = cp.deepcopy(P1)
        for ky in Res.keys():
            Mparas = P2[ky] - P1[ky] * sign
            Res[ky] =  Mparas
        return Res
        
    def avgParas(self, Paras, Ps, Scale):
        Res = cp.deepcopy(Paras[0])
        Lens = []
        for i in range(len(Ps)):
            Lens.append(Ps[i] * Scale[i])
        Sum = np.sum(Lens)
        for ky in Res.keys():
            Mparas = 0
            for i in range(len(Paras)):
                Pi = Lens[i] / Sum
                Mparas += Paras[i][ky] * Pi
            Res[ky] = Mparas
        return Res
        
    def avgEleParas(self, Paras, Ps, Scale):
        Res = cp.deepcopy(Paras[0])
        Lens = []
        for i in range(len(Ps)):
            Lens.append(Ps[i] * Scale[i])
        for ky in Res.keys():
            Mparas = 0
            Mask = 0
            for i in range(len(Paras)):
                Mask += ((Paras[i][ky] > 0) + (Paras[i][ky] < 0)) * Lens[i]
                Mparas += Paras[i][ky] * Lens[i]
            Mask = Mask + (Mask == 0) * 0.000001
            Res[ky] = Mparas / Mask
        return Res
    
    def Adagrad(self, Grad):
        for ky in Grad.keys():
            self.Vt[ky] = self.Vt[ky] + 0.25 * Grad[ky] ** 2
    
    def Yogi(self, Grad):
        for ky in Grad.keys():
            Vt = self.Vt[ky]
            self.Vt[ky] = Vt - (1 - self.Beta2) * Grad[ky] ** 2 * torch.sign(Vt - Grad[ky] ** 2)
            
    def Adam(self,Grad):
        for ky in Grad.keys():
            Vt = self.Vt[ky]
            self.Vt[ky] = self.Beta2 * Vt + (1 - self.Beta2) * Grad[ky] ** 2
        

    def aggParas(self,Optim="Yogi"):        
        self.Round += 1
        Disc = 0.9

        GParas = self.avgEleParas(self.RecvParas, self.RecvLens, self.RecvScale)
        if Optim != None and self.Round < 10:
            if self.Vt == None:
                self.Vt = cp.deepcopy(GParas)
                for ky in GParas.keys():
                    G = GParas[ky]
                    Gen = torch.zeros_like(G) + self.Tau**2
                    self.Vt[ky] = Gen
            
            GetGrad = cp.deepcopy(GParas)
            BParas = self.getParas()
            for ky in BParas.keys():
                grad = GParas[ky] - BParas[ky]
                GetGrad[ky] = grad
            
            if Optim == "Adag":
                self.Adagrad(GetGrad)
            if Optim == "Adam":
                self.Adam(GetGrad)
            if Optim == "Yogi":
                self.Yogi(GetGrad)
            
            if self.Mt == None:
                self.Mt = cp.deepcopy(GetGrad)
            
            for ky in self.Mt.keys():
                self.Mt[ky] = self.Mt[ky] * self.Beta1 + GetGrad[ky] * (1 - self.Beta1)
            
            for ky in GetGrad.keys():
                NewGrad = self.Mt[ky] / (torch.sqrt(self.Vt[ky]) + self.Tau)
                In = 0
                if "weight" in ky:
                    In = 1
                if "bias" in ky:
                    In = 1
                if In == 1:
                    Eta = torch.median(torch.sqrt(self.Vt[ky]) + self.Tau)
                    GParas[ky] = BParas[ky] + Eta * NewGrad
            
            self.Eta *= Disc
            self.Eta = max(self.Eta, self.Tau)
        
        self.updateParas(GParas)
        self.RecvParas = []
        self.RecvLens = []
        self.RecvScale = []

        if self.FixLr == False:
            self.optimizer.step()
            self.scheduler.step()

    def recvInfo(self, Para, Len, Scale):
        self.RecvParas.append(Para)
        self.RecvLens.append(Len)
        self.RecvScale.append(Scale)

    def evaluate(self, loader=None, max_samples=100000):
        self.Model.eval()

        loss, correct, samples, iters = 0, 0, 0, 0
        if loader == None:
            loader = self.TrainData
        model_size_mb = get_model_size(self.Model)
        print(f"server 模型大小: {model_size_mb:.2f} MB")
        
        with torch.no_grad():
            for i, (x, y) in enumerate(loader):
                x, y = x.to(device), y.to(device)
                #print(y)
                y_ = self.Model(x)
                _, preds = torch.max(y_.data, 1)
                loss += self.loss_fn(y_, y).item()
                
                correct += (preds == y).sum().item()
                samples += y_.shape[0]
                iters += 1

                if samples >= max_samples:
                    break

        return loss / iters, correct / samples
    
    # def clientSum_evaluate(self, loader=None, max_samples=100000):
    #     # 这里是每个客户端都有对应的测试数据集，他们测试完结果最后平均求和
    #     # 这里计算两个一个是test loss，一个train loss
    #     self.Model.eval()

    #     loss, correct, samples, iters = 0, 0, 0, 0
    #     if loader == None:
    #         loader = self.TrainData
    #     model_size_mb = get_model_size(self.Model)
    #     print(f"server 模型大小: {model_size_mb:.2f} MB")
        
    #     with torch.no_grad():
    #         for i, (x, y) in enumerate(loader):
    #             x, y = x.to(device), y.to(device)
    #             #print(y)
    #             y_ = self.Model(x)
    #             _, preds = torch.max(y_.data, 1)
    #             loss += self.loss_fn(y_, y).item()
                
    #             correct += (preds == y).sum().item()
    #             samples += y_.shape[0]
    #             iters += 1

    #             if samples >= max_samples:
    #                 break

    #     return loss / iters, correct / samples

    def saveModel(self, Path):
        torch.save(self.Model, Path)
        
    def fim(self,loader=None,max_samples=5000):
        if loader == None:
            loader = cp.deepcopy(self.TrainData)
        
        self.Model.eval()
        Ts = []
        K = 10000
        Trs = []
        KLs = []
        samples = 0
        for i, (x,y) in enumerate(loader):
                x, y = list(x.cpu().detach().numpy()), list(y.cpu().detach().numpy())
                for j in range(len(x)):
                    Ts.append([x[j],y[j]])
                if len(Ts) > K:
                    TLoader = torch.utils.data.DataLoader(dataset=Ts,batch_size=500,shuffle=False)
                    F_Diag = FIM(
                        model=self.Model,
                        loader=TLoader,
                        representation=PMatDiag,
                        n_output=10,
                        variant="classif_logits",
                        device="cuda"
                    )
                    Tr = F_Diag.trace().item()
                    Trs.append(Tr)
                    Ts = []
                    
                    Vec = PVector.from_model(self.Model)
                    KL = F_Diag.vTMv(Vec).item()
                    KLs.append(KL)


                samples += len(x)
                if samples >= max_samples:
                    break
                    
        if len(Ts) >= 100:
            TLoader = torch.utils.data.DataLoader(dataset=Ts,batch_size=500,shuffle=False)
            F_Diag = FIM(
                    model=self.Model,
                    loader=TLoader,
                    representation=PMatDiag,
                    n_output=10,
                    variant="classif_logits",
                    device="cuda"
                    )
            Tr = F_Diag.trace().item()
            Trs.append(Tr)

        Tr = np.mean(Trs)
        return Tr







