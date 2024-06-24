from Sims import *
# from Sims_other import *
from Settings import *
import logging 
from datetime import datetime  # Add this import


class FL_Proc:
    def __init__(self, configs):
        self.DataName = configs["dname"]
        self.ModelName = configs["mname"]
        self.NClients = configs["nclients"] 
        self.PClients = configs["pclients"]  
        self.IsIID = configs["isIID"]
        self.Alpha = configs["alpha"]
        self.Aug = configs["aug"]
        self.MaxIter = configs["iters"]
        self.LR = configs["learning_rate"]
        self.Normal = configs["normal"]
        self.Algo = configs["algorithm"]  
        self.Optmzer = configs["optimizer"]  
        self.FixLR = configs["fixlr"]
        self.WDecay = configs["wdecay"]
        self.DShuffle = configs["data_shuffle"]
        self.BatchSize = configs["batch_size"]
        self.Epoch = configs["epoch"]
        self.GlobalLR = configs["global_lr"]
        self.UseCP = configs["critical"]
        self.FIM = configs["fim"]
        self.CThresh = configs["CThresh"]
        self.SOM = configs["server_optim"]
        self.Ratios = {}
        self.RandNum = configs["rand_num"]
        self.CPR = configs["compression"]
        self.GModel = load_Model(self.ModelName, self.DataName)
        self.Server = None
        self.Clients = {}
        self.ClientLoaders = None
        self.TrainLoader = None
        self.TestLoader = None
        self.LogStep = configs["log_step"]
        
        #存放每个客户端的测试精度和样本数（用整个数据进行）
        self.alldata_total_acc = []
        self.alldata_samples = []   
        #存放每个客户端的测试精度和样本数（用各自客户端数据进行）
        self.total_acc = []
        self.part_samples = []  
        #记录train的loss变化
        self.train_loss_list = []
        self.train_samples = []


        #是否要使用ALA模块
        self.ALA = configs["ALA"]
         #client 更新冻结层索引
        self.layerIndex = Configs["layer_index"]
        #服务端下发时，是否压缩        
        self.SCPR = configs["SeverCompression"]
        self.rand_percent = configs["rand_percent"]
        self.Depochs = Configs["Dynamic_epochs"] 
        self.topk = Configs["topk"]
        
        self.updateIDs = []
        for i in range(self.PClients):
            self.updateIDs.append(i)
        self.Detection = CPCheck(self.NClients, self.PClients, alpha=self.Alpha, threshold=self.CThresh, dataname=self.DataName)
        self.Selection = RandomGet(self.NClients)
        self.TrainRound = 0
        
        # 创建以当前时间命名的文件夹
        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_folder = os.path.join("./fedcea_log", current_time)
        self.logfile = log_folder
        os.makedirs(log_folder, exist_ok=True)

        # 在文件夹中创建日志文件
        log_file = os.path.join(log_folder, 'fl_log.txt')
        logging.basicConfig(filename=log_file, level=logging.INFO, format='%(asctime)s - %(message)s')
        logging.info("config:%s",configs)

    
    def get_train_datas(self):
        self.ClientLoaders,self.client_testloaders, self.TrainLoader, self.TestLoader, Stat = get_loaders(self.DataName, self.NClients, self.IsIID,self.Alpha, self.Aug, False, False,self.Normal, self.DShuffle, self.BatchSize)

    def logging(self):
        #这个是聚合后在服务端的评估
        cenloss, cenaccu = self.Server.evaluate(self.TestLoader)

        if sum(self.train_samples)<=0:
            tloss = 0.1
            atest_acc =0.1
            ptest_acc =0.1
        else:
            tloss = sum(self.train_loss_list)*1.0 / sum(self.train_samples)
            # atest_acc = sum(stats[2])*1.0 / sum(stats[1])
            atest_acc = sum(self.alldata_total_acc)*1.0 / sum(self.alldata_samples)
            ptest_acc = sum(self.total_acc)*1.0 / sum(self.part_samples)
        print(f"atest_acc:{atest_acc}, ptest_acc:{ptest_acc},trainLoss:{tloss}")
        current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        logging.info(f'Train Round: {self.TrainRound}, Time: {current_time}, Participants: {len(self.updateIDs)},' 
                    f'train_loss: {tloss},atest_acc:{atest_acc}, ptest_acc:{ptest_acc},center_loss:{cenloss},center_acc:{cenaccu}')

        # logging.info(f'Train Round: {self.TrainRound}, teloss: {teloss}, teaccu: {teaccu}')


    def main(self):

        print("log save to file:",str(self.logfile)) 
        self.get_train_datas()
        self.Server = Server_Sim(self.TrainLoader, self.GModel, self.LR, self.WDecay, self.FixLR, self.DataName)
        NumCount = 0
        print("开始模拟客户端！")
        #self.NClients 模拟客户端数量128
        for c in range(self.NClients):
            # print("clinet:",c)
            # for batch_id, (inputs, targets) in enumerate(self.ClientLoaders[c]):
            #     inputs, targets = inputs.to(device), targets.to(device)
            #     print("targets:",targets)
            # print(self.ClientLoaders[c])
            if self.ALA:
                if self.SCPR:
                    self.Clients[c] = FedALA_SCPR_Client_Sim(self.layerIndex,c,self.ClientLoaders[c], self.GModel, self.LR, self.WDecay, self.Epoch,self.FixLR, self.Optmzer)
                else:
                    self.Clients[c] = FedALA_Client_Sim(self.rand_percent,self.layerIndex,c,self.ClientLoaders[c], self.GModel, self.LR, self.WDecay, self.Epoch,self.FixLR, self.Optmzer,self.Depochs,self.topk)
            else:
                self.Clients[c] = Client_Sim(self.ClientLoaders[c], self.GModel, self.LR, self.WDecay, self.Epoch,self.FixLR, self.Optmzer,self.Depochs)
            self.Selection.register_client(c)

        IDs = []
        for c in range(self.NClients):
            IDs.append(c)

        NumPartens = self.PClients
        DetStep = 2
        LStep = 0

        self.logging()
        CLP = 1
        
        for It in range(self.MaxIter):
            #每一轮开始的时候初始化
            #存放每个客户端的测试精度和loss（用整个数据进行）
            self.alldata_total_acc = []
            self.alldata_samples = []   
            #存放每个客户端的测试精度和loss（用各自客户端数据进行）
            self.total_acc = []
            self.part_samples = []  
            self.train_loss_list = []
            self.train_samples = []

            self.TrainRound = It + 1
            print(f"第 {self.TrainRound} 轮开始训练！")
            if (It + 1) % DetStep == 0:
                CLP = 0
                GetNorms = []
                for ky in self.updateIDs:
                    # 获取到本轮的梯度平均值*学习率
                    GetNorms.append(self.Clients[ky].gradnorm)

                if self.UseCP:
                    self.Detection.recvInfo(GetNorms)
                    # 返回真实的参与客户端数量，CLP = 0 表示不在关键期，CLP = 1 表示在关键
                    NumPartens,CLP = self.Detection.WinCheck(len(self.updateIDs))
                    
           # 如果不用CriticalFL方法更新，而是用随机更新，则是执行下面的程序
            if self.UseCP == False and self.RandNum == True:
                P1 = 0.5
                P2 = 0.5 
                prob = np.random.rand()
                if prob <= P1:
                    NumPartens = self.PClients + int(self.PClients)
                if prob >= P2:
                    NumPartens = self.PClients - int(self.PClients / 2)

            # 选择对应客户端数量的参与者
            print("参与客户端数量：",NumPartens)
            updateIDs = self.Selection.select_participant(NumPartens)
            if self.SCPR == True and self.ALA==True:
                GlobalParms,GModels = self.Server.getKParas(self.layerIndex)
            else:
                GlobalParms,GModels = self.Server.getParas()
            # GlobalParms,GModels = self.Server.getParas()
            LrNow = self.Server.getLR()
            TransLens = []
            TransParas = []
            TransVecs = []
            for ky in updateIDs:
                if self.GlobalLR:
                    self.Clients[ky].updateLR(LrNow)
                    
                # 全局模型下发并更新
                if self.ALA==True:
                    self.Clients[ky].local_initialization(self.SCPR,GModels)
                else:
                    self.Clients[ky].updateParas(GlobalParms)
                
                Nvec = self.Clients[ky].selftrain()

                # 这里是用整个测试集，输入到每个客户端进行一个测试，可能需要时间长一点
                At_acc,AT_ns = self.Clients[ky].evaluate(self.TestLoader) 
                # 这里每个客户端都有自己的测试集进行实验，输入到每个客户端进行一个测试
                Pt_acc,PT_ns = self.Clients[ky].evaluate(self.client_testloaders[ky]) 

                train_loss,train_ns = self.Clients[ky].evaluate_trainLoss()

                self.train_loss_list.append(train_loss)
                self.train_samples.append(train_ns)
                self.alldata_total_acc.append(At_acc)
                self.alldata_samples.append(AT_ns)
                self.total_acc.append(Pt_acc)
                self.part_samples.append(PT_ns)

                ParasNow = self.Clients[ky].getParas()
                if self.CPR and CLP == 1:
                    ParasNow = self.Clients[ky].getKParas()
                # 纪录下当前每个客户端的数据集大小、模型参数、输出
                LenNow = self.Clients[ky].DLen
                TransLens.append(LenNow)
                TransParas.append(ParasNow)
                TransVecs.append(Nvec)
            
            TauEffs = []
            SLen = np.sum(TransLens)
            for k in range(len(TransLens)):
                TauEffs.append(TransLens[k] / SLen * TransVecs[k])
            TauEff = np.sum(TauEffs)
            
            for k in range(len(TransLens)):
                GPara = TransParas[k]
                GLen = TransLens[k] / SLen
                GNvec = TauEff / TransVecs[k]
                self.Server.recvInfo(GPara, GLen, GNvec)
            self.Server.aggParas(self.SOM)
            
            if self.Optmzer == "VRL":
                GlobalParms = self.Server.getParas()
                for ky in updateIDs:
                    self.Clients[ky].updateParas(GlobalParms)
                    LSteps = self.Clients[ky].local_steps
                    self.Clients[ky].optimizer.update_delta(LSteps)
            self.updateIDs = updateIDs
            
            # Logging
            # if (It + 1) % self.LogStep == 0:
            self.logging()
        print("log save to file:",str(self.logfile))    



if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = '1'

    Configs = {}
    # Configs['dname'] = "cifar100"
    Configs['dname'] = "cifar10"
    # Configs['dname'] = "fmnist"
    Configs["mname"] = "alex"
    # Configs["mname"] = "vgg"
    # Configs["mname"] = "resnet"
    # Configs['dname'] = "shake"
    # Configs["mname"] = "lstm"

    Configs["algorithm"] = "CriticalFL"
    # Configs["algorithm"] = "fedavg"
    Configs['nclients'] = 128
    Configs['pclients'] = 16
    Configs["learning_rate"] = 0.01
    Configs["critical"] = True
    Configs["compression"] = True
    Configs["normal"] = True
    Configs["fixlr"] = False
    Configs["global_lr"] = True
    Configs["aug"] = False
    Configs["data_shuffle"] = True
    Configs["fim"] = False
    Configs['isIID'] = False
    Configs["rand_num"] = False
    Configs["epoch"] = 2
    Configs["batch_size"] = 16
    Configs["iters"] = 200
    Configs["log_step"] = 1
    Configs["wdecay"] = 1e-5
    Configs["CThresh"] = 0.01
    Configs["optimizer"] = "SGD" #"SGD", "VRL","FedProx","FedNova","ditto"
    Configs["server_optim"] = None # "Adam","Adag","Yogi"

    Configs["alpha"] = 0.1
    Configs["layer_index"] = 1
    #这两个是配套的
    Configs["ALA"] = True
    Configs["SeverCompression"] = False # 用处不大
    # rand_percent 随机采样率作为训练w的值
    Configs["rand_percent"] = 0.8
    Configs["Dynamic_epochs"] = False
    Configs["topk"] = 80 #客户端训练完后，上传最重要的20%参数

 
    FLSim = FL_Proc(Configs) 
    FLSim.main()
                



