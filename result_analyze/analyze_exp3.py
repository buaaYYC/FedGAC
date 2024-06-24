import re
import matplotlib.pyplot as plt
from datetime import datetime

def parse_log_file(file_path):
    rounds = []
    times = []
    losses = []
    accuracies = []

    with open(file_path, 'r') as file:
        for line in file:
            match = re.search(r'Train Round: (\d+), Time: (.*), Participants: \d+, teloss: ([\d.]+), teaccu: ([\d.]+)', line)
            if match:
                round_num = int(match.group(1))
                time_str = match.group(2)
                loss = float(match.group(3))
                accuracy = float(match.group(4))

                rounds.append(round_num)
                times.append(datetime.strptime(time_str, '%Y-%m-%d %H:%M:%S'))
                losses.append(loss)
                accuracies.append(accuracy)

    return rounds, times, losses, accuracies

import matplotlib.pyplot as plt
# fedavg_fit_progress_listDuration
def compare_plot_acc_vs_round(fit_dict,save_path=None):
    for k in fit_dict:
        fit_progress_list = fit_dict[k]
        
        rounds = fit_progress_list["round"]
        acc = fit_progress_list["accuracy"]
        plt.plot(rounds, acc, marker='o',label=str(k))
  
    plt.title('round vs. accuracy')
    plt.xlabel('round')
    plt.ylabel('accuracy')
    # 添加图例
    plt.legend()
    # 显示网格
    plt.grid(True)
    # 显示图形
    # 保存图形
    if save_path:
        plt.savefig(save_path)
    else:
        # 显示图形
        plt.show()
    


#Critical 初始算法
CriticalFL_path = '/root/CriticalFL/log/20240216_095815/fl_log.txt'
rounds, times, losses, accuracies = parse_log_file(CriticalFL_path)
criticalFL_alpha02_dict = {"round":rounds,"loss":losses,"accuracy":accuracies}

FedAvg_path = "/root/CriticalFL/log/20240216_095749/fl_log.txt"
rounds1, times1, losses1, accuracies1 = parse_log_file(FedAvg_path)
FedAvg_alpha02_dict = {"round":rounds1,"loss":losses1,"accuracy":accuracies1}

# 实验1
critical_ala_path = '/root/CriticalFL/log/20240229_113654/fl_log.txt'
rounds2, times2, losses2, accuracies2 = parse_log_file(critical_ala_path)
critical_ala_02_dict = {"round":rounds2,"loss":losses2,"accuracy":accuracies2}

# 实验2 
# CLP不压缩；冻结层数降低 2 ；w训练权重次数最大5000 ; 权重batch_size = 16
critical_ala_5000_path = '/root/CriticalFL/log/20240229_120540/fl_log.txt'
rounds5, times5, losses5, accuracies5 = parse_log_file(critical_ala_5000_path)
critical_ala_02_5000_dict = {"round":rounds5,"loss":losses5,"accuracy":accuracies5}
# 实验3 
# CLP不压缩；冻结层数降低 1 ；w训练权重次数最大1000 ; 权重batch_size = 16
critical_ala_1000_path = '/root/CriticalFL/log/20240301_104731/fl_log.txt'
rounds5, times5, losses5, accuracies5 = parse_log_file(critical_ala_1000_path)
critical_ala_02_1000_dict = {"round":rounds5,"loss":losses5,"accuracy":accuracies5}
# CLP不压缩；冻结层数降低 1 ；w训练权重次数最大1000 ; 权重batch_size = 64
critical_ala_1000_bs64_path = '/root/CriticalFL/log/20240301_031434/fl_log.txt'
rounds5, times5, losses5, accuracies5 = parse_log_file(critical_ala_1000_bs64_path)
critical_ala_02_1000_bs64_dict = {"round":rounds5,"loss":losses5,"accuracy":accuracies5}


fit_dict = {}
fit_dict["criticalFL_0.2"] = criticalFL_alpha02_dict
fit_dict["fedavg_0.2"] = FedAvg_alpha02_dict
# fit_dict["critical_ALA_0.2"] = critical_ala_02_dict
# fit_dict["critical_ala_02_5000_dict"] = critical_ala_02_5000_dict
fit_dict["critical_ala_02_1000_bs16_dict"] = critical_ala_02_1000_dict
fit_dict["critical_ala_02_1000_bs64_dict"] = critical_ala_02_1000_bs64_dict

# 绘制 accuracy 随着 duration 变化的图表，每隔1个元素取一个
compare_plot_acc_vs_round(fit_dict,save_path='./critical_ALA_compared.png')