import re
import matplotlib.pyplot as plt
from datetime import datetime
"""
说明：
主要比较ALA的有效性
criticalFL 基本参数：
    本次实验，本地训练轮数epoch =5
    训练轮数：200
    是否压缩：否
    异质性：alpha 0.2
ALA
    冻结CNN索引：1

"""
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
        # print(20*"-*")
        rounds = fit_progress_list["round"]
        acc = fit_progress_list["accuracy"]
        plt.plot(rounds, acc, marker='o',label=str(k))
  
    plt.title('round vs. accuracy E=5,compress=False')
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
    

#Fedavg  baseline算法
FedAvg_path = "/root/CriticalFL/log/20240216_095749/fl_log.txt"
rounds1, times1, losses1, accuracies1 = parse_log_file(FedAvg_path)
FedAvg_dict = {"round":rounds1,"loss":losses1,"accuracy":accuracies1}

#Critical 初始算法
#epoches=5;alpha=0.2
#不压缩
CriticalFL_comF_path = '/root/CriticalFL/log/20240303_120423/fl_log.txt'
rounds, times, losses, accuracies = parse_log_file(CriticalFL_comF_path)
criticalFL_comF_dict = {"round":rounds,"loss":losses,"accuracy":accuracies}
#压缩
CriticalFL_comT_path = '/root/CriticalFL/log/20240303_141419/fl_log.txt'
rounds, times, losses, accuracies = parse_log_file(CriticalFL_comT_path)
criticalFL_comT_dict = {"round":rounds,"loss":losses,"accuracy":accuracies}

#Critical+ALA 初始算法
#epoches=5;alpha=0.2
# 不压缩
critical_ala_comF_path = '/root/CriticalFL/log/20240302_144629/fl_log.txt'
rounds2, times2, losses2, accuracies2 = parse_log_file(critical_ala_comF_path)
critical_ala_comF_dict = {"round":rounds2,"loss":losses2,"accuracy":accuracies2}
# 压缩
critical_ala_comT_path = '/root/CriticalFL/log/20240303_151509/fl_log.txt'
rounds2, times2, losses2, accuracies2 = parse_log_file(critical_ala_comT_path)
critical_ala_comT_dict = {"round":rounds2,"loss":losses2,"accuracy":accuracies2}




fit_dict = {}
fit_dict["FedAvg"] = FedAvg_dict
fit_dict["criticalFL_comF"] = criticalFL_comF_dict
fit_dict["criticalFL_comT"] = criticalFL_comT_dict
fit_dict["critical_ala_comF"] = critical_ala_comF_dict
fit_dict["critical_ala_comT"] = critical_ala_comT_dict


# 绘制 accuracy 随着 duration 变化的图表，每隔1个元素取一个
compare_plot_acc_vs_round(fit_dict,save_path='./criticalALA_alpha02_e5_comFT.png')