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

if __name__=="__main__":

    #Critical 初始算法
    #epoches=5;alpha=0.1
    #压缩
    CriticalFL_comT_path = '../log/20240304_141400/fl_log.txt'
    rounds, times, losses, accuracies = parse_log_file(CriticalFL_comT_path)
    criticalFL_comT_dict = {"round":rounds,"loss":losses,"accuracy":accuracies}

    #Critical+ALA 初始算法
    #epoches=5;alpha=0.1
    # 压缩
    critical_ala_comT_path = '../log/20240304_114747/fl_log.txt'
    rounds2, times2, losses2, accuracies2 = parse_log_file(critical_ala_comT_path)
    critical_ala_comT_dict = {"round":rounds2,"loss":losses2,"accuracy":accuracies2}

    #Critical+ALA 初始算法 sever 压缩 layerindex = 1
    #epoches=5;alpha=0.1
    # 压缩
    critical_ala_SCPR1_path = '../log/20240307_022837/fl_log.txt'
    rounds2, times2, losses2, accuracies2 = parse_log_file(critical_ala_SCPR1_path)
    critical_ala_SCPR1_dict = {"round":rounds2,"loss":losses2,"accuracy":accuracies2}
    #Critical+ALA 初始算法 sever 压缩 layerindex = 2
    #epoches=5;alpha=0.1
    # 压缩/home/csis/CriticalFL/log/20240307_031036/fl_log.txt
    critical_ala_SCPR2_path = '../log/20240307_031036/fl_log.txt'
    rounds2, times2, losses2, accuracies2 = parse_log_file(critical_ala_SCPR2_path)
    critical_ala_SCPR2_dict = {"round":rounds2,"loss":losses2,"accuracy":accuracies2}





    fit_dict = {}
    fit_dict["criticalFL_comT"] = criticalFL_comT_dict
    fit_dict["critical_ala_comT"] = critical_ala_comT_dict
    fit_dict["critical_ala_SCPR1_dict"] = critical_ala_SCPR1_dict
    fit_dict["critical_ala_SCPR2_dict"] = critical_ala_SCPR2_dict



    # 绘制 accuracy 随着 duration 变化的图表，每隔1个元素取一个
    compare_plot_acc_vs_round(fit_dict,save_path='./clp-ala-scpr.png')
