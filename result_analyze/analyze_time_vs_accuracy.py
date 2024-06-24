import re
import matplotlib.pyplot as plt
from datetime import datetime

"""
说明：
主要比较ALA的有效性
criticalFL 基本参数：
    本次实验，本地训练轮数epoch = 5
    训练轮数：200
    是否压缩：否
    异质性：alpha 0.1
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


def compare_plot_acc_vs_time(log_file_paths,log_labels,save_path=None):
    time_pattern = re.compile(r'Time: (\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})')
    teaccu_pattern = re.compile(r'teaccu: (\d+\.\d+)')

    # plt.figure(figsize=(10, 6))
    L = 0
    for log_file_path in log_file_paths:
        label_ = log_labels[L]
        L = L+1
        timestamps = []
        teaccu_values = []
        print(log_file_path)
        with open(log_file_path, 'r') as file:
            initial_time = None
            for line in file:
                time_match = time_pattern.search(line)
                teaccu_match = teaccu_pattern.search(line)

                if time_match and teaccu_match:
                    timestamp_str = time_match.group(1)
                    teaccu_value = float(teaccu_match.group(1))

                    timestamp = datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S")

                    if initial_time is None:
                        initial_time = timestamp

                    time_difference_minutes = (timestamp - initial_time).total_seconds() / 60.0

                    timestamps.append(time_difference_minutes)
                    teaccu_values.append(teaccu_value)

        plt.plot(timestamps, teaccu_values, marker='o', label=f'{label_}')

    plt.xlabel('Time (minutes)')
    plt.ylabel('Teaccu')
    plt.title('Teaccu Variation Over Time')
    plt.legend()
    plt.grid(True)
    if save_path:
        plt.savefig(save_path)
    else:
        # 显示图形
        plt.show()

if __name__=="__main__":

    #Fedavg  baseline算法
    #epoches=5;alpha=0.1???
    # FedAvg_path = "/root/CriticalFL/log/20240216_095749/fl_log.txt"
    # rounds1, times1, losses1, accuracies1 = parse_log_file(FedAvg_path)
    # FedAvg_dict = {"round":rounds1,"loss":losses1,"accuracy":accuracies1}

    #Critical 初始算法
    #epoches=5;alpha=0.1
    #压缩
    CriticalFL_comT_path = '/root/CriticalFL/log/20240304_141400/fl_log.txt'
    # rounds, times, losses, accuracies = parse_log_file(CriticalFL_comT_path)
    # criticalFL_comT_dict = {"round":rounds,"loss":losses,"accuracy":accuracies}

    #Critical+ALA 初始算法
    # 压缩
    critical_ala_comT_path = '/root/CriticalFL/log/20240304_114747/fl_log.txt'
    # rounds2, times2, losses2, accuracies2 = parse_log_file(critical_ala_comT_path)
    # critical_ala_comT_dict = {"round":rounds2,"loss":losses2,"accuracy":accuracies2}


    # 请将下面的路径替换为你的日志文件路径
    log_labels = ["CriticalFL", "CriticalFL_ALA"]

    log_directory = [CriticalFL_comT_path, critical_ala_comT_path]
    compare_plot_acc_vs_time(log_directory,log_labels,"CLPALA_alpha01_e5_TimeAcc.png")
