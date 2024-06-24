import re
import matplotlib.pyplot as plt
from datetime import datetime


import re

def extract_values_from_log(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    pattern = re.compile(r'Train Round: (\d+), Time: (.*?), Participants: (\d+),train_loss: (.*?),atest_acc:(.*?), ptest_acc:(.*?),center_loss:(.*?),center_acc:(.*?)\n')

    extracted_values = []
    for line in lines:
        match = pattern.search(line)
        if match:
            round_num = int(match.group(1))
            time_stamp = match.group(2)
            participants = int(match.group(3))
            teloss = float(match.group(4))
            atest_acc = float(match.group(5))
            ptest_acc = float(match.group(6))
            center_loss = float(match.group(7))
            center_acc = float(match.group(8))

            extracted_values.append({
                'Train Round': round_num,
                'Time': time_stamp,
                'Participants': participants,
                'teloss': teloss,
                'atest_acc': atest_acc,
                'ptest_acc': ptest_acc,
                'center_loss': center_loss,
                'center_acc': center_acc
            })
    return extracted_values


import matplotlib.pyplot as plt

def plot_ptest_acc(extracted_values):
    # 提取 Train Round 和 ptest_acc 数据
    rounds = [entry['Train Round'] for entry in extracted_values]
    ptest_acc = [entry['ptest_acc'] for entry in extracted_values]

    # 绘制折线图
    plt.figure(figsize=(10, 6))
    plt.plot(rounds, ptest_acc, marker='o', linestyle='-')
    plt.title('ptest_acc vs Train Round')
    plt.xlabel('Train Round')
    plt.ylabel('ptest_acc')
    plt.grid(True)
    plt.savefig('./results_list/ptest_acc_vs_Train_Round.png')
    plt.show()


# fedavg_fit_progress_listDuration
def compare_plot_acc_vs_round(fit_dict,save_path=None):
    for k in fit_dict:
        fit_progress_list = fit_dict[k]
        
        rounds = fit_progress_list["round"]
        acc = fit_progress_list["accuracy"]
        plt.plot(rounds, acc,label=str(k))
  
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

 # fedavg_fit_progress_listDuration
def compare_plot_loss_vs_round(fit_dict,save_path=None):
    for k in fit_dict:
        fit_progress_list = fit_dict[k]
        rounds = fit_progress_list["round"]
        acc = fit_progress_list["loss"]
        plt.plot(rounds, acc,label=str(k))
  
    plt.title('round vs. loss')
    plt.xlabel('round')
    plt.ylabel('loss')
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

import matplotlib.pyplot as plt
from datetime import datetime

def plot_accuracy_over_time(fit_dict,save_path=None):

    for k in fit_dict:
        fit_progress_list = fit_dict[k]
        # 将时间字符串转换为datetime对象
        start_time = datetime.strptime(fit_progress_list["time"][0], "%Y-%m-%d %H:%M:%S")
        time_values = [(datetime.strptime(entry, "%Y-%m-%d %H:%M:%S") - start_time).total_seconds() / 60 for entry in fit_progress_list["time"]]
        acc = fit_progress_list["accuracy"]

        # plt.figure(figsize=(10, 6))
        plt.plot(time_values, acc, label=str(k))
    plt.xlabel('Time (minutes since start)')
    plt.ylabel('Accuracy')
    plt.title('Accuracy Over Time')
    plt.legend()
    if save_path:
        plt.savefig(save_path)
    else:
        # 显示图形
        plt.show()   

def plot_multiple_logs(log_file_paths):
    plt.figure(figsize=(10, 6))

    for file_path in log_file_paths:
        print(file_path)
        values = extract_values_from_log(file_path)
        rounds = [entry['Train Round'] for entry in values]
        ptest_acc = [entry['ptest_acc'] for entry in values]

        plt.plot(rounds, ptest_acc, marker='o', linestyle='-', label=file_path)

    plt.title('ptest_acc vs. Train Round (Multiple Logs)')
    plt.xlabel('Train Round')
    plt.ylabel('ptest_acc')
    plt.grid(True)
    plt.legend()
    
    # 保存为图片
    plt.savefig('./results_list/Train_Round.png')
    plt.show()



if __name__=="__main__":
    # 使用示例

    ditto_alpha01_path = '../log/20240322_041201/fl_log.txt'
    fedavg_path = "../log/20240322_053636/fl_log.txt"
    ALA_path = "../log/20240322_054009/fl_log.txt"
    # extracted_values = extract_values_from_log(ditto_alpha01_path)
    # plot_ptest_acc(extracted_values)
    # 调用函数并传入多个日志文件路径
    log_file_paths = [ditto_alpha01_path, fedavg_path, ALA_path]  # 替换为你的日志文件路径列表
    plot_multiple_logs(log_file_paths)
    

 