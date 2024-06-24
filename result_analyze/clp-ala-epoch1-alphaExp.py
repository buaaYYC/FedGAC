import re
import matplotlib.pyplot as plt
from datetime import datetime


import re

def extract_values_from_log(log_file_path):
    with open(log_file_path, 'r') as file:
        log_content = file.read()

    # Define the regular expression pattern to extract values
    pattern = r'Train Round: (\d+), Time: (.*), Participants: (\d+),train_loss: (.*),atest_acc:(.*), ptest_acc:(.*),center_loss:(.*),center_acc:(.*)'

    # Find all matches in the log content
    matches = re.findall(pattern, log_content)

    # Extracted values
    extracted_values = []

    for match in matches:
        round_num, timestamp, participants, train_loss, atest_acc, ptest_acc, center_loss, center_acc = match
        extracted_values.append({
            'Train Round': int(round_num),
            'Time': timestamp,
            'Participants': int(participants),
            'train_loss': float(train_loss),
            'atest_acc': float(atest_acc),
            'ptest_acc': float(ptest_acc),
            'center_loss': float(center_loss),
            'center_acc': float(center_acc),
        })

    return extracted_values

def old_extract_values_from_log(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    # pattern = re.compile(r'Train Round: (\d+), Time: (\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}), Participants: (\d+),train_loss: ([\d.]+),atest_acc: ([\d.]+),ptest_acc: ([\d.]+)')
    pattern = re.compile(r'Train Round: (\d+), Time: (.*?), Participants: (\d+),train_loss: (.*?),atest_acc:(.*?), ptest_acc:(\d+\.\d+)\n')

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

            extracted_values.append({
                'Train Round': round_num,
                'Time': time_stamp,
                'Participants': participants,
                'train_loss': teloss,
                'atest_acc': atest_acc,
                'ptest_acc': ptest_acc
            })
    # print(extracted_values)
    return extracted_values

# fedavg_fit_progress_listDuration
def compare_plot_acc_vs_round(fit_dict,save_path=None):
    for k in fit_dict:
        fit_progress_list = fit_dict[k]
        # print(fit_progress_list)
        try:
            rounds = fit_progress_list["round"]
        except:
            print(fit_progress_list)
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

# 使用示例
# plot_accuracy_over_time(extracted_data)
def save_dict(log_path):
    # fedavg_path = '../log/20240310_073807/fl_log.txt'
    tmp_dict = extract_values_from_log(log_path)
    if len(tmp_dict)==0:
        tmp_dict = old_extract_values_from_log(log_path)
    # print(tmp_dict)
    train_rounds = [entry['Train Round'] for entry in tmp_dict]
    # print(train_rounds)
    losses = [entry['train_loss'] for entry in tmp_dict]
    atacc_values = [entry['atest_acc'] for entry in tmp_dict]
    ptacc_values = [entry['ptest_acc'] for entry in tmp_dict]
    Time = [entry['Time'] for entry in tmp_dict]

    local_dict = {"round":train_rounds,"loss":losses,"accuracy":ptacc_values,"time":Time}      
    allData_dict = {"round":train_rounds,"loss":losses,"accuracy":atacc_values,"time":Time}  
    return local_dict,allData_dict



def calculate_elapsed_time(fit_dict,save_path=None):
    label_ = []
    time_cost = []
    for k in fit_dict:
        local_dict = fit_dict[k] 
        local_dict = local_dict["time"]
        # 将字符串转换为 datetime 对象
        start_time_str = local_dict[0]
        end_time_str = local_dict[-1]
        start_time = datetime.strptime(start_time_str, "%Y-%m-%d %H:%M:%S")
        end_time = datetime.strptime(end_time_str, "%Y-%m-%d %H:%M:%S")

        # 计算时间差
        elapsed_time = end_time - start_time
        # 转换为分钟数
        elapsed_minutes = elapsed_time.total_seconds() / 60.0
        label_.append(str(k))
        time_cost.append(elapsed_minutes)
    # 创建柱状图
    plt.bar(label_, time_cost, color='blue')
    # 旋转标签，调整图像尺寸和布局
    # plt.xticks(rotation='vertical')
    # plt.gcf().set_size_inches(len(label_), 6)
    # plt.tight_layout()

    # 添加标题和标签
    plt.title('Time Consumption of Experiments')
    plt.xlabel('Experiments')
    plt.ylabel('Time (minutes)')
    if save_path:
        plt.savefig(save_path)
    else:
        # 显示图形
        plt.show()  

    

if __name__=="__main__":

    clpala_alpha05_path = '../log/20240311_104321/fl_log.txt'
    clpala_alpha05_dict,_ = save_dict(clpala_alpha05_path) 

    clpala_alpha01_path = '../log/20240310_074555/fl_log.txt'
    clpala_alpha01_dict,_ = save_dict(clpala_alpha01_path) 

    clpala_alpha03_path = '../log/20240312_033542/fl_log.txt'
    clpala_alpha03_dict,_ = save_dict(clpala_alpha03_path) 
    
    ditto_alpha01_path = '../log/20240322_041201/fl_log.txt'
    ditto_alpha01_dict = save_dict(ditto_alpha01_path) 

    fit_128_dict = {}

    fit_128_dict["clpala_alpha01"] = clpala_alpha01_dict
    fit_128_dict["clpala_alpha03"] = clpala_alpha03_dict
    fit_128_dict["clpala_alpha05"] = clpala_alpha05_dict
    # fit_128_dict["ditto_alpha01_dict"] = ditto_alpha01_dict

    # 绘制 accuracy 随着 duration 变化的图表，每隔1个元素取一个
    # compare_plot_acc_vs_round(fit_all_dict,save_path='./clp-ala-all-acc.png')
    compare_plot_acc_vs_round(fit_128_dict,save_path='./results_list/diffMethods-acc.png')
    # compare_plot_loss_vs_round(fit_dict,save_path='./clp-ala-train-loss.png')
    # plot_accuracy_over_time(fit_dict,save_path="./local_time_acc.png")
    #对比不同方法的耗时
    # calculate_elapsed_time(fit_dict,save_path='./fedavg-clp-ala-local-cosTime.png')