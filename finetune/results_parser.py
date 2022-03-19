import re
import numpy as np
import argparse
import json

def get_results(type, cmpr_file_permission):
    train_loss = []
    test_loss = []
    test_acc = []

    if type == "seen":
        file_name = args.seen
    elif type == "unseen":
        file_name = args.unseen
    else:
        print("non correct type")

    pattern = '[-+]?\d+\.\d*,|[-+]?\d+\.\d*e-?\d+,'
    with open(file_name, "r") as a_file:
        for line in a_file:
            tmp = re.findall(pattern, line)
            tmp = [re.sub(r',', '', match) for match in tmp]
            train_loss.append(float(tmp[0]))
            test_loss.append(float(tmp[1]))
            test_acc.append(float(tmp[2]))

    with open(cmp_results_file_name, cmpr_file_permission) as f:
        print(f"{type}_train_loss_avg: ", round(np.mean(train_loss), 4), file=f)
        print(f"{type}_test_loss_avg: ", round(np.mean(test_loss), 4), file=f)
        print(f"{type}_test_acc_avg: ", round(np.mean(test_acc), 4), file=f)
        print(f"{type}_test_acc_std: ", round(np.std(test_acc), 4), file=f)
        print("", file=f)

    return test_acc

def get_diff_results(seen_acc_lst, unseen_acc_lst):
    diff = []
    for i in range(len(seen_acc_lst)):
        diff.append(round(seen_acc_lst[i] - unseen_acc_lst[i], 4))
    with open(cmp_results_file_name, 'a') as f:
        print("seen:   ", seen_acc_lst, file=f)
        print("unseen: ", unseen_acc_lst, file=f)
        print("diff: ", diff, file=f)
        print("diff_mean (expl): ", round(np.mean(diff), 4), file=f)
        print("diff_std: ", round(np.std(diff), 4), file=f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--seen", help="path to seen results file")
    parser.add_argument("-u", "--unseen", help="path to unseen results file")
    parser.add_argument("--results_path", default='./', help="path to folder for results. i.e: experiment/results/")
    args = parser.parse_args()

    cmp_results_file_name = args.results_path + " - comparing_results.txt"
    seen_acc_lst = get_results("seen", 'w')
    uneen_acc_lst = get_results("unseen", 'a')
    get_diff_results(seen_acc_lst, unseen_acc_lst)



