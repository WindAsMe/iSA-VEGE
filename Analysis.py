from scipy.stats.stats import mannwhitneyu
import numpy as np
from os import path
import csv
import matplotlib.pyplot as plt


def draw(x_VEGE, VEGE_obj, x_iVEGE, iVEGE_obj, f, Dim):
    this_path = path.realpath(__file__)
    plt.semilogy(x_VEGE, VEGE_obj, label="VEGE")
    plt.semilogy(x_iVEGE, iVEGE_obj, label="iSA-VEGE")
    font_title = {'size': 18}
    font = {'size': 16}
    plt.title('$f_' + '{' + str(f) + '}$', font_title)
    plt.xlabel('Fitness evaluation times', font)
    plt.ylabel('Objective value', font)

    plt.legend()
    plt.savefig(path.dirname(this_path) + '/pic/' + 'f' + str(f) + "_" + str(Dim), dpi=750)
    plt.show()


def open_csv(path):
    data = []
    with open(path) as f:
        f_csv = csv.reader(f)
        for row in f_csv:
            d = []
            for s in row:
                d.append(float(s))
            data.append(d)
    return np.array(data)


def ave(obj):
    means = []
    for i in range(len(obj[0])):
        means.append(np.mean(obj[:, i]))
    return means


this_path = path.dirname(path.abspath(__file__))
func_nums = [1]
func_nums.extend(list(range(3, 31)))
Dims = [50]
for func_num in func_nums:

    for Dim in Dims:
        FEs = 500 * Dim
        path_VEGE = this_path + "\\VEGE_Data\\F" + str(func_num) + "_" + str(Dim) + "D.csv"
        path_iVEGE = this_path + "\\iVEGE_Data\\F" + str(func_num) + "_" + str(Dim) + "D.csv"
        VEGE_obj = open_csv(path_VEGE)
        iVEGE_obj = open_csv(path_iVEGE)

        final_VEGE = VEGE_obj[:, len(VEGE_obj[0])-1]
        final_iVEGE = iVEGE_obj[:, len(iVEGE_obj[0])-1]
        print("f: ", func_num, "Dim: ", Dim)
        print(np.mean(final_VEGE))
        print(np.mean(final_iVEGE))
        print(mannwhitneyu(final_VEGE, final_iVEGE))
        # VEGE_ave = ave(VEGE_obj)
        # iVEGE_ave = ave(iVEGE_obj)
        # x_VEGE = np.linspace(0, FEs, len(VEGE_ave))
        # x_iVEGE = np.linspace(0, FEs, len(iVEGE_ave))
        # f = func_num
        # draw(x_VEGE, VEGE_ave, x_iVEGE, iVEGE_ave, f, Dim)