import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import argparse
from sklearn.manifold import TSNE
from mpl_toolkits.mplot3d import Axes3D


def parse_arg():

    parser = argparse.ArgumentParser()

    parser.add_argument('-input_path', type=str, default='log')
    parser.add_argument('-output_path', type=str, default='log')
    parser.add_argument('-note', type=str, default='')
    parser.add_argument('-signal', type=int, default=2)
    parser.add_argument('-onehot', type=bool, default=False)
    args = parser.parse_args()

    return args


def read_train_line(line):
    raw_data = line.split(' ')
    return float(raw_data[1])


def read_eval_block(block):
    if not block:
        return []
    sig_row_list = []
    sig_col_list = []
    ac_row_list = []
    ac_col_list = []
    for line in block:
        ep, sig_row, sig_col, ac_row, ac_col = read_eval_line(line)
        sig_row_list.append(sig_row)
        sig_col_list.append(sig_col)
        ac_row_list.append(ac_row)
        ac_col_list.append(ac_col)
    return ep, sig_row_list, sig_col_list, ac_row_list, ac_col_list


def read_eval_line(line):
    raw_data = line.split('|')
    raw_data[-1] = raw_data[-1].split(' ')[0][:-1]
    # print(raw_data)
    ep = int(raw_data[0][6:][:-1])
    sig_row = []
    sig_col = []
    temp = raw_data[1][1:][:-1]
    temp = temp.strip().split(', ')
    for elem in temp:
        sig_row.append(float(elem.strip()))
    temp = raw_data[2][1:][:-1]
    temp = temp.strip().split(', ')
    for elem in temp:
        sig_col.append(float(elem.strip()))
    # print(raw_data)
    ac_row = int(raw_data[3])
    ac_col = int(raw_data[4])

    return ep, sig_row, sig_col, ac_row, ac_col


if __name__ == '__main__':

    args = parse_arg()
    input_path = args.input_path + args.note + '/'
    output_path = args.output_path + args.note + '/'

    train_curve = []
    eval_block = []
    tsne = TSNE(n_components=2, random_state=0)

    def visualize(sig, ac_row, ac_col, save_path):
        if not list(np.array(sig).shape)[1] == 2:
            trans_sig = tsne.fit_transform(np.array(sig))
        else:
            trans_sig = np.array(sig)
        plt.figure(figsize=(6, 5))
        colors = ['r', 'g', 'b', 'c']
        ac_joint = np.array(ac_row) * 2 + np.array(ac_col)
        portion = [0, 0, 0, 0]
        for i in range(len(sig)):
            plt.scatter(trans_sig[i, 0], trans_sig[i, 1], c=colors[ac_joint[i]])
            portion[ac_joint[i]] += 1 / len(sig)
        plt.title('r: %.3f, g: %.3f, b: %.3f, c: %.3f' % (portion[0], portion[1], portion[2], portion[3]))
        plt.savefig(save_path)
        plt.cla()
        return portion

    def visualize_3d(sig, ac_row, ac_col, save_path):
        tsne = TSNE(n_components=3, random_state=0)
        if not list(np.array(sig).shape)[1] == 3:
            sig = tsne.fit_transform(np.array(sig))
        else:
            sig = np.array(sig)
        fig = plt.figure(figsize=(6, 5))
        fig = fig.add_subplot(111, projection='3d')
        colors = ['r', 'g', 'b', 'c']
        ac_joint = np.array(ac_row) * 2 + np.array(ac_col)
        portion = [0, 0, 0, 0]
        for i in range(len(sig)):
            # fig.scatter(sig[i, 0], sig[i, 1], sig[i, 2], c=colors[ac_joint[i]])
            fig.scatter(sig[i, 1], sig[i, 0], sig[i, 2], c=colors[ac_joint[i]])
            portion[ac_joint[i]] += 1 / len(sig)
        # fig.title('r: %.3f, g: %.3f, b: %.3f, c: %.3f' % (portion[0], portion[1], portion[2], portion[3]))
        fig.set_xlabel('sig_1')
        fig.set_ylabel('sig_0')
        fig.set_zlabel('sig_2')
        plt.savefig(save_path)
        # fig.cla()
        return portion

    def visualize_signal(signal, ac_row, ac_col, save_path):
        plt.figure(figsize=(6, 5))
        colors = ['r', 'g', 'b', 'c']
        for i in range(len(signal)):
            plt.scatter(signal[i][0], i, c = colors[2*ac_row[i] + ac_col[i]])
        plt.title('r:0|0, g:0|1, b:1|0, c:1|1')
        plt.savefig(save_path)
        plt.cla()

    def visualize_2signals(signal, ac_row, ac_col, save_path):
        plt.figure(figsize=(6, 5))
        colors = ['r', 'g', 'b', 'c']
        for i in range(len(signal)):
            plt.scatter(signal[i][0], signal[i][1], c=colors[2*ac_row[i] + ac_col[i]])
        plt.title('r:0|0, g:0|1, b:1|0, c:1|1')
        plt.savefig(save_path)
        plt.cla()

    def visualize_onehot(signal, ac_row, ac_col, save_path):
        plt.figure(figsize=(6, 5))
        colors = ['r', 'g', 'b', 'c']
        signal = tsne.fit_transform(np.array(signal))
        for i in range(len(signal)):
            plt.scatter(signal[i][0], i, c=colors[2 * ac_row[i] + ac_col[i]])
        plt.title('r:0|0, g:0|1, b:1|0, c:1|1')
        plt.savefig(save_path)
        plt.cla()


    with open(input_path + 'results.txt', 'r', encoding='utf8') as f:
        count = 0
        for line in f:
            if 'Train' in line:
                train_curve.append(read_train_line(line))
                if count:
                    eval_data = read_eval_block(eval_block)
                    if eval_data:
                        ep, sig_row, sig_col, ac_row, ac_col = eval_data
                        save_path = output_path+'%i.jpg' % ep
                        print(ep)

                        if args.onehot:
                            visualize_onehot(sig_row, ac_row, ac_col, save_path)
                        elif args.signal == 1:
                            visualize_signal(sig_row, ac_row, ac_col, output_path+'%i.jpg' % ep)
                            # visualize_2signals(sig_row, ac_row, ac_col, output_path+'%i.jpg' % ep)
                        else:
                            ac_portion = visualize(sig_row, ac_row, ac_col, output_path+'%i.jpg' % ep)
                            print(ac_portion)

                        # print(sig_col[:10])
                        # print(ac_col[:10])
                        # ac_portion = visualize_3d(sig_row, ac_row, ac_col, output_path+'%i.jpg' % ep)

                    count = 0
                    eval_block = []
            if 'Eval' in line:
                eval_block.append(line)
                count += 1
            if len(train_curve) and not len(train_curve) % 500:
                plt.plot(range(len(train_curve)), np.array(train_curve))
                plt.savefig(output_path + 'train_curve.jpg')
                plt.cla()
                plt.close()

    plt.plot(range(len(train_curve)), np.array(train_curve))
    plt.savefig(output_path+'train_curve.jpg')
    plt.cla()
    plt.close()