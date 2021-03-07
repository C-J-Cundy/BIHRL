import numpy as np
import matplotlib.pyplot as plt

def plot_bar():
    """Plot a bar chart showing the probabilities assigned to the 
    various thetas. Get the numbers by running join_count_std
    on the directory with the chains"""
    ns = np.arange(5)
    # results = [[0.0985, 0.0525, 0.0108, 0.00551, 0.00522],
    #            [0.1029, 0.088, 0.095, 0.07118, 0.1007]]
    # errors = [[0.0162, 0.00985, 0.00360, 0.002446, 0.0],
    #           [0.018, 0.028, 0.022, 0.0189, 0.0246]]

    # results = [[0.078, 0.0239, 0.0548, 0.047, 0.0779],
    #            [0.11604, 0.0799, 0.1917, 0.19709, 0.21134]]
    # errors = [[0.0215, 0.00608, 0.00717, 0.0072, 0.0103],
    #           [0.0404, 0.0414, 0.034, 0.05312, 0.036]]

    results = [[0.0679, 0.032, 0.0354, 0.045, 0.0169],
               [0.10892, 0.08872, 0.187, 0.2136, 0.2189]]
    errors = [[0.011, 0.0044, 0.0049, 0.00627, 0.00264],
              [0.0130, 0.0275, 0.043, 0.0425, 0.046]]
                   
    
    fig, ax = plt.subplots()
    colors = ['0.15', 'indianred']
    le = ['BIRL', 'BIHRL']
    width = 0.4
    ax.bar(ns + width*0, results[0], width, color=colors[0],
           label=le[0], yerr=errors[0], capsize=10, edgecolor='k',
           linewidth=1)
    ax.bar(ns + width*1, results[1], width, color=colors[1],
           label=le[1], yerr=errors[1], capsize=10, hatch="//\\\\",
           edgecolor='k', linewidth=1)
    ax.legend(loc='upper left')
    ax.set_ylabel(r'$P(\theta|\mathcal{T})$')
    ax.set_xlabel(r'$n$')
    ax.set_xticks(ns + 1*width / 2)
    ax.set_xticklabels(['1', '2', '3', '4', '5'])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()    
    fig.savefig('./plots/bar_newest_bhirl_irl_given_omega', dpi=1000)
    plt.show()

def plot_theta_margs(marg_dirs1, marg_dirs2, betas=[0.1, 0.5, 1.2, 2]):
    thetas  = [[-1]*5]
    colors = ['indianred', '0.2']
    fig, ax = plt.subplots()
    for theta in thetas:
        probs1 = []
        errs1 = []
        probs2 = []
        errs2 = []
        for index, marg in enumerate(marg_dirs1):
            probs1.append(float(join_count_std(marg, theta)[0]))
            errs1.append(float(join_count_std(marg, theta)[1]))
            print "For BHIRL theta = {}, marg {} have {} with uncertainty {}".format(
                theta, betas[index], probs1[index], join_count_std(marg, theta)[1])
        for index, marg in enumerate(marg_dirs2):
            probs2.append(float(join_count_std(marg, theta)[0]))
            errs2.append(float(join_count_std(marg, theta)[1]))
            print "For BIRL theta = {}, marg {} have {} with uncertainty{}".format(
                theta, betas[index], probs2[index], join_count_std(marg, theta)[1])            
        (plt_line, cap, barlin) = ax.errorbar(betas[:len(marg_dirs1)],
                                               probs1, yerr=errs1, linestyle='-', marker='o',
                                               linewidth=1, markersize=8, color=colors[0],
                                               label=r"$P_{BIHRL}(\theta|\beta, O_a)$",
                                               capsize=10)
        plt_line2, cap2, barlin2 = ax.errorbar(betas[:len(marg_dirs2)],
                                                probs2, yerr=errs2, linestyle='--', marker='o',
                                                linewidth=1, markersize=8, color=colors[1],
                                                label=r"$P_{BIRL}(\theta|\beta, O_a)$",
                                                capsize=10)

        cap[0].set_markeredgewidth(0.5)
        cap2[0].set_markeredgewidth(0.5)
        #plt.grid(True, which='both')    
        #plt.minorticks_on()
        plt.xlabel(r"$\beta$", size=20)                                                    
        plt.ylabel(r"$P(\theta)$", size=20)                      #
        plt.legend()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.get_xaxis().tick_bottom()
        ax.get_yaxis().tick_left()    
        fig.savefig('./plots/line_graph_best_options', dpi=1000)        
        plt.show()

def join_count_std(directory, theta):
    files = os.listdir(directory)
    probs = []
    num = 0
    for f in files:
        probs.append(float(count_partial_theta(np.load(directory + '/' + f), theta))
                     /np.shape(np.load(directory + '/' + f))[1])
        num += 1
    print probs
    return np.mean(probs), np.std(probs) / np.sqrt(num)

def count_partial_theta(margs, theta, size=5):
    count = 0
    for n in range(np.shape(margs)[1]):
        tmp = list(margs[:size, n])
        flag = True
        for p in theta:
            if p in tmp:
                del(tmp[tmp.index(p)])
            else:
                flag = False                
                break
        if flag:
            count += 1
    return count


plot_theta_margs( ['./data_newest/line_graph/b_o_0.1', './data_newest/line_graph/b_o_0.3', './data_newest/line_graph/b_o_0.5', './data_newest/line_graph/b_o_0.8', 
                   './data_newest/line_graph/b_o_1.2', './data_newest/line_graph/b_o_1.6', './data_newest/line_graph/b_o_2'],
                  ['./data_newest/line_graph/b_0.1', './data_newest/line_graph/b_0.3', './data_newest/line_graph/b_0.5', './data_newest/line_graph/b_0.8', 
                   './data_newest/line_graph/b_1.2', './data_newest/line_graph/b_1.6', './data_newest/line_graph/b_2'],                          
                  [0.1, 0.3, 0.5, 0.8, 1.2, 1.6, 2, 1.6])

        
