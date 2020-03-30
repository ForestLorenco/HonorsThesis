import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

#Global path variable names

NOSKIP = "NoSkip"
SKIP = "Skip"
DEULING = "Dueling"

def SvsNS():
    paths = ["NetworkLSTM/", "NetworkNewRew/", "NetworkOldRew/", "Dueling/"]
    names = ["LSTM", "R2 Reward", "R1 Reward", "Dueling"]
    i = 0
    for p in paths:
        #individual
        data_ns = pd.read_csv(NOSKIP+p+"data.csv")
        points_ns = data_ns.values
        

        plt.plot(points_ns, linewidth=.5)
        plt.title("No Skip "+names[i])
        plt.ylabel('Reward')
        plt.xlabel('Games over Time (25 hrs)')
        
        plt.savefig("Plots/"+NOSKIP+p[:-1]+"Graph")
        #plt.show()
        plt.clf()

        data_s = pd.read_csv(SKIP+p+"data.csv")
        points_s = data_s.values
        

        plt.plot(points_s, linewidth=.5)
        plt.title("Skip "+names[i])
        plt.ylabel('Reward')
        plt.xlabel('Games over Time (25 hrs)')
        
        plt.savefig("Plots/"+SKIP+p[7:-1]+"Graph")
        #plt.show()
        plt.clf()

        if p == paths[3]:
            points_s = points_s[0:len(points_ns)+150]
            p = "Network" + p
        #both 
        print(p[7:-1])
        plt.plot(points_s, linewidth=.5, label="Skip")
        plt.plot(points_ns, linewidth=.5, label="No Skip")
        plt.title("Skip vs No Skip of "+p[7:-1])
        plt.ylabel('Reward')
        plt.xlabel('Games over Time (25 hrs)')
        plt.legend(loc="best")
        plt.savefig("Plots/S_vs_NS"+p[7:-1]+"Graph")
        plt.clf()
        i += 1

def LSTMvsFF():
    paths = ["LSTM/", "NewRew/"]
    data_l = pd.read_csv(NOSKIP+paths[0]+"data.csv")
    points_l = data_l.values

    data_ff = pd.read_csv(NOSKIP+paths[1]+"data.csv")
    points_ff = data_ff.values

    plt.plot(points_l, linewidth=.5, label="LSTM")
    plt.plot(points_ff, linewidth=.5, label="Feed Forward")
    plt.title("LSTM vs Feed Forward Network")
    plt.ylabel('Reward')
    plt.xlabel('Games over Time (25 hrs)')
    plt.legend(loc="best")
    plt.savefig("Plots/LSTM_vs_FF_Graph")
    plt.clf()

def duelvsa3c():
    data_d = pd.read_csv(DEULING+"NoSkip/data.csv")
    points_d = data_d.values

    data_a = pd.read_csv(NOSKIP+"OldRew/data.csv")
    points_a = data_a.values

    plt.plot(points_d, linewidth=.5, label="Dueling")
    plt.plot(points_a, linewidth=.5, label="A3C")
    plt.title("Dueling vs A3C No Skip")
    plt.ylabel('Reward')
    plt.xlabel('Games over Time (25 hrs)')
    plt.legend(loc="best")
    plt.savefig("Plots/D_vs_A3C_NoSkipGraph")
    plt.clf()

    data_d = pd.read_csv(DEULING+"Skip/data.csv")
    points_d = data_d.values[0:300]

    data_a = pd.read_csv(SKIP+"OldRew/data.csv")
    points_a = data_a.values

    plt.plot(points_d, linewidth=.5, label="Dueling")
    plt.plot(points_a, linewidth=.5, label="A3C")
    plt.title("Dueling vs A3C Skip")
    plt.ylabel('Reward')
    plt.xlabel('Games over Time (25 hrs)')
    plt.legend(loc="best")
    plt.savefig("Plots/D_vs_A3C_SkipGraph")
    plt.clf()

if __name__ == "__main__":
    SvsNS()
    #LSTMvsFF()
    #duelvsa3c()


