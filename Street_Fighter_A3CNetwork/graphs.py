import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

#Global path variable names

NOSKIP = "NoSkipNetwork"
SKIP = "SkipNetwork"
DEULING = "Dueling"

def SvsNS():
    paths = ["LSTM/", "NewRew/", "OldRew/"]
    for p in paths:
        #individual
        data_ns = pd.read_csv(NOSKIP+p+"data.csv")
        points_ns = data_ns.values
        if p == paths[2]:
            for i in range(len(points_ns)):
                if points_ns[i] > 0:
                    points_ns[i] += 176

        plt.plot(points_ns, linewidth=.5)
        plt.title(NOSKIP+p[:-1])
        plt.ylabel('Reward')
        plt.xlabel('Games over Time (25 hrs)')
        
        plt.savefig("Plots/"+NOSKIP+p[:-1]+"Graph")
        #plt.show()
        plt.clf()

        data_s = pd.read_csv(SKIP+p+"data.csv")
        points_s = data_s.values
        if p == paths[2]:
            for i in range(len(points_ns)):
                if points_ns[i] > 0:
                    points_ns[i] += 176
                    
        plt.plot(points_s, linewidth=.5)
        plt.title(NOSKIP+p[:-1])
        plt.ylabel('Reward')
        plt.xlabel('Games over Time (25 hrs)')
        
        plt.savefig("Plots/"+SKIP+p[:-1]+"Graph")
        #plt.show()
        plt.clf()

        #both 
        plt.plot(points_s, linewidth=.5, label="Skip")
        plt.plot(points_ns, linewidth=.5, label="No Skip")
        plt.title("Skip vs No Skip of"+p[:-1])
        plt.ylabel('Reward')
        plt.xlabel('Games over Time (25 hrs)')
        plt.legend(loc="best")
        plt.savefig("Plots/S_vs_NS"+p[:-1]+"Graph")
        plt.clf()

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
    points_d = data_d.values

    data_a = pd.read_csv(SKIP+"OldRew/data.csv")
    points_a = data_a.values

    plt.plot(points_d, linewidth=.5, label="Dueling")
    plt.plot(points_a, linewidth=.5, label="A3C")
    plt.title("Dueling vs A3C No Skip")
    plt.ylabel('Reward')
    plt.xlabel('Games over Time (25 hrs)')
    plt.legend(loc="best")
    plt.savefig("Plots/D_vs_A3C_SkipGraph")
    plt.clf()

if __name__ == "__main__":
    #SvsNS()
    #LSTMvsFF()
    duelvsa3c()


