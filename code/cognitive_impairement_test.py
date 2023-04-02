from helper import reparameterization
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import KFold
from sklearn.model_selection import LeaveOneOut

from sklearn.model_selection import cross_val_score, cross_val_predict, KFold, LeaveOneGroupOut, LeaveOneOut, StratifiedKFold
from sklearn.metrics import roc_curve, roc_auc_score, auc, classification_report, precision_recall_fscore_support,accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from preprocess import process_eegs
import numpy as np
import pandas as pd 
import torch

def classify_Strat(features, labels, clf= GaussianNB(), adjust_sampling_weight=False, folds=15):

    #clf = GaussianNB()
    kf = StratifiedKFold(folds)

    all_y = []
    all_probs=[]
    
    auc_scores=[]

    for train, test in kf.split(features, labels):
        if adjust_sampling_weight:
            sampling_dict = {0:1./np.sum(labels[train]==0), 1:1./np.sum(labels[train]==1)}
            sample_weight = sklearn.utils.class_weight.compute_sample_weight(sampling_dict, labels[train])
            fit_model = clf.fit(features[train], labels[train], sample_weight=sample_weight)
        else:
            fit_model = clf.fit(features[train], labels[train])
        all_y.append(labels[test])
        all_probs.append(fit_model.predict_proba(features[test])[:,1])
        try:
            auc_scores.append(roc_auc_score(all_y[-1], all_probs[-1]))
        except:
            pass
    print("AUC Mean:",np.mean(auc_scores))
    print("AUC Std:", np.std(auc_scores))
    all_y = np.concatenate(all_y)
    all_probs = np.concatenate(all_probs)
    
    return all_y, all_probs


if __name__ == "__main__":
    DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
    MEAN = True
    print(f"Using {DEVICE} device")

    model = torch.load("../checkpoints/teja_vae_cp_supervised_laplacian_total_variation_rank_6.pth")
    model.eval()

    full_psds, _, _, _, _, grade, epi_dx, alz_dx, pids, _, _, _ = process_eegs()

    pids = np.array(pids)

    pop_psds= full_psds[(epi_dx<0) & (alz_dx<0)]

    alz_psds = full_psds[alz_dx>=0]
    alz_psds = (alz_psds - torch.min(pop_psds))/(torch.max(pop_psds) - torch.min(pop_psds))

    alz_labels = alz_dx[alz_dx>=0].detach().cpu().numpy()
    alz_pids  = pids[alz_dx.detach().cpu().numpy()>=0]

    alz_psds = alz_psds.to(DEVICE)
    alz_psds = alz_psds.to(torch.float32)

    alz_mean, alz_log_var, _  = model.encoder(alz_psds)

    if not MEAN:
        alz_latent = reparameterization(alz_mean, alz_log_var)
    else:
        alz_latent = alz_mean 

    alz_latent = alz_latent.detach().cpu().numpy()

    df = pd.DataFrame()

    df["Alzheimers"] = alz_labels
    df['MCI_vs_all'] = (df['Alzheimers'] == 1).astype(int)
    df['AD_vs_all'] = (df['Alzheimers'] == 2).astype(int)
    df["PID"] = alz_pids

    used_factors = []
    for i in range(alz_latent.shape[1]):
        used_factors.append(f"F{i+1}")
        df[f"F{i+1}"] = alz_latent[:,i]

    #used_factors = ["F1", "F2", "F6"]

    print("Gaussian Naive Bayes:")
    print("MCI vs CN:")
    sub_df = df[(df['Alzheimers']==0) | (df['Alzheimers']==1)]
    features = sub_df.groupby(['PID']).mean()[used_factors].values
    labels = sub_df.groupby(['PID']).mean()['MCI_vs_all'].values
    y_mci, probs_mci = classify_Strat(features, labels)
    print("AD vs CN:")
    sub_df = df[(df['Alzheimers']==0) | (df['Alzheimers']==2)]
    features = sub_df.groupby(['PID']).mean()[used_factors].values
    labels = sub_df.groupby(['PID']).mean()['AD_vs_all'].values
    y_ad, probs_ad = classify_Strat(features, labels)

    print("SVM:")
    print("MCI vs CN:")
    sub_df = df[(df['Alzheimers']==0) | (df['Alzheimers']==1)]
    features = sub_df.groupby(['PID']).mean()[used_factors].values
    labels = sub_df.groupby(['PID']).mean()['MCI_vs_all'].values
    y_mci, probs_mci = classify_Strat(features, labels, clf = SVC(probability = True, class_weight = "balanced"))
    print("AD vs CN:")
    sub_df = df[(df['Alzheimers']==0) | (df['Alzheimers']==2)]
    features = sub_df.groupby(['PID']).mean()[used_factors].values
    labels = sub_df.groupby(['PID']).mean()['AD_vs_all'].values
    y_ad, probs_ad = classify_Strat(features, labels, clf = SVC(probability = True, class_weight = "balanced"))

    print("NN:")
    print("MCI vs CN:")
    sub_df = df[(df['Alzheimers']==0) | (df['Alzheimers']==1)]
    features = sub_df.groupby(['PID']).mean()[used_factors].values
    labels = sub_df.groupby(['PID']).mean()['MCI_vs_all'].values
    y_mci, probs_mci = classify_Strat(features, labels, clf = MLPClassifier(hidden_layer_sizes=(200,),max_iter=100000, early_stopping=True))
    print("AD vs CN:")
    sub_df = df[(df['Alzheimers']==0) | (df['Alzheimers']==2)]
    features = sub_df.groupby(['PID']).mean()[used_factors].values
    labels = sub_df.groupby(['PID']).mean()['AD_vs_all'].values
    y_ad, probs_ad = classify_Strat(features, labels, clf = MLPClassifier(hidden_layer_sizes=(200,),max_iter=100000, early_stopping=True))
