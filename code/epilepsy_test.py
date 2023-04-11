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
    DEVICE = "cuda:1" if torch.cuda.is_available() else "cpu"
    MEAN = True
    print(f"Using {DEVICE} device")

    #model = torch.load("../checkpoints/teja_vae_cp_supervised_laplacian_total_variation_rank_6.pth")
    model = torch.load("../checkpoints/teja_vae_cp_epoch_100.pth")
    model.eval()

    full_psds, _, _, _, _, grade, epi_dx, alz_dx, pids, _, _, _ = process_eegs()

    pids = np.array(pids)

    pop_psds= full_psds[(epi_dx<0) & (alz_dx<0)]

    epi_psds = full_psds[epi_dx>=0]
    epi_psds = (epi_psds - torch.min(pop_psds))/(torch.max(pop_psds) - torch.min(pop_psds))

    epi_labels = epi_dx[epi_dx>=0].detach().cpu().numpy()
    epi_pids  = pids[epi_dx.detach().cpu().numpy()>=0]

    epi_psds = epi_psds.to(DEVICE)
    epi_psds = epi_psds.to(torch.float32)

    epi_mean, epi_log_var, _  = model.encoder(epi_psds)

    if not MEAN:
        epi_latent = reparameterization(epi_mean, epi_log_var)
    else:
        epi_latent = epi_mean 

    epi_latent = epi_latent.detach().cpu().numpy()

    df = pd.DataFrame()

    df["Epilepsy"] = epi_labels
    df['PNES_vs_all'] = (df['Epilepsy'] == 1).astype(int)
    df['DRE_vs_all'] = (df['Epilepsy'] == 2).astype(int)
    df['MRE_vs_all'] = (df['Epilepsy'] == 3).astype(int)
    df["PID"] = epi_pids

    used_factors = []
    for i in range(epi_latent.shape[1]):
        used_factors.append(f"F{i+1}")
        df[f"F{i+1}"] = epi_latent[:,i]

    used_factors = ["F4", "F5"]

    print("Gaussian Naive Bayes:")
    print("PNES vs CN:")
    sub_df = df[(df['Epilepsy']==0) | (df['Epilepsy']==1)]
    features = sub_df.groupby(['PID']).mean()[used_factors].values
    labels = sub_df.groupby(['PID']).mean()['PNES_vs_all'].values
    y_mci, probs_mci = classify_Strat(features, labels)
    print("DRE vs CN:")
    sub_df = df[(df['Epilepsy']==0) | (df['Epilepsy']==2)]
    features = sub_df.groupby(['PID']).mean()[used_factors].values
    labels = sub_df.groupby(['PID']).mean()['DRE_vs_all'].values
    y_mci, probs_mci = classify_Strat(features, labels)
    print("MRE vs CN:")
    sub_df = df[(df['Epilepsy']==0) | (df['Epilepsy']==3)]
    features = sub_df.groupby(['PID']).mean()[used_factors].values
    labels = sub_df.groupby(['PID']).mean()['MRE_vs_all'].values
    y_mci, probs_mci = classify_Strat(features, labels)
    print("MRE vs PNES:")
    sub_df = df[(df['Epilepsy']==1) | (df['Epilepsy']==3)]
    features = sub_df.groupby(['PID']).mean()[used_factors].values
    labels = sub_df.groupby(['PID']).mean()['MRE_vs_all'].values
    y_mci, probs_mci = classify_Strat(features, labels)

    print("SVM:")
    print("PNES vs CN:")
    sub_df = df[(df['Epilepsy']==0) | (df['Epilepsy']==1)]
    features = sub_df.groupby(['PID']).mean()[used_factors].values
    labels = sub_df.groupby(['PID']).mean()['PNES_vs_all'].values
    y_mci, probs_mci = classify_Strat(features, labels, clf = SVC(probability = True, class_weight = "balanced"))
    print("DRE vs CN:")
    sub_df = df[(df['Epilepsy']==0) | (df['Epilepsy']==2)]
    features = sub_df.groupby(['PID']).mean()[used_factors].values
    labels = sub_df.groupby(['PID']).mean()['DRE_vs_all'].values
    y_ad, probs_ad = classify_Strat(features, labels, clf = SVC(probability = True, class_weight = "balanced"))
    print("MRE vs CN:")
    sub_df = df[(df['Epilepsy']==0) | (df['Epilepsy']==3)]
    features = sub_df.groupby(['PID']).mean()[used_factors].values
    labels = sub_df.groupby(['PID']).mean()['MRE_vs_all'].values
    y_ad, probs_ad = classify_Strat(features, labels, clf = SVC(probability = True, class_weight = "balanced"))
    print("MRE vs PNES:")
    sub_df = df[(df['Epilepsy']==1) | (df['Epilepsy']==3)]
    features = sub_df.groupby(['PID']).mean()[used_factors].values
    labels = sub_df.groupby(['PID']).mean()['MRE_vs_all'].values
    y_ad, probs_ad = classify_Strat(features, labels, clf = SVC(probability = True, class_weight = "balanced"))

    print("NN:")
    print("PNES vs CN:")
    sub_df = df[(df['Epilepsy']==0) | (df['Epilepsy']==1)]
    features = sub_df.groupby(['PID']).mean()[used_factors].values
    labels = sub_df.groupby(['PID']).mean()['PNES_vs_all'].values
    y_mci, probs_mci = classify_Strat(features, labels, clf = MLPClassifier(hidden_layer_sizes=(200,),max_iter=100000, early_stopping=True))
    print("DRE vs CN:")
    sub_df = df[(df['Epilepsy']==0) | (df['Epilepsy']==2)]
    features = sub_df.groupby(['PID']).mean()[used_factors].values
    labels = sub_df.groupby(['PID']).mean()['DRE_vs_all'].values
    y_ad, probs_ad = classify_Strat(features, labels, clf = MLPClassifier(hidden_layer_sizes=(200,),max_iter=100000, early_stopping=True))
    print("MRE vs CN:")
    sub_df = df[(df['Epilepsy']==0) | (df['Epilepsy']==3)]
    features = sub_df.groupby(['PID']).mean()[used_factors].values
    labels = sub_df.groupby(['PID']).mean()['MRE_vs_all'].values
    y_ad, probs_ad = classify_Strat(features, labels, clf = MLPClassifier(hidden_layer_sizes=(200,),max_iter=100000, early_stopping=True))
    print("MRE vs PNES:")
    sub_df = df[(df['Epilepsy']==1) | (df['Epilepsy']==3)]
    features = sub_df.groupby(['PID']).mean()[used_factors].values
    labels = sub_df.groupby(['PID']).mean()['MRE_vs_all'].values
    y_ad, probs_ad = classify_Strat(features, labels, clf = MLPClassifier(hidden_layer_sizes=(200,),max_iter=100000, early_stopping=True))
