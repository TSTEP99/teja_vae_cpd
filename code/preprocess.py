"""Functions used to preprocess EEG records"""
from concurrent.futures import ProcessPoolExecutor as Executor
from glob import glob
import numpy as np
import torch

#Maps the given string labels to integers

gender_map = {'nan':-1, 'Unknown':-1, 'Male':0, 'Female':1}
grade_map = {'nan':-1, 'normal':0, 'abnormal':1}
handed_map = {'nan':-1, 'right':0, 'left':1}
soz_map = {'nan':-1, 'right':0, 'left':1}
epi_map = {'nan':-1, 'unk':-1, 'NC': 0, 'PNES':1, 'DRE':2, 'MRE':3}
alz_map = {'nan':-1, 'unk':-1, 'CN':0, 'MCI':1, 'AD':2}

def create_indices(dims):
    """
    Takes tensor shape as input and 
    creates list of all possible indices
    """
    
    indices = []

    for dim in dims:
        indices.append(torch.arange(dim))
    indices = torch.cartesian_prod(*indices)
    return indices.long()

def parse_single(npz, average=False):
    """Function for preprocessing one NPZ file"""
    
    pib_key = 'pib'
        
    f1 = np.load(npz)
    
    ch_names = [ch.lower() for ch in f1['ch_names']]

    rep_factor = 1 if average else f1['psd'].shape[0]

    raw_psd = np.log10(f1['psd'])

    if average:
        curr_psd = np.expand_dims(np.mean(raw_psd, axis=0), axis=0)
    else:
        curr_psd = raw_psd

    raw_eeg = f1['raw']
    age = np.repeat(f1['age'], rep_factor)
    gender = np.repeat(gender_map[str(f1['gender'])], rep_factor)
    handed = np.repeat(handed_map[str(f1['handed'])], rep_factor)
    sz_side = np.repeat(soz_map[str(f1['sz_side'])], rep_factor)
    grade = np.repeat(grade_map[str(f1['abnormal'])], rep_factor)
    epi_dx = np.repeat(epi_map[str(f1['epilepsy_grp'])], rep_factor)
    alz_dx = np.repeat(alz_map[str(f1['alzheimer_grp'])], rep_factor)
    
    patient_id = np.repeat(f1['subject_id'], rep_factor)
    session_id = np.repeat(f1['session_id'], rep_factor)
    clip_id = np.repeat(f1['clip_id'], rep_factor)
    report = np.repeat(f1['report'], rep_factor)
    
    return {'psd': curr_psd, 'age': age, 'handed': handed, 'sz_side': sz_side, 'grade': grade, 
            'epi_dx': epi_dx, 'alz_dx': alz_dx, 'gender': gender, 'pid': patient_id, 
            'sid': session_id, 'clip_id': clip_id, 'raw': raw_eeg, 'report': report}

def process_eegs(stats_dir = '/mnt/ssd_4tb_0/TUH/processed_yoga/', return_raw=False):
    """Function for preprocessing multiple EEG records"""

    #computes a list of the npz files containing the EEG data
    all_npz = sorted(glob(stats_dir + '*.npz'))

    #Uses executor to parse each individual EEG
    with Executor(max_workers=30) as executor:
        results = [res for i,res in zip(all_npz, executor.map(parse_single, all_npz))]
    
    #Stacks the resultes of parse single into a 1-D numpy array
    raw = np.vstack([res['raw'] for res in results])
    full_psds = np.vstack([res['psd'] for res in results])
    age = np.concatenate([res['age'] for res in results])
    gender = np.concatenate([res['gender'] for res in results])
    handed = np.concatenate([res['handed'] for res in results])
    sz_side = np.concatenate([res['sz_side'] for res in results])
    grade = np.concatenate([res['grade'] for res in results])
    epi_dx = np.concatenate([res['epi_dx'] for res in results])
    alz_dx = np.concatenate([res['alz_dx'] for res in results])

    pids = np.concatenate([res['pid'] for res in results])
    pids = [str(idx).strip() for idx in pids]
    sids = np.concatenate([res['sid'] for res in results])
    sids = [str(idx).strip() for idx in sids]
    cids = np.concatenate([res['clip_id'] for res in results])
    cids = [str(idx).strip() for idx in cids]
    reports = np.concatenate([res['report'] for res in results])
    reports = [str(rep).strip().lower() for rep in reports]

    #Converts each numpy array into a torch tensor
    full_psds = torch.from_numpy(full_psds)
    age = torch.from_numpy(age)
    gender = torch.from_numpy(gender)
    handed = torch.from_numpy(handed)
    sz_side = torch.from_numpy(sz_side)
    grade = torch.from_numpy(grade)
    epi_dx = torch.from_numpy(epi_dx)
    alz_dx = torch.from_numpy(alz_dx)

    if return_raw:
        return full_psds, age, gender, handed, sz_side, grade, epi_dx, alz_dx, pids, sids, cids, reports, raw
    
    return full_psds, age, gender, handed, sz_side, grade, epi_dx, alz_dx, pids, sids, cids, reports

if __name__ == "__main__":
    returned_arrays = process_eegs()

    full_psds = returned_arrays[0]
    grade = returned_arrays[5]
    epi_dx = returned_arrays[6]
    alz_dx = returned_arrays[7]

    print("full_psds dimensions:", full_psds.shape)
    print("grade dimensions:", grade.shape)
    print("epi dx dimensions:", epi_dx.shape)
    print("alz_dx dimensions:", alz_dx.shape)

    print("full_psds min:", torch.min(full_psds))
    print("full_psds max:", torch.max(full_psds))