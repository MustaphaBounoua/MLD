import os
from src.eval_metrics.fid.fid_score import calculate_fid_given_paths
from src.utils import clean_folder
from src.eval_metrics.fad.compute_fad import get_fad_given_paths
INCEPTION_FILE ="data/pt_inception-2015-12-05-6726825d.pth"



def compute_fid(path_list,modalities_dict,subset_dict,device):


    
    results_fid = {}
    for i, key in enumerate(modalities_dict.keys()):
   # for i, key in enumerate(["svhn"]  ):   
        if modalities_dict[key].gen_quality =="fid":
            results_fid[key] = {} 

            path_real_mod = os.path.join(path_list["real"], key )
            path_random_mod = os.path.join(path_list["random"], key )
            results_fid[key]["random"] = calculate_fid_given_paths(paths=[path_real_mod,path_random_mod], batch_size=256, dims =2048, filename_state_dict = INCEPTION_FILE,device =device)
           # clean_folder(path_random_mod)
            for s_key in subset_dict.keys():
                path_gen_mod_subset = os.path.join(path_list[s_key], key )
                results_fid[key][s_key] = calculate_fid_given_paths(paths=[path_real_mod,path_gen_mod_subset], batch_size=256, dims =2048, filename_state_dict = INCEPTION_FILE,device = device)
               # clean_folder(path_gen_mod_subset)
           # clean_folder(path_real_mod)
    return results_fid




def compute_fad(path_list,modalities_dict,subset_dict,device):


    results_fad = {}
    for i, key in enumerate(modalities_dict.keys()): 

        if modalities_dict[key].fad==True:
            results_fad[key] = {}  
            path_real_mod = os.path.join(path_list["real"], key )
            path_random_mod = os.path.join(path_list["random"], key )
            results_fad[key]["random"] = get_fad_given_paths(path_1= path_real_mod,path_2= path_random_mod)
           # clean_folder(path_random_mod)
            for s_key in subset_dict.keys():
                path_gen_mod_subset = os.path.join(path_list[s_key], key )
                results_fad[key][s_key] = get_fad_given_paths(path_1= path_real_mod,path_2= path_gen_mod_subset)
           #     clean_folder(path_gen_mod_subset)
           # clean_folder(path_real_mod)
    return results_fad

