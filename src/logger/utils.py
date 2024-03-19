import numpy as np




    
def log_results_train_step(logger, metrics, epoch, prefix ="train/" ):
        # res=  { "loss":metrics["loss"].cpu().detach(),
        #         "KLD_joint": metrics["KLD_joint"]
        #         }
        # for key in res.keys():
        #     logger.experiment.add_scalar(prefix+key, res[key],epoch)
            
        # keys_unimodal = [ key for key in ["KLDs", "Rec_loss","unimodal_elbos","loss_nx","loss_mod","Rec_nex"] if key in metrics.keys()] 
        #print(metrics)
        for m_key in metrics.keys(): 
            if isinstance(metrics[m_key],dict):
                logger.experiment.add_scalars(prefix+m_key, metrics[m_key], epoch)
            else:
                logger.experiment.add_scalar(prefix+m_key, metrics[m_key], epoch)
            
            
            
           
            
def log_results_eval_step(logger, metrics, epoch, prefix ="eval/"):
        for key in metrics.keys():
            for m_key in metrics[key].keys():
                if isinstance(metrics[key][m_key],dict):
                    logger.experiment.add_scalars(prefix+key+"_"+m_key, metrics[key][m_key],epoch)     
                else:
                    logger.experiment.add_scalar(prefix+key+"_"+m_key, metrics[key][m_key],epoch)  


        
def log_modalities(logger,output,modalities_list,epoch,nb_samples=4, prefix="sampling/"):
    
    for mod in modalities_list:
        data_mod = output[mod.name].cpu()
        ready_to_plot = mod.plot(data_mod[:nb_samples])
  
        if mod.modality_type == "img":
            logger.experiment.add_image(prefix + mod.name, ready_to_plot, global_step=epoch)
        elif mod.modality_type == "txt":
            logger.experiment.add_text(prefix + mod.name, ready_to_plot, global_step=epoch)
        elif mod.modality_type =="audio":
            for idx,audio in enumerate(ready_to_plot):
                logger.experiment.add_audio(prefix + mod.name + str(idx), audio, global_step=epoch,sample_rate=16000)
                logger.experiment.add_figure(prefix +"spectograms/"+str(idx),mod.plot_spec(data_mod[idx]), global_step=epoch )
               


def log_cond_modalities(logger,output,modalities_list,epoch,nb_samples=4, prefix="cond_("):
    for mod in modalities_list:
        for cond in output.keys():
            data_mod = output[cond][mod.name].cpu()
            ready_to_plot = mod.plot(data_mod[:nb_samples])
            if mod.modality_type == "img":
                logger.experiment.add_image(prefix +cond+")/" +mod.name, ready_to_plot, global_step=epoch)
            elif mod.modality_type == "txt":
                logger.experiment.add_text(prefix +cond+")/" + mod.name, ready_to_plot, global_step=epoch)
            elif mod.modality_type =="audio":
                for idx,audio in enumerate(ready_to_plot):
                    logger.experiment.add_audio(prefix +cond+")/"+ mod.name + str(idx), audio, global_step=epoch,sample_rate=16000)
                    logger.experiment.add_figure(prefix +cond+")/""spectograms/"+str(idx),mod.plot_spec(data_mod[idx]), global_step=epoch )
                


def flatten_dict(dict_res):
    res_dict= { key :{} for key in   dict_res["cond"][ list(dict_res["cond"].keys())[0]].keys()
    }
      
    for key in dict_res["cond"].keys():
        cond = dict_res["cond"][key]
        for key_res in cond.keys():
            res_dict[key_res][key] = dict_res["cond"][key][key_res]
    if "random/joint_cohrence" in res_dict.keys():
        res_dict["random/joint_cohrence"] = dict_res["random"]
    return res_dict






def log_results(logger, metrics, epoch):
        
        res=  { "loss":np.mean([ m["loss"].cpu().detach() for m in metrics ] ),
                "KLD_joint":np.mean([ m["KLD_joint"].cpu().detach()  for m in metrics ]) 
                }
        for key in res.keys():
            logger.experiment.add_scalar(key, res[key],epoch)
            
        keys_unimodal = [ key for key in ["KLDs", "Rec_loss","unimodal_elbos"] if key in metrics[0].keys()]
        
        for m_key in keys_unimodal:    
            res[m_key] = {}
            for s_key in metrics[0][m_key].keys():
                res[m_key][s_key] = []

        for m in metrics:
            for m_key in keys_unimodal:
                for s_key in m[m_key]:
                    res[m_key][s_key].append(m["Rec_loss"][s_key].cpu())
                    
        for m_key in keys_unimodal:    
            res[m_key] = np.mean(res[m_key])
            logger.experiment.add_scalars(m_key, res[m_key],epoch)
        return res