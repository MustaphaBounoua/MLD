

import numpy as np
from sklearn.metrics import accuracy_score
from src.utils import save_generated_samples_singlegroup
from tqdm import tqdm

from src.eval_metrics.MID.MID import *
from src.eval_metrics.fd.FD import get_inception_net ,populate_metrics_step_fid , init_metric_fd



def classify_cond_gen_samples(labels , modalities_list , cond_samples, device ):
    
    
    clfs = {}
    for modality in modalities_list:
        clfs[modality.name] = modality.classifier.to(device)
    
    eval_labels = dict()

        
    for idx,key in enumerate(clfs.keys() ):
        if key in cond_samples.keys():
            mod_cond_gen = cond_samples[key]
           # mod_clf = clfs[key].eval().to(device)
            mod_clf = clfs[key].to(device)
            attr_hat = mod_clf(  modalities_list[idx].get_reconstruction(mod_cond_gen) )
            pred = np.argmax(attr_hat.cpu().data.numpy(), axis=1).astype(int);
            score = accuracy_score(pred, labels)
            eval_labels[key] = score
        else:
            print(str(key) + 'not existing in cond_gen_samples')
            
    return eval_labels;


def calculate_coherence( modalities_list, samples ,batch_size,device):
    
    clfs = {}
    for modality in modalities_list:
        clfs[modality.name] = modality.classifier.to(device)
        
    mods = modalities_list
    # TODO: make work for num samples NOT EQUAL to batch_size

    pred_mods = np.zeros((len(mods), batch_size))
    for idx, mod in enumerate(modalities_list):
    
           # clf_mod = mod.classifier.eval().to(device)
            clf_mod = mod.classifier.to(device)
            samples_mod = samples[mod.name]
            #attr_mod = clf_mod(samples_mod)
            
            attr_mod = clf_mod(  modalities_list[idx].get_reconstruction(samples_mod) )
            
            output_prob_mod = attr_mod.cpu().data.numpy()
            
            pred_mod = np.argmax(output_prob_mod, axis=1).astype(int)
            
            pred_mods[idx, :] = pred_mod
    
    coh_mods = np.all(pred_mods == pred_mods[0, :], axis=0)
    coherence = np.sum(coh_mods.astype(int))/float(batch_size);
    
    return coherence


# def test_generation(model, subset_list, modalities_list,d_loader,batch_size ,num_samples_fid, device ,path_fid ,do_fd , nb_batchs = None):
#     mods = { mod.name: mod for mod in modalities_list}
#     mm_vae = model.to(device)
#     mm_vae.eval()
#     modalities_str = np.array([mod.name for mod in modalities_list])
#     subsets = { ','.join( modalities_str[s]) : s for s in subset_list}
#     gen_perf = dict();
     
#     gen_perf = {'cond': dict(),
#                 'random': dict()}
#     gen_perf['cond'] = dict();
#     for k, s_key in enumerate(subsets.keys()):
#         if s_key != '':
#             gen_perf['cond'][s_key] = dict();
#             for m, m_key in enumerate(mods.keys()):
#                 gen_perf['cond'][s_key][m_key] = [];
#     gen_perf['random'] = [];
#     num_batches_epoch = int(d_loader.dataset.__len__() /float(batch_size));
#     cnt_s = 0;
#     for iteration, batch in tqdm( enumerate(d_loader), desc =" Executing Coherence Evaluation" ):
#         print("iteration {}".format(iteration))
#         if nb_batchs !=None and iteration > nb_batchs :
#             break;
        
#         batch_d = batch[0]
#         batch_l = batch[1]
        
#         bs = len(batch_l)
        
#         rand_gen =  mm_vae.sample(bs)
#         coherence_random = calculate_coherence( modalities_list, rand_gen ,bs ,device )
#         gen_perf['random'].append(coherence_random);

        
        
        
#         if (batch_size*iteration) < num_samples_fid and do_fd:
            
#             save_generated_samples_singlegroup( batch_id = iteration, group_name ='random',
#                                               samples = rand_gen, batch_size =batch_size,
#                                               modalities_list = modalities_list, 
#                                               paths_fid = path_fid )
#             to_save = {}
#             for mod in modalities_list:
#                to_save[mod.name] = batch_d[mod.name] 
#             save_generated_samples_singlegroup( batch_id = iteration, 
#                                                 group_name ='real',
#                                                 samples = to_save,
#                                                 batch_size =batch_size,
#                                                 modalities_list = modalities_list, 
#                                                 paths_fid = path_fid 
#                                                )
            
#         for k, m_key in enumerate(batch_d.keys()):
#             batch_d[m_key] = batch_d[m_key].to(device);
            
#         output_cond_gen = mm_vae.conditional_gen_all_subsets(batch_d) 
        
#         for k, s_key in enumerate(output_cond_gen.keys()):
#             clf_cg = classify_cond_gen_samples(labels=batch_l,
#                                                modalities_list = modalities_list,
#                                                cond_samples= output_cond_gen[s_key],
#                                                device = device)
#             for m, m_key in enumerate(mods.keys()):
#                     gen_perf['cond'][s_key][m_key].append(clf_cg[m_key])    
                        
#             if (batch_size*iteration) < num_samples_fid and do_fd:
#                 save_generated_samples_singlegroup( batch_id = iteration, group_name = s_key,
#                                             samples = output_cond_gen[s_key], batch_size =batch_size, modalities_list = modalities_list
#                                             ,  paths_fid  = path_fid )
#        # if iteration<5:
#        #     print(str(gen_perf))  
                
#     for k, s_key in enumerate(subsets.keys()):
#         if s_key != '':
#             for l, m_key in enumerate(mods.keys()):
#                 perf = mean_eval_metric(gen_perf['cond'][s_key][m_key])
#                 gen_perf['cond'][s_key][m_key] = perf        
#     gen_perf['random'] = mean_eval_metric(gen_perf['random'])
#     return gen_perf



def mean_eval_metric( values):
        return np.mean(np.array(values));
    





def test_gen_base(model, subset_list, modalities_list,d_loader,batch_size ,num_samples_fid, device ,path_fid ,do_fd , nb_batchs = None):
    modalities_str = np.array([mod.name for mod in modalities_list])
    subsets = { ','.join( modalities_str[s]) : s for s in subset_list}
    
    if do_fd:
        inception = get_inception_net()
        gen_perf_fd = dict();
        for k, s_key in enumerate(subsets.keys()):
            gen_perf_fd[s_key] = dict();
            for i, mod in enumerate(modalities_list):
                if mod.gen_quality:
                    gen_perf_fd[s_key][mod.name] = init_metric_fd(mod.fd["act_dim"],limit=num_samples_fid,device=device)
        gen_perf_fd["random"]= dict()           
        for i, mod in enumerate(modalities_list):
                if mod.gen_quality:
                    gen_perf_fd["random"][mod.name] = init_metric_fd(mod.fd["act_dim"],limit=num_samples_fid,device=device)
        
 #   print(gen_perf_fd)
    mods = { mod.name: mod for mod in modalities_list}
    mm_vae = model.to(device)
    mm_vae.eval()
    
    gen_perf = dict();
     
    gen_perf = {'cond': dict(),
                'random': dict()}
    gen_perf['cond'] = dict();
    for k, s_key in enumerate(subsets.keys()):
        if s_key != '':
            gen_perf['cond'][s_key] = dict();
            for m, m_key in enumerate(mods.keys()):
                gen_perf['cond'][s_key][m_key] = [];
    gen_perf['random'] = [];
    
    for iteration, batch in tqdm( enumerate(d_loader), desc =" Executing Coherence Evaluation" ):
        print("iteration {}".format(iteration))
        if nb_batchs !=None and iteration > nb_batchs :
            break;
        
        batch_d = batch[0]
        batch_l = batch[1]
        
        bs = len(batch_l)
        
        rand_gen =  mm_vae.sample(bs)
        coherence_random = calculate_coherence( modalities_list, rand_gen ,bs ,device )
        gen_perf['random'].append(coherence_random);

        
        
        
        if (batch_size*iteration) < num_samples_fid and do_fd:
            
            for i, mod in enumerate(modalities_list):
                if mod.gen_quality:
                    if mod.fd["fd"] =="inception":
                        populate_metrics_step_fid( gen_perf_fd["random"][mod.name] ,x =batch_d[mod.name] ,y = rand_gen[mod.name], inception_model= inception )
                    else:
                        populate_metrics_step_fid( gen_perf_fd["random"][mod.name] ,x =batch_d[mod.name] ,y = rand_gen[mod.name], classifier= mod.classifier )
        

            save_generated_samples_singlegroup( batch_id = iteration, group_name ='random',
                                              samples = rand_gen, batch_size =batch_size,
                                              modalities_list = modalities_list, 
                                              paths_fid = path_fid )
            to_save = {}
            for mod in modalities_list:
               to_save[mod.name] = batch_d[mod.name] 
            save_generated_samples_singlegroup( batch_id = iteration, 
                                                group_name ='real',
                                                samples = to_save,
                                                batch_size =batch_size,
                                                modalities_list = modalities_list, 
                                                paths_fid = path_fid 
                                               )
            
        for k, m_key in enumerate(batch_d.keys()):
            batch_d[m_key] = batch_d[m_key].to(device);
            
        output_cond_gen = mm_vae.conditional_gen_all_subsets(batch_d) 
        
        for k, s_key in enumerate(output_cond_gen.keys()):
            clf_cg = classify_cond_gen_samples(labels=batch_l,
                                               modalities_list = modalities_list,
                                               cond_samples= output_cond_gen[s_key],
                                               device = device)
            for m, m_key in enumerate(mods.keys()):
                    gen_perf['cond'][s_key][m_key].append(clf_cg[m_key])    
                        
            if (batch_size*iteration) < num_samples_fid and do_fd:
                save_generated_samples_singlegroup( batch_id = iteration, group_name = s_key,
                                            samples = output_cond_gen[s_key], batch_size =batch_size, modalities_list = modalities_list
                                            ,  paths_fid  = path_fid )
            if (batch_size*iteration) < num_samples_fid and do_fd:
                for i, mod in enumerate(modalities_list):
                    if mod.gen_quality:
                        if mod.fd["fd"] =="inception":
                            populate_metrics_step_fid( gen_perf_fd[s_key][mod.name] ,x =batch_d[mod.name] ,y = output_cond_gen[s_key][mod.name], inception_model= inception )
                        else:
                            populate_metrics_step_fid( gen_perf_fd[s_key][mod.name] ,x =batch_d[mod.name] ,y = output_cond_gen[s_key][mod.name], classifier= mod.classifier )
                        
            
    fd_results = dict()
    if do_fd:
        for k, s_key in enumerate(subsets.keys()):
            fd_results[s_key] = dict();
            for i, mod in enumerate(modalities_list):
                if mod.gen_quality:
                    #print(mod)
                    fd_results[s_key][mod.name] = gen_perf_fd[s_key][mod.name].compute()
        fd_results["random"] = dict()             
        for i, mod in enumerate(modalities_list):
                
                if mod.gen_quality:
                        fd_results["random"][mod.name] = gen_perf_fd["random"][mod.name].compute()
           
        
              
    for k, s_key in enumerate(subsets.keys()):
        if s_key != '':
            for l, m_key in enumerate(mods.keys()):
                perf = mean_eval_metric(gen_perf['cond'][s_key][m_key])
                gen_perf['cond'][s_key][m_key] = perf        

    gen_perf['random'] = mean_eval_metric(gen_perf['random'])
    print(gen_perf['random'])
    
    return gen_perf,fd_results








def test_celebA(model, subset_list, modalities_list,d_loader,batch_size ,num_samples_fid, device ,path_fid ,do_fd , nb_batchs = None):
    modalities_str = np.array([mod.name for mod in modalities_list])
    subsets = { ','.join( modalities_str[s]) : s for s in subset_list}
    
    if do_fd:
        inception = get_inception_net()
        gen_perf_fd = dict();
        for k, s_key in enumerate(subsets.keys()):
            gen_perf_fd[s_key] = dict();
            for i, mod in enumerate(modalities_list):
                if mod.gen_quality:
                    gen_perf_fd[s_key][mod.name] = init_metric_fd(mod.fd["act_dim"],limit=num_samples_fid,device=device)
        gen_perf_fd["random"]= dict()           
        for i, mod in enumerate(modalities_list):
                if mod.gen_quality:
                    gen_perf_fd["random"][mod.name] = init_metric_fd(mod.fd["act_dim"],limit=num_samples_fid,device=device)
        
 #   print(gen_perf_fd)
    mods = { mod.name: mod for mod in modalities_list}
    mm_vae = model.to(device)
    mm_vae.eval()
    
    gen_perf = dict();
     
    gen_perf = {'cond': dict(),
                'random': dict()}
    gen_perf['cond'] = dict();
    for k, s_key in enumerate(subsets.keys()):
        if s_key != '':
            gen_perf['cond'][s_key] = dict();
            for m, m_key in enumerate(mods.keys()):
                if m_key!="image":
                    gen_perf['cond'][s_key][m_key] = [];
    gen_perf['random'] = [];
    
    for iteration, batch in tqdm( enumerate(d_loader), desc =" Executing Coherence Evaluation" ):
        print("iteration {}".format(iteration))
        if nb_batchs !=None and iteration > nb_batchs :
            break;
        if len(batch) ==2:
            batch_d = batch[0]
        else:
            batch_d =batch
        #batch_l = batch[1]
        
        bs = batch_d["image"].size(0)
        
        rand_gen =  mm_vae.sample(bs)
        #coherence_random = calculate_coherence( modalities_list, rand_gen ,bs ,device )
        #gen_perf['random'].append(coherence_random);

        
        
        
        if (batch_size*iteration) < num_samples_fid and do_fd:
            

                populate_metrics_step_fid( gen_perf_fd["random"]["image"] ,x =batch_d["image"] ,y = rand_gen["image"], inception_model= inception )

        #   #  save_generated_samples_singlegroup( batch_id = iteration, group_name ='random',
        #                                       samples = rand_gen, batch_size =batch_size,
        #                                       modalities_list = modalities_list, 
        #                                       paths_fid = path_fid )
        #    # to_save = {}
        #    # for mod in modalities_list:
        #    #    to_save[mod.name] = batch_d[mod.name] 
        #    # save_generated_samples_singlegroup( batch_id = iteration, 
        #                                         group_name ='real',
        #                                         samples = to_save,
        #                                         batch_size =batch_size,
        #                                         modalities_list = modalities_list, 
        #                                         paths_fid = path_fid 
        #                                        )
            
        for k, m_key in enumerate(batch_d.keys()):
            batch_d[m_key] = batch_d[m_key].to(device);
            
        output_cond_gen = mm_vae.conditional_gen_all_subsets(batch_d) 
        cond_fd=["image","mask,attributes","mask","attributes"]
        for k, s_key in enumerate(output_cond_gen.keys()):
       # for k, s_key in enumerate(cond_fd):
                # clf_cg = classify_cond_gen_samples(labels=batch_l,
                #                                 modalities_list = modalities_list,
                #                                 cond_samples= output_cond_gen[s_key],
                #                                 device = device)
                
                print(s_key)
                f_1_mask = modalities_list[1].get_f_1_score( output_cond_gen[s_key] ["mask"], batch_d["mask"]  )
                f_1_att = modalities_list[2].get_f_1_score( output_cond_gen[s_key] ["attributes"], batch_d["attributes"]  )

                gen_perf['cond'][s_key]["mask"].append(  f_1_mask )    
                gen_perf['cond'][s_key]["attributes"].append(  f_1_att ) 



                # if (batch_size*iteration) < num_samples_fid and do_fd:
                #     save_generated_samples_singlegroup( batch_id = iteration, group_name = s_key,
                #                                 samples = output_cond_gen[s_key], batch_size =batch_size, modalities_list = modalities_list
                #                                 ,  paths_fid  = path_fid )
                if (batch_size*iteration) < num_samples_fid and do_fd:
                    if s_key in cond_fd:
                        print(s_key)
                        for i, mod in enumerate(modalities_list):
                            if mod.gen_quality:
                                if mod.fd["fd"] =="inception":
                                    populate_metrics_step_fid( gen_perf_fd[s_key][mod.name] ,x =batch_d[mod.name] ,y = output_cond_gen[s_key][mod.name], inception_model= inception )
                            
                                else:
                                    populate_metrics_step_fid( gen_perf_fd[s_key][mod.name] ,x =batch_d[mod.name] ,y = output_cond_gen[s_key][mod.name], classifier= mod.classifier )

    fd_results = dict()
    if do_fd:
        for k, s_key in enumerate(cond_fd):
            fd_results[s_key] = dict();
            for i, mod in enumerate(modalities_list):
                if mod.gen_quality: 
                    fd_results[s_key][mod.name] = gen_perf_fd[s_key][mod.name].compute()
        fd_results["random"] = gen_perf_fd["random"]["image"].compute()           
        
        
              
    for k, s_key in enumerate(subsets.keys()):
        if s_key != '':
            for l, m_key in enumerate(mods.keys()):
                if m_key !="image":
                    perf = mean_eval_metric(gen_perf['cond'][s_key][m_key])
                    gen_perf['cond'][s_key][m_key] = perf        

    #gen_perf['random'] = mean_eval_metric(gen_perf['random'])
    #print(gen_perf['random'])
    
    return gen_perf,fd_results

















def test_Clip(model, modalities_list,d_loader,batch_size ,num_samples_fid, device , do_fd ,limit_clip =20000, nb_batchs = None,path_fid =None):
   
    mm_vae = model.to(device)
    mm_vae.eval()
    
    eval_model = "ViT-B/32"       
    limit = max( limit_clip,num_samples_fid)  # number of reference samples
    clip_model, clip_prep = get_clip(eval_model, device)
    inception = get_inception_net()
   # num_samples_fid=500
   # limit_clip=10000
    gen_perf = dict()

    gen_perf_list = [
            #    'cond_image_(image,sentence)', 'cond_image_(sentence)',
                'sentence_cond_(image)',
                'image_cond_(sentence)',
            #    'cond_sentence_(image,sentence)',
            #    'cond_sentence_(sentence)',
            #   'cond_image_(image)',
                'dataset',
                'random']
    gen_perf =dict()
    for e in gen_perf_list:
        gen_perf[e] = init_metric_list(clip_model,limit_clip,device,e != "random")       
                
    
    gen_perf_fid = {
                'FD_image_random':init_metric_fd(act_dim = 2048,limit=num_samples_fid, device=device),
                'FD_image_cond_sentence':init_metric_fd(act_dim = 2048,limit=num_samples_fid, device=device)
    }
    nb_iter = limit//batch_size
    
    #nb_batchs =5 
    
    for iteration, batch in tqdm( enumerate(d_loader), desc =" Executing Coherence Evaluation" ):
        print("iteration {}".format(iteration))
        if (nb_batchs !=None and iteration > nb_batchs ) or nb_iter<iteration :
            break
        
        batch_d = batch[0]
        batch_l = batch[1]
        
        bs = batch[0]["image"].size(0)
      
        rand_gen =  mm_vae.sample(bs)
        
        populate_metrics_step(gen_perf["random"] , image = rand_gen["image"],text = rand_gen["sentence"],clip_prep= clip_prep,clip_model= clip_model,modalities_list=modalities_list)


        if (batch_size*iteration) < num_samples_fid and do_fd:
            populate_metrics_step_fid(gen_perf_fid["FD_image_random"],x=batch_d["image"] ,y=  rand_gen["image"] ,inception_model=  inception,device=device)
            
            # save_generated_samples_singlegroup( batch_id = iteration, group_name ='random',
            #                                    samples = rand_gen, batch_size =batch_size,
            #                                    modalities_list = modalities_list, 
            #                                    paths_fid = path_fid )
            # to_save = {}
            # for mod in modalities_list:
            #     to_save[mod.name] = batch_d[mod.name] 
            #     save_generated_samples_singlegroup( batch_id = iteration, 
            #                                          group_name ='real',
            #                                          samples = to_save,
            #                                          batch_size =batch_size,
            #                                          modalities_list = modalities_list, 
            #                                          paths_fid = path_fid 
            #                                         )
            
        for k, m_key in enumerate(batch_d.keys()):
            batch_d[m_key] = batch_d[m_key].to(device);
            
        output_cond_gen = mm_vae.conditional_gen_all_subsets(batch_d) 
        
        # populate_metrics_step(gen_perf["cond_image_(image,sentence)"] , image= output_cond_gen["image,sentence"]["image"], text= batch_d["sentence"],
        #                       clip_model = clip_model,clip_prep = clip_prep,modalities_list=modalities_list)
        # populate_metrics_step(gen_perf["cond_image_(image)"] , image = output_cond_gen["image"]["image"], text = batch_d["sentence"],
        #                       clip_model = clip_model,clip_prep = clip_prep,modalities_list=modalities_list)
        
        
        populate_metrics_step(gen_perf["image_cond_(sentence)"] ,image=  output_cond_gen["sentence"]["image"], 
                              text= batch_d["sentence"], 
                              img_ref=batch_d["image"],
                              clip_model = clip_model,clip_prep = clip_prep,modalities_list=modalities_list)
        
        populate_metrics_step(gen_perf["sentence_cond_(image)"] ,
                              text =  output_cond_gen["image"]["sentence"], 
                              image= batch_d["image"],
                              text_ref=batch_d["sentence"],
                              clip_model = clip_model,clip_prep = clip_prep,modalities_list=modalities_list)
        
        populate_metrics_step(gen_perf["dataset"] ,
                              text =  batch_d["sentence"], 
                              image= batch_d["image"],
                              img_ref=batch_d["image"],
                              clip_model = clip_model,clip_prep = clip_prep,modalities_list=modalities_list)
        # populate_metrics_step(gen_perf["cond_sentence_(image,sentence)"] ,text = output_cond_gen["image,sentence"]["sentence"],image= batch_d["image"],
        #                       clip_model = clip_model,clip_prep = clip_prep,modalities_list=modalities_list)
        # populate_metrics_step(gen_perf["cond_sentence_(sentence)"] , text =output_cond_gen["sentence"]["sentence"],image= batch_d["image"], 
        #                       clip_model = clip_model,clip_prep = clip_prep,modalities_list=modalities_list)
       
        
        # for k, s_key in enumerate(output_cond_gen.keys()):           
        #     if (batch_size*iteration) < num_samples_fid and do_fd:
        #         save_generated_samples_singlegroup( batch_id = iteration, group_name = s_key,
        #                                     samples = output_cond_gen[s_key], batch_size =batch_size, modalities_list = modalities_list
        #                                     ,  paths_fid  = path_fid )
        if (batch_size*iteration) < num_samples_fid and do_fd:      
            populate_metrics_step_fid(gen_perf_fid["FD_image_cond_sentence"],x=batch_d["image"] ,y=  output_cond_gen["sentence"]["image"] ,inception_model=  inception)
        
             
        # if iteration<5:
        #     print(str(gen_perf))  
               
                
                
    results=dict()     
    for k, s_key in enumerate(gen_perf.keys()):
        perf = [m.compute(reduction=True) for m in  gen_perf[s_key]]
        results[s_key]= dict()
        if s_key =="random":
            for r,m in zip(perf,["CLIP-S"] ):
                results[s_key][m]= r.detach().cpu().item()
        else:
            for r,m in zip(perf,["CLIP-S","MID"] ):
                results[s_key][m]= r.detach().cpu().item()
         
    results["FD_image_cond_sentence"] = gen_perf_fid["FD_image_cond_sentence"].compute(reduction=False)
    results["FD_image_random"] = gen_perf_fid["FD_image_random"].compute(reduction=True)
    return results
















