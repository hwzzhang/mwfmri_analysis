
import csv
import os
from pathlib import Path
import sys

import numpy as np
from joblib import Parallel, delayed
from scipy.stats import pearsonr
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import cross_validate
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.svm import LinearSVC, SVC
from tqdm import tqdm


SHARED_DIR = Path(__file__).resolve().parents[1] / 'shared'
if str(SHARED_DIR) not in sys.path:
    sys.path.insert(0, str(SHARED_DIR))
from masking import apply_fold_mask

# Reuse the identical Pearson screen across L1-C candidates within one fold.
# The cache key
# contains every quantity that can alter the result. Cleared at each outer
# fold, so this changes runtime only, not the fitted pipeline.
_CORR_SELECTION_CACHE = {}


def load_feature(save_npz_path,args,config_module):
    feature_dict = {}

    if args.demogra:
        age_list=np.load(save_npz_path,allow_pickle = True)['age_list']
        edu_list=np.load(save_npz_path,allow_pickle = True)['edu_list']
        gender_list=np.load(save_npz_path,allow_pickle = True)['gender_list']
        
        age_list=np.expand_dims(age_list, axis=1)
        edu_list=np.expand_dims(edu_list, axis=1)
        gender_list=np.expand_dims(gender_list, axis=1)
        demo_features=np.hstack((age_list,edu_list,gender_list))

        feature_dict['demogra'] = demo_features

    if args.rois_non_speech_expert:
        roi_vcsf_feature_list=np.load(save_npz_path,allow_pickle = True)['roi_vcsf_feature_list']
        roi_vgm_feature_list=np.load(save_npz_path,allow_pickle = True)['roi_vgm_feature_list']
        roi_vwm_feature_list=np.load(save_npz_path,allow_pickle = True)['roi_vwm_feature_list']
        roi_cortical_features=np.hstack((roi_vcsf_feature_list,roi_vgm_feature_list,roi_vwm_feature_list))

        feature_dict['rois_non_speech_expert']= func_rois_index(roi_cortical_features,config_module.BNA_index_Hippocampus+config_module.BNA_index_parahippocampal)
        
    if args.rois_speech_mask:
        roi_vcsf_feature_list=np.load(save_npz_path,allow_pickle = True)['roi_vcsf_feature_list']
        roi_vgm_feature_list=np.load(save_npz_path,allow_pickle = True)['roi_vgm_feature_list']
        roi_vwm_feature_list=np.load(save_npz_path,allow_pickle = True)['roi_vwm_feature_list']
        roi_cortical_features=np.hstack((roi_vcsf_feature_list,roi_vgm_feature_list,roi_vwm_feature_list))

        roi_suit_vcsf_feature_list=np.load(save_npz_path,allow_pickle = True)['roi_suit_vcsf_feature_list']
        roi_suit_vgm_feature_list=np.load(save_npz_path,allow_pickle = True)['roi_suit_vgm_feature_list']
        roi_suit_vwm_feature_list=np.load(save_npz_path,allow_pickle = True)['roi_suit_vwm_feature_list']
        roi_suit_features=np.hstack((roi_suit_vcsf_feature_list,roi_suit_vgm_feature_list,roi_suit_vwm_feature_list))
        
        rois_speech_mask0 = func_rois_index(roi_cortical_features,config_module.BNA_index_overlap_speech_mask)
        #rois_speech_mask0 = func_rois_index(roi_cortical_features,config_module.BNA_index_important_speech_mask)
        feature_dict['rois_speech_mask'] = np.hstack((rois_speech_mask0,roi_suit_features))

    if args.rois_whole:
        
        roi_vcsf_feature_list=np.load(save_npz_path,allow_pickle = True)['roi_vcsf_feature_list']
        roi_vgm_feature_list=np.load(save_npz_path,allow_pickle = True)['roi_vgm_feature_list']
        roi_vwm_feature_list=np.load(save_npz_path,allow_pickle = True)['roi_vwm_feature_list']
        roi_cortical_features=np.hstack((roi_vcsf_feature_list,roi_vgm_feature_list,roi_vwm_feature_list))


        #suit_rois
        roi_suit_vcsf_feature_list=np.load(save_npz_path,allow_pickle = True)['roi_suit_vcsf_feature_list']
        roi_suit_vgm_feature_list=np.load(save_npz_path,allow_pickle = True)['roi_suit_vgm_feature_list']
        roi_suit_vwm_feature_list=np.load(save_npz_path,allow_pickle = True)['roi_suit_vwm_feature_list']
        roi_suit_features=np.hstack((roi_suit_vcsf_feature_list,roi_suit_vgm_feature_list,roi_suit_vwm_feature_list))

        feature_dict['rois_whole']=np.hstack((roi_cortical_features,roi_suit_features))

        
    if args.tmap_speech_silence:
        feature_dict['tmap_speech_silence'] = np.load(save_npz_path,allow_pickle = True)['tmap_speech_silence_feature_list']
    if args.tmap_silence_speech:
        feature_dict['tmap_silence_speech'] = np.load(save_npz_path,allow_pickle = True)['tmap_silence_speech_feature_list']
    if args.tmap_monolog_dialog:
        feature_dict['tmap_monolog_dialog'] = np.load(save_npz_path,allow_pickle = True)['tmap_monolog_dialog_feature_list']
    if args.tmap_dialog_monolog:
        feature_dict['tmap_dialog_monolog'] = np.load(save_npz_path,allow_pickle = True)['tmap_dialog_monolog_feature_list']
    if args.tmap_speech_baseline:
        feature_dict['tmap_speech_baseline'] = np.load(save_npz_path,allow_pickle=True)['tmap_speech_baseline_feature_list']
    if args.tmap_silence_baseline:
        feature_dict['tmap_silence_baseline'] = np.load(save_npz_path,allow_pickle=True)['tmap_silence_baseline_feature_list']

    if args.AAL_FC:
        feature_dict['AAL_FC'] = np.load(save_npz_path,allow_pickle = True)['AAL_FC_feature_list']
    if args.ALFF:
        feature_dict['ALFF'] = np.load(save_npz_path,allow_pickle = True)['ALFF_feature_list']
    if args.fALFF:
        feature_dict['fALFF'] = np.load(save_npz_path,allow_pickle = True)['fALFF_feature_list']
    if args.ICA_FC:
        feature_dict['ICA_FC'] = np.load(save_npz_path,allow_pickle = True)['ICA_FC_feature_list']
    if args.ICA_FC_part:
        feature_dict['ICA_FC_part'] = np.load(save_npz_path,allow_pickle = True)['ICA_FC_particical_list']
    if args.ReHo:
        feature_dict['ReHo'] = np.load(save_npz_path,allow_pickle = True)['ReHo_feature_list']

        
    #base_moca score
    base_moca_score = np.load(save_npz_path,allow_pickle = True)['base_moca_score']
    base_moca_score=np.expand_dims(base_moca_score, axis=1)
    feature_dict['base_moce_score'] = base_moca_score
    
    # ids
    feature_dict['id_list'] = np.load(save_npz_path,allow_pickle = True)['id_list']
    
    #labels
    if 'ratio_change' in args.label_type:
        feature_dict['score_label'] = np.load(save_npz_path,allow_pickle = True)['two_year_follow_ratio_change_label']
        feature_dict['raw_score'] = np.load(save_npz_path,allow_pickle = True)['two_year_ratio_change_list']
    
    elif args.label_type == 'moca':
        feature_dict['score_label'] = np.load(save_npz_path,allow_pickle = True)['base_moca_label']
        feature_dict['raw_score'] = np.load(save_npz_path,allow_pickle = True)['base_moca_score']

    elif args.label_type == 'two_year_moca_full':
        feature_dict['score_label'] = np.load(save_npz_path,allow_pickle = True)['two_year_follow_moca_label']
        feature_dict['raw_score'] = np.load(save_npz_path,allow_pickle = True)['two_year_follow_moca_full_list']
    
    else:
        print('load features and labels error, please check......')
        exit()

    return feature_dict

def check_feature_input(args):
    feature_str=''
    if args.demogra:
        feature_str=feature_str+'_'+'demographics'
    if args.base_moca:
        feature_str=feature_str+'_'+'base_moca'
    if args.base_moca_mini:
        feature_str=feature_str+'_'+'base_moca_mini'
    
    if args.rois_non_speech_expert:
        feature_str=feature_str+'_'+'rois_non_speech_expert'
    if args.rois_speech_mask:
        feature_str=feature_str+'_'+'rois_speech_mask'
    if args.rois_whole:
        feature_str=feature_str+'_'+'rois_whole'
    
    
    if args.tmaps:
        feature_str=feature_str+'_'+'tmaps'
    if args.tmap_speech_silence:
        feature_str=feature_str+'_'+'tmaps01'
    if args.tmap_silence_speech:
        feature_str=feature_str+'_'+'tmaps02'
    if args.tmap_monolog_dialog:
        feature_str=feature_str+'_'+'tmaps03'
    if args.tmap_dialog_monolog:
        feature_str=feature_str+'_'+'tmaps04'
    if args.tmap_speech_baseline:
        feature_str=feature_str+'_'+'tmaps_control_01'
    if args.tmap_silence_baseline:
        feature_str=feature_str+'_'+'tmaps_control_02'
    if args.AAL_FC:
        feature_str=feature_str+'_'+'AAL_FC'
    if args.movie_AAL_FC:
        feature_str=feature_str+'_'+'movie_AAL_FC'
    if args.ALFF:
        feature_str=feature_str+'_'+'ALFF'
    if args.fALFF:
        feature_str=feature_str+'_'+'fALFF'
    if args.ICA_FC:
        feature_str=feature_str+'_'+'ICA_FC_MapMasker20'
    if args.ICA_FC_part:
        feature_str=feature_str+'_'+'ICA_FC_part_MapMasker20'
    if args.ReHo:
        feature_str=feature_str+'_'+'ReHo'
    if args.residualize_movie_motion:
        feature_str=feature_str+'_'+'motionResidualized'
    if args.residualize_movie_mean_fd:
        feature_str=feature_str+'_'+'meanFDResidualized'

    return feature_str


def func_rois_index(input_features,input_index):
    vcsf_input_index = [i-1 for i in input_index]
    vgm_input_index = [i-1+246 for i in input_index]
    vwm_input_index = [i-1+246*2 for i in input_index]
    return input_features[:,vcsf_input_index+vgm_input_index+vwm_input_index] 

    
def pearsonr_wyj(feature,label):
    r,p = pearsonr(feature,label)
    return (r,p)


def residualize_features(raw_feature_list, nuisance, train_index, test_index):
    """Remove linear movie-motion effects using the current training fold only."""
    raw_feature_list = np.asarray(raw_feature_list, dtype=float)
    nuisance = np.asarray(nuisance, dtype=float)
    design = np.column_stack((np.ones(nuisance.shape[0]), nuisance))
    beta = np.linalg.lstsq(
        design[train_index], raw_feature_list[train_index], rcond=None)[0]
    residualized = raw_feature_list.copy()
    residualized[train_index] -= design[train_index].dot(beta)
    residualized[test_index] -= design[test_index].dot(beta)
    return residualized

def corr_feature_select(hyper_parameter,raw_feature_list,label_list,args,train_index,test_index):
    _, num_voxel=raw_feature_list.shape
    raw_feature_list_new=raw_feature_list[train_index,:]
    label_list_new=label_list[train_index]

    results=Parallel(n_jobs=5,return_as='list')(delayed(pearsonr_wyj)(raw_feature_list_new[:,temp_col],label_list_new) for temp_col in range(num_voxel))
    corr_values=[]
    corr_p=[]

    for temp_col in range(num_voxel):
        corr_values.append(results[temp_col][0])
        corr_p.append(results[temp_col][1])
    
    corr_values=np.array(corr_values)
    corr_p=np.array(corr_p)
    
    # start feature selection
    voxel_idx=np.argwhere(corr_p < args.fsthreshold)
    
    # select feature by partion
    if voxel_idx.size == 0 and args.fsthreshold <= 0.1:
        #print('pearson corr is failed.')
        temp_fsthreshold=0.1
        order_idx=np.argsort(corr_p)
        idx_num = round(order_idx.size*temp_fsthreshold)
        voxel_idx = order_idx[:idx_num]
    elif voxel_idx.size == 0 or args.fsthreshold >= 0.1:
        order_idx=np.argsort(corr_p)
        idx_num = round(order_idx.size*args.fsthreshold)
        voxel_idx = order_idx[:idx_num]
        
    new_feature_list = raw_feature_list[:, voxel_idx].squeeze()
    return new_feature_list, voxel_idx


def feature_process(outer_flag,config_module,hyper_parameter, index, raw_feature_list,label_list,moca_score, feature_name,args,train_index,test_index):
    if 'tmap' in feature_name or 'roi' in feature_name or 'brain' in feature_name or 'FC' in feature_name or 'ALFF' in feature_name or 'ReHo' in feature_name:
        
        # correlation-based feature selection
        if args.feature_select:
            corr_key = (
                getattr(args, '_outer_cache_token', None), bool(outer_flag),
                int(index), feature_name, tuple(np.asarray(train_index).tolist()),
                float(args.fsthreshold),
            )
            cached_corr = _CORR_SELECTION_CACHE.get(corr_key)
            if cached_corr is None:
                cached_corr = corr_feature_select(
                    hyper_parameter, raw_feature_list, moca_score, args,
                    train_index, test_index)
                _CORR_SELECTION_CACHE[corr_key] = cached_corr
            raw_feature_list, voxcel_idx = cached_corr
            
        
        if args.scaler_type == 'min_max':
            scaler=MinMaxScaler((0,1))
        elif args.scaler_type == 'z_score':
            scaler = StandardScaler()

        if len(raw_feature_list[train_index].shape)==2:
            scaler.fit(raw_feature_list[train_index])
            feature_train=scaler.transform(raw_feature_list[train_index])
            feature_test=scaler.transform(raw_feature_list[test_index])
        else: 
            raw_feature_list=raw_feature_list.reshape(-1,1)
            scaler.fit(raw_feature_list[train_index])
            feature_train=scaler.transform(raw_feature_list[train_index])
            feature_test=scaler.transform(raw_feature_list[test_index])
        
        selected_linearSVC_voxcel=[]
        if args.l1_select:
            
            lsvc=LinearSVC(penalty='l1',C=hyper_parameter,dual=False,random_state=0).fit(feature_train,label_list[train_index])
            model=SelectFromModel(lsvc,prefit=True,threshold='mean')
            feature_train_select=model.transform(feature_train)
            #print('before linearSVC:',feature_test.shape)
            feature_test_select=model.transform(feature_test)
            #print('after linearSVC:',feature_test_select.shape)
            

            if feature_test_select.shape[1]==0:
                
                model=SelectFromModel(lsvc,prefit=True,threshold='mean')
                feature_train_select=model.transform(feature_train)
                feature_test_select=model.transform(feature_test)
                

            feature_train=feature_train_select 
            feature_test=feature_test_select
            selected_linearSVC_voxcel = voxcel_idx[model.get_support()]
            #print((selected_linearSVC_voxcel))

        row=feature_train.shape[0]+feature_test.shape[0]
        col=feature_test.shape[1]
        processed_feature=np.zeros((row,col))

        processed_feature[train_index,:]=feature_train
        processed_feature[test_index,:]=feature_test

        if args.save_index and outer_flag == True:

            save_npz_base_path=config_module.SAVE_FEATURE_DIR+'/cross_validation/'+feature_name+'/'+'test_'+str(args.test_proportion)+'_fs'+str(args.fsthreshold)
            save_npz_path=save_npz_base_path+'/'+'Pearson_select_'+str(index)

            if not os.path.exists(save_npz_base_path):
                        os.makedirs(save_npz_base_path)
                        print('[The dir that saves features is {0}]'.format(save_npz_base_path))

            if args.l1_select:
                np.savez(save_npz_path,voxel_after_pearson = voxcel_idx,voxel_after_l1 = selected_linearSVC_voxcel)
            elif args.feature_select:
                np.savez(save_npz_path,voxel_after_pearson = voxcel_idx)


        return processed_feature
    
    elif feature_name == 'demogra' or feature_name == 'base_moca' or feature_name == 'base_moca_mini':

        if args.scaler_type == 'min_max':
            scaler=MinMaxScaler((0,1))
        elif args.scaler_type == 'z_score':
            scaler=StandardScaler()

        scaler.fit(raw_feature_list[train_index])
        features_train=scaler.transform(raw_feature_list[train_index])
        features_test=scaler.transform(raw_feature_list[test_index])

        row=features_train.shape[0]+features_test.shape[0]
        col=features_train.shape[1]
        processed_feature=np.zeros((row,col))

        processed_feature[train_index,:]=features_train
        processed_feature[test_index,:]=features_test

        return processed_feature



def feature_select_and_combine(outer_flag,config_module,index,feature_dict,classes,Pearson_corr_label,args,train_index,test_index,hyper_parameter,mask_id=None):

    if args.demogra:
        feature_demogra = feature_process(outer_flag,config_module, hyper_parameter, index,feature_dict['demogra'],label_list=classes,moca_score=Pearson_corr_label,feature_name='demogra',\
            args=args,train_index=train_index,test_index=test_index)
    else: feature_demogra=None

    if args.tmap_speech_silence:
        # Apply the mask made from this split's training participants before
        # Pearson screening and L1 selection.
        raw_tmap_speech_silence = feature_dict['tmap_speech_silence']
        mask_train_ids = feature_dict['id_list'][train_index]
        raw_tmap_speech_silence = apply_fold_mask(
            raw_tmap_speech_silence, mask_id, args.mask_dir,
            mask_train_ids)
        if args.residualize_movie_motion or args.residualize_movie_mean_fd:
            raw_tmap_speech_silence = residualize_features(
                raw_tmap_speech_silence,
                feature_dict['movie_motion_nuisance'], train_index, test_index)
        feature_tmap_speech_silence = feature_process(outer_flag,config_module, hyper_parameter,index,raw_tmap_speech_silence,label_list=classes,moca_score=Pearson_corr_label,\
            feature_name='tmap_speech_silence',args=args,train_index=train_index,test_index=test_index)
    else: feature_tmap_speech_silence=None

    if args.tmap_silence_speech:
        feature_tmap_silence_speech = feature_process(outer_flag,config_module, hyper_parameter,index,feature_dict['tmap_silence_speech'],label_list=classes,moca_score=Pearson_corr_label,
        feature_name='tmap_silence_speech',args=args,train_index=train_index,test_index=test_index)
    else: feature_tmap_silence_speech=None

    if args.tmap_monolog_dialog:
        feature_tmap_monolog_dialog = feature_process(outer_flag, config_module, hyper_parameter, index, feature_dict['tmap_monolog_dialog'], label_list=classes, moca_score=Pearson_corr_label, feature_name='tmap_monolog_dialog', args=args, train_index=train_index, test_index=test_index)
    else: feature_tmap_monolog_dialog=None

    if args.tmap_dialog_monolog:
        feature_tmap_dialog_monolog = feature_process(outer_flag, config_module, hyper_parameter, index, feature_dict['tmap_dialog_monolog'], label_list=classes, moca_score=Pearson_corr_label, feature_name='tmap_dialog_monolog', args=args, train_index=train_index, test_index=test_index)
    else: feature_tmap_dialog_monolog=None


    if args.rois_non_speech_expert:
        feature_rois_non_speech_expert = feature_process(outer_flag,config_module, hyper_parameter,index, feature_dict['rois_non_speech_expert'],label_list=classes,feature_name='rois_non_speech_expert',moca_score=Pearson_corr_label,args=args,train_index=train_index,test_index=test_index)
    else: feature_rois_non_speech_expert = None

    if args.rois_speech_mask:
        feature_rois_speech_mask = feature_process(outer_flag,config_module, hyper_parameter,index,feature_dict['rois_speech_mask'],label_list=classes,moca_score=Pearson_corr_label,feature_name='rois_speech_mask',args=args,train_index=train_index,test_index=test_index)
    else: feature_rois_speech_mask = None

    if args.rois_whole:
        feature_rois_whole = feature_process(outer_flag,config_module, hyper_parameter,index,feature_dict['rois_whole'],label_list=classes,moca_score = Pearson_corr_label,feature_name='rois_whole',args=args,train_index=train_index,test_index=test_index)
    else: feature_rois_whole=None


    if args.AAL_FC:
        feature_AAL_FC = feature_process(outer_flag,config_module, hyper_parameter,index,feature_dict['AAL_FC'],label_list=classes,moca_score=Pearson_corr_label,feature_name='AAL_FC',args=args,train_index=train_index,test_index=test_index)
    else: feature_AAL_FC = None

    if args.movie_AAL_FC:
        feature_movie_AAL_FC = feature_process(outer_flag,config_module, hyper_parameter,index,feature_dict['movie_AAL_FC'],label_list=classes,moca_score=Pearson_corr_label,feature_name='movie_AAL_FC',args=args,train_index=train_index,test_index=test_index)
    else: feature_movie_AAL_FC = None

    if args.ALFF:
        feature_ALFF = feature_process(outer_flag,config_module, hyper_parameter,index,feature_dict['ALFF'],label_list=classes,moca_score=Pearson_corr_label,feature_name='ALFF',args=args,train_index=train_index,test_index=test_index)
    else: feature_ALFF = None

    if args.fALFF:
        feature_fALFF = feature_process(outer_flag,config_module, hyper_parameter,index,feature_dict['fALFF'],label_list=classes,moca_score=Pearson_corr_label,feature_name='fALFF',args=args,train_index=train_index,test_index=test_index)
    else: feature_fALFF = None

    if args.ICA_FC:
        feature_ICA_FC = feature_process(outer_flag,config_module, hyper_parameter,index,feature_dict['ICA_FC'],label_list=classes,moca_score=Pearson_corr_label,feature_name='ICA_FC',args=args,train_index=train_index,test_index=test_index)
    else: feature_ICA_FC = None

    if args.ICA_FC_part:
        feature_ICA_FC_part = feature_process(outer_flag,config_module, hyper_parameter,index,feature_dict['ICA_FC_part'],label_list=classes,moca_score=Pearson_corr_label,feature_name='ICA_FC_part',args=args,train_index=train_index,test_index=test_index)
    else: feature_ICA_FC_part = None

    if args.ReHo:
        feature_ReHo = feature_process(outer_flag,config_module, hyper_parameter,index,feature_dict['ReHo'],label_list=classes,moca_score=Pearson_corr_label,feature_name='ReHo',args=args,train_index=train_index,test_index=test_index)
    else: feature_ReHo = None

    
    feature_tuple=(feature_demogra, feature_tmap_speech_silence, feature_tmap_silence_speech, feature_tmap_monolog_dialog, feature_tmap_dialog_monolog, feature_rois_non_speech_expert,feature_rois_speech_mask,feature_rois_whole,feature_AAL_FC,feature_movie_AAL_FC,feature_ALFF,feature_fALFF,feature_ICA_FC,feature_ICA_FC_part,feature_ReHo)

    new_feature_tuple=tuple(x for x in feature_tuple if x is not None)
    del feature_tuple
    features_selected=np.hstack(new_feature_tuple)
    
    return features_selected


def nested_cross_validation(args,feature_str,cv_outer,cv_inner,Pearson_corr_label,label_list,id_list,feature_dict,config_module,result_dir):
    
    with open(result_dir+"/record_inner_CV.csv","a+",newline="") as csvfile_inner, open(result_dir+"/record_out_CV.csv","a+",newline="") as csvfile_outer:
    
        writer_inner = csv.writer(csvfile_inner,delimiter=',')
        writer_outer = csv.writer(csvfile_outer,delimiter=',')

        # Do not append duplicate headers when a run resumes.
        if csvfile_inner.tell() == 0:
            writer_inner.writerow(["index_out","temp_para","index_inner","AUC"])
        if csvfile_outer.tell() == 0:
            writer_outer.writerow(['index_out','best_C','test_accuracy','test_precision','test_f1','test_recall','test_roc_auc',
                'train_accuracy','train_precision','train_f1','train_recall','train_roc_auc','test_index'])

    ##### setup cross validataion and estimator ##########
    
    _, classes = np.unique(label_list, return_inverse=True)
    iter_for_prediction = cv_outer.split(id_list, classes)
    c_list=config_module.C_LIST
    print('There are hyper-parameters:',c_list)
    
    for index_out, (train_index, test_index) in tqdm(enumerate(iter_for_prediction)):
        if index_out >= args.stop_num:
            break
        if index_out < args.continue_num:
            continue

        _CORR_SELECTION_CACHE.clear()
        args._outer_cache_token = index_out

        id_list_inner=id_list[train_index]
        classes_inner=classes[train_index]
        Pearson_corr_label_inner = Pearson_corr_label[train_index]

        # prepare the inner features
        feature_dict_inner ={}
        # Retain IDs to verify the mask's training-set provenance.
        feature_dict_inner['id_list'] = id_list_inner
        if args.demogra:
            feature_dict_inner['demogra'] = feature_dict['demogra'][train_index]
        if args.rois_non_speech_expert:
            feature_dict_inner['rois_non_speech_expert'] = feature_dict['rois_non_speech_expert'][train_index]
        if args.rois_speech_mask:
            feature_dict_inner['rois_speech_mask'] = feature_dict['rois_speech_mask'][train_index]
        if args.rois_whole:
            feature_dict_inner['rois_whole'] = feature_dict['rois_whole'][train_index]
        if args.tmap_speech_silence:
            feature_dict_inner['tmap_speech_silence'] = feature_dict['tmap_speech_silence'][train_index]
        if args.residualize_movie_motion or args.residualize_movie_mean_fd:
            feature_dict_inner['movie_motion_nuisance'] = feature_dict['movie_motion_nuisance'][train_index]
        if args.tmap_silence_speech:
            feature_dict_inner['tmap_silence_speech'] = feature_dict['tmap_silence_speech'][train_index]
        if args.tmap_monolog_dialog:
            feature_dict_inner['tmap_monolog_dialog'] = feature_dict['tmap_monolog_dialog'][train_index]
        if args.tmap_dialog_monolog:
            feature_dict_inner['tmap_dialog_monolog'] = feature_dict['tmap_dialog_monolog'][train_index]
        if args.AAL_FC:
            feature_dict_inner['AAL_FC'] = feature_dict['AAL_FC'][train_index]
        if args.movie_AAL_FC:
            feature_dict_inner['movie_AAL_FC'] = feature_dict['movie_AAL_FC'][train_index]
        if args.ALFF:
            feature_dict_inner['ALFF'] = feature_dict['ALFF'][train_index]
        if args.fALFF:
            feature_dict_inner['fALFF'] = feature_dict['fALFF'][train_index]
        if args.ICA_FC:
            feature_dict_inner['ICA_FC'] = feature_dict['ICA_FC'][train_index]
        if args.ICA_FC_part:
            feature_dict_inner['ICA_FC_part'] = feature_dict['ICA_FC_part'][train_index]
        if args.ReHo:
            feature_dict_inner['ReHo'] = feature_dict['ReHo'][train_index]

        feature_dict_inner['base_moce_score']=feature_dict['base_moce_score'][train_index]
        
        
        temp_auc_dict={}

        for temp_c in c_list:

            iter_for_prediction_inner = cv_inner.split(id_list_inner, classes_inner)

            for index_inner,(train_index_inner,valid_index_inner) in enumerate(iter_for_prediction_inner):
                outer_flag = False
                mask_id = 'inner_{:03d}_{}'.format(index_out, index_inner)
                features_selected_inner = feature_select_and_combine(outer_flag,config_module=config_module,index=index_inner,feature_dict=feature_dict_inner,classes=classes_inner,Pearson_corr_label=Pearson_corr_label_inner,args=args,train_index=train_index_inner,test_index=valid_index_inner,hyper_parameter=temp_c[1],mask_id=mask_id)

                for temp_model_paras in temp_c[0]:
                    estimator =  SVC(probability=True,kernel='linear',random_state=0 ,C = temp_model_paras)

                    scores_0 = cross_validate(estimator, features_selected_inner,
                                        classes_inner,scoring=('roc_auc','f1'),
                                        return_train_score=True,return_estimator=True,
                                        cv=[(train_index_inner, valid_index_inner)])
                    
                    if str(temp_model_paras)+'_'+str(temp_c[1]) not in temp_auc_dict.keys():
                        temp_auc_dict[str(temp_model_paras)+'_'+str(temp_c[1])]=[]
                        temp_auc_dict[str(temp_model_paras)+'_'+str(temp_c[1])].append(scores_0['test_roc_auc'][0])
                    else:
                        temp_auc_dict[str(temp_model_paras)+'_'+str(temp_c[1])].append(scores_0['test_roc_auc'][0])

                    with open(result_dir+"/record_inner_CV.csv","a+",newline="") as csvfile_inner:
                        writer_inner = csv.writer(csvfile_inner,delimiter=',')
                        #  ["index_out","temp_para","index_inner","AUC"]
                        writer_inner.writerow([index_out,[temp_model_paras,temp_c[1]],index_inner,scores_0['test_roc_auc'][0]])
            
        ave_temp_auc_dict = {}
        
        for k,v in temp_auc_dict.items():
            ave_temp_auc_dict[k]=np.mean(v)
        

        # end of inner cv #
        best_paras_str = max(ave_temp_auc_dict, key=lambda x: ave_temp_auc_dict[x])
        best_C = [float(best_paras_str.split('_')[0]),float(best_paras_str.split('_')[1])]

        best_C_model =  SVC(probability=True,kernel='linear',random_state=0 ,C = best_C[0])
        #print('best c:',best_C,'test_index:',test_index)

        # the outer cv #
        outer_flag = True
        outer_mask_id = 'outer_{:03d}'.format(index_out)
        features_selected_outer = feature_select_and_combine(outer_flag,config_module=config_module,index=index_out,feature_dict=feature_dict,classes=classes,Pearson_corr_label=Pearson_corr_label,args=args,train_index=train_index,test_index=test_index,hyper_parameter=best_C[1],mask_id=outer_mask_id)

        scores_1 = cross_validate(best_C_model, features_selected_outer,
                                    classes, scoring=('accuracy', 'precision','f1','recall','roc_auc'),
                                    return_train_score=True,return_estimator=True,
                                    cv=[(train_index, test_index)])
        

        estimated_best_C_model = scores_1['estimator'][0]
        coefs = estimated_best_C_model.coef_
        #ndarray of shape (1, n_features)

        if args.save_index:
            feature_name = feature_str.split('_')[-1]
            save_npz_base_path=config_module.SAVE_FEATURE_DIR+'/cross_validation/'+feature_name+'/'+'test_'+str(args.test_proportion)+'_fs'+str(args.fsthreshold)
            save_npz_path=save_npz_base_path+'/'+'coefs_classify_'+str(index_out)

            if not os.path.exists(save_npz_base_path):
                        os.makedirs(save_npz_base_path)
                        print('[The dir that saves features is {0}]'.format(save_npz_base_path))

            np.savez(save_npz_path,model_coefs= coefs)

        with open(result_dir+"/record_out_CV.csv","a+",newline="") as csvfile_outer:
    
            writer_outer = csv.writer(csvfile_outer,delimiter=',')
            #(['index_out','best_C','test_accuracy','test_precision','test_f1','test_recall','test_roc_auc','train_accuracy','train_precision','train_f1','train_recall','train_roc_auc'])
            writer_outer.writerow([index_out,best_C,scores_1['test_accuracy'][0],scores_1['test_precision'][0],scores_1['test_f1'][0],scores_1['test_recall'][0],scores_1['test_roc_auc'][0],scores_1['train_accuracy'][0],scores_1['train_precision'][0],scores_1['train_f1'][0],scores_1['train_recall'][0],scores_1['train_roc_auc'][0],test_index])
    
    
    return
