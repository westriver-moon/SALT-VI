
import numpy as np
import torch
from torch.autograd import Variable
from salt_vi.utils import eval_regdb, eval_sysu, eval_llcm
import os


def _eval_image_feature(base, visual_output, mode="RGB", use_backup=False):
    if use_backup and base._uses_spatial_map_visual():
        feat = base.backup_pool(visual_output).flatten(1)
        return base.backup_classifier(feat)
    feat = base.extract_global_feat(visual_output)
    return base.classifier(feat, mode)


def _needs_text(test_modality):
    return any(modality in test_modality for modality in ("Fusion", "Text"))

def test(base, loader, config, device):
    embed_dim = base.embed_dim
    base.set_eval()
    print('Extracting Query Feature...')
    ptr = 0
    print('Test Mode: ', config.test_modality)
    result_dict = dict()
    
    if loader.dataset not in ['sysu', 'regdb', 'llcm']:
        raise ValueError(f"Invalid dataset: {loader.dataset}")
    if not any(modality in config.test_modality for modality in ('IR', 'Fusion', 'Text')):
        raise ValueError(f"Invalid test modality: {config.test_modality}")

    if loader.dataset == 'regdb':
        query_feat_ir_list = []
        query_feat_fusion_list = []
        query_feat_text_list = []
        query_feat_ir_f_list = []
        query_feat_text_f_list = []
        for i in range(config.eval_num_regdb):
            ptr = 0
            query_loader = loader.query_loaders[i]
            n_query = len(loader.query_labels[i])
            if 'IR' in config.test_modality:
                query_feat_ir = np.zeros((n_query, embed_dim), dtype=np.float32)
            if 'Fusion' in config.test_modality:
                query_feat_fusion = np.zeros((n_query, embed_dim), dtype=np.float32)
                if config.CAT_EVAL:
                    query_feat_ir_f = np.zeros((n_query, embed_dim), dtype=np.float32)
                    query_feat_text_f = np.zeros((n_query, embed_dim), dtype=np.float32)
            if 'Text' in config.test_modality:
                query_feat_text = np.zeros((n_query, embed_dim), dtype=np.float32)
            with torch.no_grad():
                # for batch_idx, (input, text, label) in enumerate(loader.query_loader):
                for batch_idx, batch_dict in enumerate(query_loader):
                    input = batch_dict['img']
                    # label = batch_dict['target']

                    batch_num = input.size(0)
                    input = Variable(input.to(device))
                    
                    if _needs_text(config.test_modality):
                        text = batch_dict['text']
                        text = text.to(device).long()
                        
                    # get the feature of the last layer 
                    if 'IR' in config.test_modality:
                        feat_ir_map = base.encode_image_featmap(input,'ir') # IR mode
                        feat_ir = _eval_image_feature(base, feat_ir_map, mode="IR", use_backup=config.Fix_Visual)
                        query_feat_ir[ptr:ptr + batch_num, :] = feat_ir.detach().cpu().numpy()
                    if 'Fusion' in config.test_modality:
                        if config.Feat_Filter:
                            text_filter = batch_dict['text_filter']
                            text_filter = text_filter.to(device).long()
                            feat_fusion = base.encode_filtered_fusion(text,text_filter,input)  # Fusion mode  rgbtext + irimage
                        else:
                            feat_fusion = base.encode_fusion(text,input,'ir')  # Fusion mode  rgbtext + irimage
                            
                        feat_fusion = base.classifier(feat_fusion,"Fusion") # [64, 512] -> [64, 512]
                        query_feat_fusion[ptr:ptr + batch_num, :] = feat_fusion.detach().cpu().numpy()
                        if config.CAT_EVAL:
                            ##################################################
                            feat_text = base.encode_text_feat(text)
                            feat_text = base.classifier(feat_text, 'Text')
                            query_feat_text_f[ptr:ptr + batch_num, :] = feat_text.detach().cpu().numpy()
                            ##################################################
                            feat_ir = base.encode_image_feat(input,'ir')
                            feat_ir = base.classifier(feat_ir, 'IR')
                            query_feat_ir_f[ptr:ptr + batch_num, :] = feat_ir.detach().cpu().numpy()
                            ##################################################

                    if 'Text' in config.test_modality:
                        feat_text = base.encode_text_feat(text)
                        feat_text = base.classifier(feat_text,'Text')
                        query_feat_text[ptr:ptr + batch_num, :] = feat_text.detach().cpu().numpy()

                    ptr = ptr + batch_num
                if 'IR' in config.test_modality:
                    query_feat_ir_list.append(query_feat_ir)
                if 'Fusion' in config.test_modality:
                    query_feat_fusion_list.append(query_feat_fusion)
                    if config.CAT_EVAL:
                        query_feat_ir_f_list.append(query_feat_ir_f)
                        query_feat_text_f_list.append(query_feat_text_f)
                if 'Text' in config.test_modality:
                    query_feat_text_list.append(query_feat_text)
                

    else:
        if 'IR' in config.test_modality:
            query_feat_ir = np.zeros((loader.n_query, embed_dim), dtype=np.float32)
        if 'Fusion' in config.test_modality:
            query_feat_fusion = np.zeros((loader.n_query, embed_dim), dtype=np.float32)
            if config.CAT_EVAL:
                #####################
                query_feat_ir_f = np.zeros((loader.n_query, embed_dim), dtype=np.float32)
                #####################
                query_feat_text_f = np.zeros((loader.n_query, embed_dim), dtype=np.float32)
                #####################
        if 'Text' in config.test_modality:
            query_feat_text = np.zeros((loader.n_query, embed_dim), dtype=np.float32)
        with torch.no_grad():
            # for batch_idx, (input, text, label) in enumerate(loader.query_loader):
            for batch_idx, batch_dict in enumerate(loader.query_loader):
                input = batch_dict['img']
                # label = batch_dict['target']

                batch_num = input.size(0)
                input = Variable(input.to(device))
                
                if _needs_text(config.test_modality):
                    text = batch_dict['text']
                    text = text.to(device).long()
                    
                # get the feature of the last layer 
                if 'IR' in config.test_modality:
                    feat_ir_map = base.encode_image_featmap(input,'ir') # IR mode
                    feat_ir = _eval_image_feature(base, feat_ir_map, mode="IR", use_backup=config.Fix_Visual)
                    query_feat_ir[ptr:ptr + batch_num, :] = feat_ir.detach().cpu().numpy()
                if 'Fusion' in config.test_modality:
                    if config.Feat_Filter:
                        text_filter = batch_dict['text_filter']
                        text_filter = text_filter.to(device).long()
                        feat_fusion = base.encode_filtered_fusion(text,text_filter,input)  # Fusion mode  rgbtext + irimage
                    else:
                        feat_fusion = base.encode_fusion(text,input,'ir')  # Fusion mode  rgbtext + irimage
                        
                    feat_fusion = base.classifier(feat_fusion,"Fusion") # [64, 512] -> [64, 512]
                    query_feat_fusion[ptr:ptr + batch_num, :] = feat_fusion.detach().cpu().numpy()
                    if config.CAT_EVAL:
                        ##################################################
                        feat_text = base.encode_text_feat(text)
                        feat_text = base.classifier(feat_text, 'Text')
                        query_feat_text_f[ptr:ptr + batch_num, :] = feat_text.detach().cpu().numpy()
                        ##################################################
                        feat_ir = base.encode_image_feat(input,'ir')
                        feat_ir = base.classifier(feat_ir, 'IR')
                        query_feat_ir_f[ptr:ptr + batch_num, :] = feat_ir.detach().cpu().numpy()
                        ##################################################
                if 'Text' in config.test_modality:
                    feat_text = base.encode_text_feat(text)
                    feat_text = base.classifier(feat_text,'Text')
                    query_feat_text[ptr:ptr + batch_num, :] = feat_text.detach().cpu().numpy()

                ptr = ptr + batch_num



    print('Extracting Gallery Feature...')

    if loader.dataset == 'sysu':
        all_cmc_ir = 0
        all_mAP_ir = 0
        all_mINP_ir = 0
        all_cmc_fusion = 0
        all_mAP_fusion = 0
        all_mINP_fusion = 0
        all_cmc_text = 0
        all_mAP_text = 0
        all_mINP_text = 0
        for i in range(10):
            ptr = 0
            gall_loader = loader.gallery_loaders[i]
            gall_label = loader.gallery_labels[i]
            gall_cam = loader.gallery_cams[i]
            n_gallery = len(gall_label)
            if 'IR' in config.test_modality and config.Fix_Visual:
                    gall_feat_for_IR = np.zeros((n_gallery, embed_dim), dtype=np.float32)
            if 'IR' in config.test_modality or 'Fusion' in config.test_modality or 'Text' in config.test_modality:
                gall_feat = np.zeros((n_gallery, embed_dim), dtype=np.float32)
            with torch.no_grad():
                # for batch_idx, (input, text, label) in enumerate(gall_loader):
                for batch_idx, batch_dict in enumerate(gall_loader):
                    input = batch_dict['img']
                    # label = batch_dict['target']

                    batch_num = input.size(0)
                    input = Variable(input.to(device))

                    # get the feature of the last layer 
                    feat_map = base.encode_image_featmap(input,'rgb')
                    if 'IR' in config.test_modality and config.Fix_Visual:
                        feat_for_IR = _eval_image_feature(base, feat_map, mode="RGB", use_backup=True)
                        gall_feat_for_IR[ptr:ptr + batch_num, :] = feat_for_IR.detach().cpu().numpy()
                    if 'IR' in config.test_modality or 'Text' in config.test_modality or 'Fusion' in config.test_modality:
                        feat = _eval_image_feature(base, feat_map, mode="RGB", use_backup=False)
                        gall_feat[ptr:ptr + batch_num, :] = feat.detach().cpu().numpy()
                    else:
                        raise ValueError('Error: test_modality not found!')
                    ptr = ptr + batch_num
                    
            if 'IR' in config.test_modality:
                if config.Fix_Visual:
                    distmat_ir = np.matmul(query_feat_ir, np.transpose(gall_feat_for_IR))
                else:
                    distmat_ir = np.matmul(query_feat_ir, np.transpose(gall_feat))
                cmc_ir, mAP_ir, mINP_ir = eval_sysu(-distmat_ir, loader.query_label, gall_label, loader.query_cam,
                                        gall_cam)
                all_cmc_ir += cmc_ir
                all_mAP_ir += mAP_ir
                all_mINP_ir += mINP_ir
            if 'Fusion' in config.test_modality:
                if config.CAT_EVAL:
                    distmat_ir_f = np.matmul(query_feat_ir_f, np.transpose(gall_feat))
                    distmat_text_f = np.matmul(query_feat_text_f, np.transpose(gall_feat))
                    distmat_fusion = np.matmul(query_feat_fusion, np.transpose(gall_feat)) + distmat_ir_f + distmat_text_f
                else:
                    distmat_fusion = np.matmul(query_feat_fusion, np.transpose(gall_feat))
                cmc_fusion, mAP_fusion, mINP_fusion = eval_sysu(-distmat_fusion, loader.query_label, gall_label, loader.query_cam,
                                        gall_cam)
                all_cmc_fusion += cmc_fusion
                all_mAP_fusion += mAP_fusion
                all_mINP_fusion += mINP_fusion
            if 'Text' in config.test_modality:
                distmat_text = np.matmul(query_feat_text, np.transpose(gall_feat))
                cmc_text, mAP_text, mINP_text = eval_sysu(-distmat_text, loader.query_label, gall_label, loader.query_cam,
                                        gall_cam)
                all_cmc_text += cmc_text
                all_mAP_text += mAP_text
                all_mINP_text += mINP_text
        
        if 'IR' in config.test_modality:
            all_cmc_ir /= 10.0
            all_mAP_ir /= 10.0
            all_mINP_ir /= 10.0
            result_dict['IR'] = (all_mINP_ir, all_mAP_ir, all_cmc_ir)
        if 'Fusion' in config.test_modality:
            all_cmc_fusion /= 10.0
            all_mAP_fusion /= 10.0
            all_mINP_fusion /= 10.0
            result_dict['Fusion'] = (all_mINP_fusion, all_mAP_fusion, all_cmc_fusion)
        if 'Text' in config.test_modality:
            all_cmc_text /= 10.0
            all_mAP_text /= 10.0
            all_mINP_text /= 10.0
            result_dict['Text'] = (all_mINP_text, all_mAP_text, all_cmc_text)


    elif loader.dataset == 'regdb':
        all_cmc_ir = 0
        all_mAP_ir = 0
        all_mINP_ir = 0
        all_cmc_fusion = 0
        all_mAP_fusion = 0
        all_mINP_fusion = 0
        all_cmc_text = 0
        all_mAP_text = 0
        all_mINP_text = 0
        for i in range(config.eval_num_regdb):
            ptr = 0
            gall_loader = loader.gallery_loaders[i]
            query_label = loader.query_labels[i]
            gall_label = loader.gallery_labels[i]
            n_gallery = len(gall_label)
            if 'IR' in config.test_modality and config.Fix_Visual:
                    gall_feat_for_IR = np.zeros((n_gallery, embed_dim), dtype=np.float32)
            if 'IR' in config.test_modality or 'Fusion' in config.test_modality or 'Text' in config.test_modality:
                gall_feat = np.zeros((n_gallery, embed_dim), dtype=np.float32)
            with torch.no_grad():
                for batch_idx, batch_dict in enumerate(gall_loader):
                    input = batch_dict['img']

                    batch_num = input.size(0)
                    input = Variable(input.to(device))

                    # get the feature of the last layer 
                    feat_map = base.encode_image_featmap(input,'rgb')
                    if 'IR' in config.test_modality and config.Fix_Visual:
                        feat_for_IR = _eval_image_feature(base, feat_map, mode="RGB", use_backup=True)
                        gall_feat_for_IR[ptr:ptr + batch_num, :] = feat_for_IR.detach().cpu().numpy()
                    if 'IR' in config.test_modality or 'Text' in config.test_modality or 'Fusion' in config.test_modality:
                        feat = _eval_image_feature(base, feat_map, mode="RGB", use_backup=False)
                        gall_feat[ptr:ptr + batch_num, :] = feat.detach().cpu().numpy()
                    else:
                        raise ValueError('Error: test_modality not found!')
                    ptr = ptr + batch_num
            
            if 'IR' in config.test_modality:
                query_feat_ir = query_feat_ir_list[i]
                if config.regdb_test_mode == 't-v':
                    if config.Fix_Visual:
                        distmat_ir = np.matmul(query_feat_ir, np.transpose(gall_feat_for_IR))
                    else:
                        distmat_ir = np.matmul(query_feat_ir, np.transpose(gall_feat))
                    cmc_ir, mAP_ir, mINP_ir = eval_regdb(-distmat_ir, query_label, gall_label)
                else:
                    if config.Fix_Visual:
                        distmat_ir = np.matmul(gall_feat_for_IR, np.transpose(query_feat_ir))
                    else:
                        distmat_ir = np.matmul(gall_feat, np.transpose(query_feat_ir))
                    cmc_ir, mAP_ir, mINP_ir = eval_regdb(-distmat_ir, gall_label, query_label)
                all_cmc_ir += cmc_ir
                all_mAP_ir += mAP_ir
                all_mINP_ir += mINP_ir
            
            if 'Fusion' in config.test_modality:
                query_feat_fusion = query_feat_fusion_list[i]
                if config.CAT_EVAL:
                    query_feat_ir_f = query_feat_ir_f_list[i]
                    query_feat_text_f = query_feat_text_f_list[i]
                if config.regdb_test_mode == 't-v':
                    if config.CAT_EVAL:
                        distmat_ir_f = np.matmul(query_feat_ir_f, np.transpose(gall_feat))
                        distmat_text_f = np.matmul(query_feat_text_f, np.transpose(gall_feat))
                        distmat_fusion = np.matmul(query_feat_fusion, np.transpose(gall_feat)) + distmat_ir_f + distmat_text_f
                    else:
                        distmat_fusion = np.matmul(query_feat_fusion, np.transpose(gall_feat))
                    cmc_fusion, mAP_fusion, mINP_fusion = eval_regdb(-distmat_fusion, query_label, gall_label)
                else:
                    distmat_fusion = np.matmul(gall_feat, np.transpose(query_feat_fusion))
                    if config.CAT_EVAL:
                        distmat_fusion += np.matmul(gall_feat, np.transpose(query_feat_ir_f))
                        distmat_fusion += np.matmul(gall_feat, np.transpose(query_feat_text_f))
                    cmc_fusion, mAP_fusion, mINP_fusion = eval_regdb(-distmat_fusion, gall_label, query_label)
                all_cmc_fusion += cmc_fusion
                all_mAP_fusion += mAP_fusion
                all_mINP_fusion += mINP_fusion
            
            if 'Text' in config.test_modality:
                query_feat_text = query_feat_text_list[i]
                if config.regdb_test_mode == 't-v':
                    distmat_text = np.matmul(query_feat_text, np.transpose(gall_feat))
                    cmc_text, mAP_text, mINP_text = eval_regdb(-distmat_text, query_label, gall_label)
                else:
                    distmat_text = np.matmul(gall_feat, np.transpose(query_feat_text))
                    cmc_text, mAP_text, mINP_text = eval_regdb(-distmat_text, gall_label, query_label)
                all_cmc_text += cmc_text
                all_mAP_text += mAP_text
                all_mINP_text += mINP_text

        num_eval = config.eval_num_regdb
        if 'IR' in config.test_modality:
            all_cmc_ir /= num_eval
            all_mAP_ir /= num_eval
            all_mINP_ir /= num_eval
            result_dict['IR'] = (all_mINP_ir, all_mAP_ir, all_cmc_ir)
        if 'Fusion' in config.test_modality:
            all_cmc_fusion /= num_eval
            all_mAP_fusion /= num_eval
            all_mINP_fusion /= num_eval
            result_dict['Fusion'] = (all_mINP_fusion, all_mAP_fusion, all_cmc_fusion)
        if 'Text' in config.test_modality:
            all_cmc_text /= num_eval
            all_mAP_text /= num_eval
            all_mINP_text /= num_eval
            result_dict['Text'] = (all_mINP_text, all_mAP_text, all_cmc_text)
    
    elif loader.dataset == 'llcm':
        all_cmc_ir = 0
        all_mAP_ir = 0
        all_mINP_ir = 0
        all_cmc_fusion = 0
        all_mAP_fusion = 0
        all_mINP_fusion = 0
        all_cmc_text = 0
        all_mAP_text = 0
        all_mINP_text = 0
        for i in range(10):
            ptr = 0
            gall_loader = loader.gallery_loaders[i]
            gall_label = loader.gallery_labels[i]
            gall_cam = loader.gallery_cams[i]
            n_gallery = len(gall_label)
            if 'IR' in config.test_modality and config.Fix_Visual:
                    gall_feat_for_IR = np.zeros((n_gallery, embed_dim), dtype=np.float32)
            if 'IR' in config.test_modality or 'Fusion' in config.test_modality or 'Text' in config.test_modality:
                gall_feat = np.zeros((n_gallery, embed_dim), dtype=np.float32)
            with torch.no_grad():
                for batch_idx, batch_dict in enumerate(gall_loader):
                    input = batch_dict['img']

                    batch_num = input.size(0)
                    input = Variable(input.to(device))

                    # get the feature of the last layer 
                    feat_map = base.encode_image_featmap(input,'rgb')
                    if 'IR' in config.test_modality and config.Fix_Visual:
                        feat_for_IR = _eval_image_feature(base, feat_map, mode="RGB", use_backup=True)
                        gall_feat_for_IR[ptr:ptr + batch_num, :] = feat_for_IR.detach().cpu().numpy()
                    if 'IR' in config.test_modality or 'Text' in config.test_modality or 'Fusion' in config.test_modality:
                        feat = _eval_image_feature(base, feat_map, mode="RGB", use_backup=False)
                        gall_feat[ptr:ptr + batch_num, :] = feat.detach().cpu().numpy()
                    else:
                        raise ValueError('Error: test_modality not found!')
                    ptr = ptr + batch_num
            
            if 'IR' in config.test_modality:
                if config.Fix_Visual:
                    distmat_ir = np.matmul(query_feat_ir, np.transpose(gall_feat_for_IR))
                else:
                    distmat_ir = np.matmul(query_feat_ir, np.transpose(gall_feat))
                cmc_ir, mAP_ir, mINP_ir = eval_llcm(-distmat_ir, loader.query_label, gall_label, loader.query_cam,
                                        gall_cam)
                all_cmc_ir += cmc_ir
                all_mAP_ir += mAP_ir
                all_mINP_ir += mINP_ir
            
            if 'Fusion' in config.test_modality:
                if config.CAT_EVAL:
                    distmat_ir_f = np.matmul(query_feat_ir_f, np.transpose(gall_feat))
                    distmat_text_f = np.matmul(query_feat_text_f, np.transpose(gall_feat))
                    distmat_fusion = np.matmul(query_feat_fusion, np.transpose(gall_feat)) + distmat_ir_f + distmat_text_f
                else:
                    distmat_fusion = np.matmul(query_feat_fusion, np.transpose(gall_feat))
                cmc_fusion, mAP_fusion, mINP_fusion = eval_llcm(-distmat_fusion, loader.query_label, gall_label, loader.query_cam,
                                        gall_cam)
                all_cmc_fusion += cmc_fusion
                all_mAP_fusion += mAP_fusion
                all_mINP_fusion += mINP_fusion
            if 'Text' in config.test_modality:
                distmat_text = np.matmul(query_feat_text, np.transpose(gall_feat))
                cmc_text, mAP_text, mINP_text = eval_llcm(-distmat_text, loader.query_label, gall_label, loader.query_cam,
                                        gall_cam)
                all_cmc_text += cmc_text
                all_mAP_text += mAP_text
                all_mINP_text += mINP_text
        if 'IR' in config.test_modality:
            all_cmc_ir /= 10.0
            all_mAP_ir /= 10.0
            all_mINP_ir /= 10.0
            result_dict['IR'] = (all_mINP_ir, all_mAP_ir, all_cmc_ir)
        if 'Fusion' in config.test_modality:
            all_cmc_fusion /= 10.0
            all_mAP_fusion /= 10.0
            all_mINP_fusion /= 10.0
            result_dict['Fusion'] = (all_mINP_fusion, all_mAP_fusion, all_cmc_fusion)
        if 'Text' in config.test_modality:
            all_cmc_text /= 10.0
            all_mAP_text /= 10.0
            all_mINP_text /= 10.0
            result_dict['Text'] = (all_mINP_text, all_mAP_text, all_cmc_text)

    return result_dict
