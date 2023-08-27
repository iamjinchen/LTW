import time
import os
import sys
import argparse
import shutil
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.utils.data as data
from torch.optim import lr_scheduler
from tensorboardX import SummaryWriter
from easydict import EasyDict
from network import model_selection_meta as model_selection
from torch.utils.data.sampler import SubsetRandomSampler, RandomSampler
import random
from network import FNet
from utils.loss import CompactLoss
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn import functional as F
import copy
from dataloader.dataloader_ffpp import FFpp
from dataloader.dataloader_dfdc import DFDCDetection
from dataloader.dataloader_celebdf import CeleDF
from dataloader.transform import get_transform
from utils.config import *
from utils.roc import cal_metric
from utils import *
import time
os.environ['CUDA_VISIBLE_DEVICES']='0,1'#important
from torch.nn.parallel import DistributedDataParallel as DDP
import apex 
# from apex.optimizers import FP16_Optimizer
# from apex.optimizers import FusedAdam
# from transformers import get_linear_schedule_wit_warmup

def worker_init_fn(x):
    seed = RNG_SEED
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    return

def model_forward(image, model, post_function=nn.Sigmoid(),feat = False):
    """
    Predicts the label of an input image. Preprocesses the input image and
    casts it to cuda if required

    :param image: numpy image
    :param model: torch model with linear layer at the end
    :param post_function: e.g., softmax
    :return: prediction (1 = fake, 0 = real)
    """
    # start_time = time.time()
    # Model prediction
    # if fp16:
    #     image = image.half()
    with torch.cuda.amp.autocast():
        output,feature = model(image)
    #TODO: RI 
        output = post_function(output)
        output = output.squeeze()
        prediction = (output >= 0.5).float()
    # print(f"model_forward took {time.time() - start_time} seconds")
    if feat==False:
        return prediction, output
    else:
        return prediction, output, feature


def train(model,optimizer,fnet,optimizer_fnet,train_dataloader,meta_dataloader,criterion_oc,epoch,epoch_size,device):
    '''
    train the LTW framework.
    '''
    if meta_dataloader!=None:
        meta_dataloader_iter = iter(meta_dataloader)
    losses = AverageMeter()
    acces = AverageMeter()
    meta_losses = AverageMeter()
    meta_acces = AverageMeter()
    post_function=nn.Sigmoid()
    criterion_norm = torch.nn.BCELoss().to(device)
    criterion = torch.nn.BCELoss(reduce = False).to(device)
    meta_lr = metalr * ((gamma ** int(epoch >= step_size)) * (gamma ** int(epoch >= step_size*2)) * (gamma ** int(epoch >= step_size*3)) * (gamma ** int(epoch >= step_size*4)))         
    
    print(f"meta lr = {meta_lr}")  
    scaler = torch.cuda.amp.GradScaler()

    for i, datas in enumerate(train_dataloader):
        
        images = datas[0].to(device)
        targets = datas[1].float().to(device)
        p = datas[3]

        meta_model = model_selection(model_name=model_name, num_classes=1,pretained = False)
        #to get theta'
        if parallel:
            meta_model.copyModel(model.module)
        else:
            meta_model.copyModel(model)
        
     
        prediction, output,feature = model_forward(images,model,feat = True)
        targets = targets.to(output, non_blocking=True)
        compact_loss = criterion_oc(output,targets)
        cost = criterion(output, targets)
        cost_v = torch.reshape(cost, (len(cost), 1))
        with torch.cuda.amp.autocast():
            f_lambda = fnet(feature)

            f_lambda_norm = nn.Sigmoid()(f_lambda)

            l_f_model = torch.sum(cost_v)/len(cost_v) + lamda * compact_loss

        model.zero_grad()
        
        if parallel:
            grads = torch.autograd.grad(l_f_model, (model.module.params()),create_graph=True)
        else:
            grads = torch.autograd.grad(l_f_model, (model.params()),create_graph=True)
        
        #Virtual update model
        meta_model.update_params(lr_inner=meta_lr, source_params=grads,solver = 'adam')
        
        if meta_dataloader!=None:
            try:
                meta_datas = next(meta_dataloader_iter)
            except StopIteration:
                meta_dataloader_iter = iter(meta_dataloader)
                meta_datas = next(meta_dataloader_iter)
            meta_images = meta_datas[0].to(device)
            meta_targets = meta_datas[1].float().to(device)
            meta_p = meta_datas[3]
        else:
            #use opposite image
            opposite_images = datas[4].to(device)
            opposite_targets = (targets-1)*-1
            opposite_p = list(datas[5])
            domain_images = datas[6].to(device)
            domain_targets = targets
            domain_p = list(datas[7])


            meta_images = opposite_images
            meta_targets = opposite_targets
        
        prediction_meta, output_mata = model_forward(meta_images,meta_model)
        
        l_g_meta = criterion_norm(output_mata, meta_targets)
        with torch.no_grad():
            w_new = fnet(feature)
            w_new_norm = nn.Sigmoid()(w_new)

        loss = torch.sum(cost_v * (w_new_norm+1))/len(cost_v)

        loss_add = loss + alpha*l_g_meta + lamda*compact_loss
        
        start_time = time.time()
        
        optimizer.zero_grad()
        optimizer_fnet.zero_grad()
        
        # Scales the loss, and calls backward()
        # to create scaled gradients
        scaler.scale(loss_add).backward()

        # Unscales gradients and calls
        # or skips optimizer.step()
        scaler.step(optimizer)
        scaler.step(optimizer_fnet) 

        # Updates the scale for next iteration
        scaler.update()
          
        print(f"backward steps took {time.time()-start_time} seconds") #backward steps took 8.353310585021973 seconds
        acc = (prediction==targets).float().mean()
        meta_acc = (prediction_meta==meta_targets).float().mean()

        meta_acces.update(meta_acc.item(), targets.size(0))
        acces.update(acc.item(), targets.size(0))

        losses.update(loss.item(), targets.size(0))
        meta_losses.update(l_g_meta.item(),targets.size(0))

        if i % print_interval == 0:
            print(f'{time.ctime()} || Epoch:{epoch} || Iter:{i}/{epoch_size} || ' + 
                    f'Loss:{losses.val:.4f}({losses.avg:.4f}) || Accuracy:{acces.val:.4f}({acces.avg:.4f}) || MetaAccuracy:{meta_acces.val:.4f}({meta_acces.avg:.4f}) || MetaLoss:{meta_losses.val:.4f}({meta_losses.avg:.4f})'
                    )

def test(data_loader, model, device):
    model.eval()
    acces = []
    losses = 0.0
    label_list = []
    output_list = []
    criterion = torch.nn.BCELoss().to(device)
    wrongimg = []
    for i, datas in enumerate(tqdm(data_loader)):
        images = datas[0].to(device)#3,3,224,224
        targets = datas[1].float().to(device)
        
        with torch.no_grad():
            prediction, output = model_forward(images, model)
        label_list.extend(targets.cpu().numpy().tolist())
        output_list.extend(output.cpu().numpy().tolist())
        this_acces = (targets == prediction).cpu().numpy()
        if i == 0 or len(this_acces) == len(acces[-1]) :
            acces.append((targets == prediction).cpu().numpy())
        else:
             print(f"{i} th acces is not appended.")
        loss = criterion(output, targets).item()
        losses += loss
    metrics = EasyDict()
    metrics.acc = np.mean(acces)
    eer,TPRs, auc,scaler = cal_metric(label_list,output_list,False)

    metrics.loss = losses / len(data_loader)
    metrics.auc = auc
    metrics.tpr = TPRs
    metrics.eer = eer
    model.train()
    
    return metrics

def updatebest(metrics,best_acc,best_loss,name,model,pnet):
    if metrics.acc>best_acc:
        best_acc = metrics.acc
        if model!=None:
            save_checkpoint(model.state_dict(), fpath=f'{save_dir}/{model_name}_best{name}acc.pth')
        if pnet!=None:
            save_checkpoint(pnet.state_dict(), fpath=f'{save_dir}/{model_name}_pnet_best{name}acc.pth')
        print(f'best_{name}_acc:{best_acc:.5f} (updated)', end=' ')
    else:
        print(f'best_{name}_acc:{best_acc:.5f} ')

    if metrics.loss<best_loss:
        best_loss = metrics.loss
        if model!=None:
            save_checkpoint(model.state_dict(), fpath=f'{save_dir}/{model_name}_best{name}loss.pth')
        if pnet!=None:
            save_checkpoint(pnet.state_dict(), fpath=f'{save_dir}/{model_name}_pnet_best{name}loss.pth')
        print(f'best_{name}_loss:{best_loss:.5f} (updated)')
    else:
        print(f'best_{name}_loss:{best_loss:.5f} ')

    return best_acc,best_loss


def main():#param = rank,world_size
    # print(f"Running basic DDP example on rank {rank}.")
    # setup(rank, world_size)
    
    #TODO: save yaml config
    # save_dir = save_dir + '/'+ str(time.time())
    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)
    print(f'save dir :{save_dir}')
    sys.stdout = Logger(os.path.join(save_dir, 'train.log'))

    device = 'cuda' if torch.cuda.is_available else 'cpu'

    model = model_selection(model_name=model_name, num_classes=1)

    fnet = FNet(model.num_ftrs).to(device)


    if model_path is not None:    
        state_dict = torch.load(model_path)
        model.load_state_dict(state_dict)
        print(f'resume model from {model_path}')
    else:
        print('No model found, initializing random model.')
    if fnet_path is not None:    
        state_dict = torch.load(fnet_path)
        fnet.load_state_dict(state_dict)
        print(f'resume model from {fnet_path}')
    else:
        print('No fnet_model found, initializing random model.')

    
    # if fp16:
    #     model.half()
    model = model.to(device)
    
    if parallel:
        model = nn.DataParallel(model)
    """
    choose optimizer depending on precision type
    """
    # param_optimizer = list(model.named_parameters())
    # fnet_param_optimizer =  list(fnet.named_parameters())
    # no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    # optimizer_grouped_parameters = [
    #     {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
    #     {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    #     ]
    # fnet_optimizer_grouped_parameters = [
    #     {'params': [p for n, p in fnet_param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
    #     {'params': [p for n, p in fnet_param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    #     ]
    # if fp16:
        
    #     optimizer = FusedAdam(optimizer_grouped_parameters,
    #                             lr=learning_rate,
    #                             bias_correction=False,
    #                             max_grad_norm=1.0,
    #                             betas=(beta1,beta2))
    #     optimizer_fnet = FusedAdam(fnet_optimizer_grouped_parameters, lr=plr, bias_correction=False,
    #                             max_grad_norm=1.0,betas=(beta1,beta2))
    #     if loss_scale == 0:
    #         optimizer = FP16_Optimizer(optimizer, dynamic_loss_scale=True)
    #         optimizer = FP16_Optimizer(optimizer_fnet, dynamic_loss_scale=True)
    #     else:
    #         optimizer = FP16_Optimizer(optimizer, static_loss_scale=loss_scale)
    #         optimizer = FP16_Optimizer(optimizer_fnet, static_loss_scale=loss_scale)

    # else:
    if parallel:
        optimizer = torch.optim.Adam(model.module.params(), lr=learning_rate, betas=(beta1,beta2))
    else:
        optimizer = torch.optim.Adam(model.params(), lr=learning_rate, betas=(beta1,beta2))

    optimizer_fnet = torch.optim.Adam(fnet.params(), lr=plr, betas=(beta1,beta2))


    scheduler = lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    scheduler_fnet = lr_scheduler.StepLR(optimizer_fnet, step_size=step_size, gamma=gamma)

    criterion = torch.nn.BCELoss().to(device)
    criterion_oc = CompactLoss().to(device)



    _preproc = get_transform(input_size)['train']#image transformation
    df_train_dataset = FFpp(split='train', frame_nums=frame_nums, transform=_preproc,detect_name = detect_name,compress = compress,type = "Deepfakes",pair = True,original_path=ffpp_original_path,fake_path=ffpp_fake_path)
    f2f_train_dataset = FFpp(split='train', frame_nums=frame_nums, transform=_preproc,detect_name = detect_name,compress = compress,type = 'Face2Face',pair = True,original_path=ffpp_original_path,fake_path=ffpp_fake_path)
    fs_train_dataset = FFpp(split='train', frame_nums=frame_nums, transform=_preproc,detect_name = detect_name,compress = compress,type = 'FaceSwap',pair = True,original_path=ffpp_original_path,fake_path=ffpp_fake_path)
    nt_train_dataset = FFpp(split='train', frame_nums=frame_nums, transform=_preproc,detect_name = detect_name,compress = compress,type = 'NeuralTextures',pair = True,original_path=ffpp_original_path,fake_path=ffpp_fake_path)
    
    datasetlist = [df_train_dataset,f2f_train_dataset,fs_train_dataset,nt_train_dataset]
    # if test_index<len(datasetlist):
    #     del datasetlist[test_index]
    
    _preproc = get_transform(input_size)['test']

    cele_test_dataset = CeleDF(train = False, frame_nums=frame_nums, transform=_preproc,data_root = celebdf_path,split_path = split_path)
    df_test_dataset = FFpp(split='test', frame_nums=frame_nums, transform=_preproc,detect_name = detect_name,compress = compress,type = "Deepfakes",original_path=ffpp_original_path,fake_path=ffpp_fake_path)
    f2f_test_dataset = FFpp(split='test', frame_nums=frame_nums, transform=_preproc,detect_name = detect_name,compress = compress,type = 'Face2Face',original_path=ffpp_original_path,fake_path=ffpp_fake_path)
    fs_test_dataset = FFpp(split='test', frame_nums=frame_nums, transform=_preproc,detect_name = detect_name,compress = compress,type = 'FaceSwap',original_path=ffpp_original_path,fake_path=ffpp_fake_path)
    nt_test_dataset = FFpp(split='test', frame_nums=frame_nums, transform=_preproc,detect_name = detect_name,compress = compress,type = 'NeuralTextures',original_path=ffpp_original_path,fake_path=ffpp_fake_path)
    dfdc_test_dataset = DFDCDetection(root = dfdc_path, train=False, frame_nums=dfdc_frame_nums, transform=_preproc,split_path=dfdc_data_list)
    df_test_dataloader = data.DataLoader(df_test_dataset, batch_size=test_batch_size, shuffle=False, num_workers=8)
    f2f_test_dataloader = data.DataLoader(f2f_test_dataset, batch_size=test_batch_size, shuffle=False, num_workers=8)
    fs_test_dataloader = data.DataLoader(fs_test_dataset, batch_size=test_batch_size, shuffle=False, num_workers=8)
    nt_test_dataloader = data.DataLoader(nt_test_dataset, batch_size=test_batch_size, shuffle=False, num_workers=8)
    cele_test_dataloader = data.DataLoader(cele_test_dataset, batch_size=test_batch_size, shuffle=True, num_workers=8)
    dfdc_test_dataloader = data.DataLoader(dfdc_test_dataset, batch_size=test_batch_size, shuffle=True, num_workers=8)



    model.train()
    
    best_acc = 0.
    best_loss = 1000.
    domain_best_acc = 0.
    domain_best_loss = 1000.
    celedf_best_acc = 0.
    celedf_best_loss = 1000.
    dfdc_best_acc = 0.
    dfdc_best_loss = 1000.

    for epoch in range(epochs):
        '''
        random shuffile all the source domain and split it into training domain and meta domain
        '''
        copydatalist = copy.deepcopy(datasetlist)
        random.seed(epoch)
        random.shuffle(copydatalist)
        meta_dataset = copydatalist[2].cat(copydatalist[2],randomseed = epoch)

        copydatalist[0].set_meta_type(copydatalist[2].type)
        copydatalist[1].set_meta_type(copydatalist[2].type)
        train_dataset = copydatalist[0].cat(copydatalist[1])

        epoch_size = len(train_dataset) //batch_size
        print(f"train dataset is:{copydatalist[0].type},{copydatalist[1].type},meta dataset is:{meta_dataset.type}")
        train_dataloader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8,worker_init_fn=worker_init_fn)

        # warmup_linear = get_linear_schedule_with_warmup(optimizer, num_warmup_steps = warm_up_ratio * epoch_size, num_training_steps = epoch_size)
        # fnet_warmup_linear = get_linear_schedule_with_warmup(optimizer_fnet, num_warmup_steps = warm_up_ratio * epoch_size, num_training_steps = epoch_size) 
        train(model,optimizer,fnet,optimizer_fnet,train_dataloader,None,criterion_oc,epoch,epoch_size,device)
        #train2(model,optimizer,train_dataloader,criterion,epoch,epoch_size,device,meta_dataloader=None)

        scheduler.step()
        scheduler_fnet.step()

        if (epoch + 1) % test_interval == 0:
            '''
            Testing model on serval test-set.
            '''
            dfdc_metrics = test(dfdc_test_dataloader,model,device)
            df_metrics = test(df_test_dataloader, model, device)
            f2f_metrics = test(f2f_test_dataloader, model, device)
            fs_metrics = test(fs_test_dataloader, model, device)
            nt_metrics = test(nt_test_dataloader, model, device)
            celedf_metrics = test(cele_test_dataloader,model,device)
            

            metrics_list = [df_metrics,f2f_metrics,fs_metrics,nt_metrics]
            avg_metrics = EasyDict()
            all_avg_metrics = EasyDict()
            avg_metrics.acc = (df_metrics.acc+f2f_metrics.acc+fs_metrics.acc+nt_metrics.acc)/4
            avg_metrics.loss = (df_metrics.loss+f2f_metrics.loss+fs_metrics.loss+nt_metrics.loss)/4
            avg_metrics.auc = (df_metrics.auc+f2f_metrics.auc+fs_metrics.auc+nt_metrics.auc)/4
            avg_metrics.eer = (df_metrics.eer+f2f_metrics.eer+fs_metrics.eer+nt_metrics.eer)/4

            all_avg_metrics.acc = (df_metrics.acc+f2f_metrics.acc+fs_metrics.acc+nt_metrics.acc+celedf_metrics.acc+dfdc_metrics.acc)/6
            all_avg_metrics.loss = (df_metrics.loss+f2f_metrics.loss+fs_metrics.loss+nt_metrics.loss+celedf_metrics.loss+dfdc_metrics.loss)/6
            all_avg_metrics.auc = (df_metrics.auc+f2f_metrics.auc+fs_metrics.auc+nt_metrics.auc+celedf_metrics.auc+dfdc_metrics.auc)/6
            all_avg_metrics.eer = (df_metrics.eer+f2f_metrics.eer+fs_metrics.eer+nt_metrics.eer+celedf_metrics.eer+dfdc_metrics.eer)/6

            print(f"df acc:{df_metrics.acc:.5f},loss:{df_metrics.loss:.3f},auc:{df_metrics.auc:.3f},eer:{df_metrics.eer:.3f}")
            print(f"f2f acc:{f2f_metrics.acc:.3f},loss:{f2f_metrics.loss:.3f},auc:{f2f_metrics.auc:.3f},eer:{f2f_metrics.eer:.3f}")
            print(f"fs acc:{fs_metrics.acc:.3f},loss:{fs_metrics.loss:.3f},auc:{fs_metrics.auc:.3f},eer:{fs_metrics.eer:.3f}")
            print(f"nt acc:{nt_metrics.acc:.3f},loss:{nt_metrics.loss:.3f},auc:{nt_metrics.auc:.3f},eer:{nt_metrics.eer:.3f}")
            print(f"avg acc:{avg_metrics.acc:.3f},loss:{avg_metrics.loss:.3f},auc:{avg_metrics.auc:.3f},eer:{avg_metrics.eer:.3f}")
            print(f"celedf acc:{celedf_metrics.acc:.3f},loss:{celedf_metrics.loss:.3f},auc:{celedf_metrics.auc:.3f},eer:{celedf_metrics.eer:.3f}")
            print(f"dfdc acc:{dfdc_metrics.acc:.3f},loss:{dfdc_metrics.loss:.3f},auc:{dfdc_metrics.auc:.3f},eer:{dfdc_metrics.eer:.3f}")
            print(f"all_avg acc:{all_avg_metrics.acc:.3f},loss:{all_avg_metrics.loss:.3f},auc:{all_avg_metrics.auc:.3f},eer:{all_avg_metrics.eer:.3f}")
            
            best_acc,best_loss = updatebest(avg_metrics,best_acc,best_loss,"avg",model,fnet)
            #domain_best_acc,domain_best_loss = updatebest(metrics_list[test_index],domain_best_acc,domain_best_loss,"domain",model,pnet)
            celedf_best_acc,celedf_best_loss = updatebest(celedf_metrics,celedf_best_acc,celedf_best_loss,"celedf",model,fnet)
            dfdc_best_acc,dfdc_best_loss = updatebest(dfdc_metrics,dfdc_best_acc,dfdc_best_loss,"dfdc",model,fnet)
            save_checkpoint(model.state_dict(), fpath=f'{save_dir}/{model_name}_lastepoch.pth')
            save_checkpoint(fnet.state_dict(), fpath=f'{save_dir}/{model_name}_pnet_lastepoch.pth')
            
    print(f'save dir :{save_dir} done!!!')
    
def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = master_port

    # initialize the process group
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

if __name__ == '__main__':
    if DETERMINSTIC:
        np.random.seed(RNG_SEED)         
        torch.manual_seed(RNG_SEED)
        torch.cuda.manual_seed(RNG_SEED)           
        torch.cuda.manual_seed_all(RNG_SEED)         
        random.seed(RNG_SEED)         
        torch.backends.cudnn.benchmark = False         
        torch.backends.cudnn.deterministic = True         
        torch.backends.cudnn.enabled = False   
    main()
    # mp.spawn(main,
    #          args=(world_size,),
    #          nprocs=world_size,
    #          join=True)
