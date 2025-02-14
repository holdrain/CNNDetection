import os
import sys
import time
import torch
import torch.nn
import argparse
import wandb
import torch.multiprocessing as mp


from PIL import Image
from tqdm.auto import tqdm
from validate import validate,Custom_validate
from data import create_dataloader
from earlystop import EarlyStopping
from networks.trainer import Trainer
from options.train_options import TrainOptions
from accelerate import Accelerator
from util import *
from models.model_choice import model_dic

"""Currently assumes jpg_prob, blur_prob 0 or 1"""
def get_val_opt():
    val_opt = TrainOptions().parse(print_options=False)
    val_opt.dataroot = '{}/{}/'.format(val_opt.dataroot, val_opt.val_split)
    val_opt.isTrain = False
    val_opt.no_resize = False
    val_opt.no_crop = False
    val_opt.serial_batches = True
    val_opt.jpg_method = ['pil']
    if len(val_opt.blur_sig) == 2:
        b_sig = val_opt.blur_sig
        val_opt.blur_sig = [(b_sig[0] + b_sig[1]) / 2]
    if len(val_opt.jpg_qual) != 1:
        j_qual = val_opt.jpg_qual
        val_opt.jpg_qual = [int((j_qual[0] + j_qual[-1]) / 2)]

    return val_opt


if __name__ == '__main__':
    opt = TrainOptions().parse()
    opt.dataroot = '{}/{}/'.format(opt.dataroot, opt.train_split)
    val_opt = get_val_opt()
    
    # set launch method 
    mp.set_start_method("spawn")

    # multiple GPUs
    accelerator = Accelerator(
        split_batches = True,
        dispatch_batches = False,
    )
    # set print method and seed
    setup_for_distributed(accelerator.is_main_process)
    set_seeds(opt.seed)

    # dl,vdl and model
    dl = create_dataloader(opt)
    vdl = create_dataloader(val_opt)
    print('#training images = %d' % len(dl))
    print('#validating images = %d' % len(vdl))
    # weights is none in default

    model = model_dic[opt.arch]

    # optimizer
    if opt.optim == 'adam':
        optimizer = torch.optim.Adam(model.parameters(),lr=opt.lr, betas=(opt.beta1, 0.999))
    elif opt.optim == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(),lr=opt.lr, momentum=0.0, weight_decay=0)
    else:
        raise ValueError("optim should be [adam, sgd]")

    # wandb
    if accelerator.is_main_process and not opt.debug:
        # init wandb logging
        if opt.continue_train:
            # project为项目名称，config用来记录当前实验的参数配置，id是本次实验的唯一标识符
            # must代表强制恢复一个之前中断的实验，恢复失败则报错，id，name不是必要参数
            wandb.init(project=opt.wandb_project, config=opt, id=opt.wandb_id,resume="must")
        else:
            wandb.init(project=opt.wandb_project, name = opt.name+"_"+opt.arch, id = opt.wandb_id ,config=opt)

    # continue training
    if opt.continue_train:
        raise NotImplemented("continue training!")

    # earlystopper
    if opt.continue_train and accelerator.is_main_process:
        raise NotImplemented("continue training!")
    else:
        early_stopping = EarlyStopping(accelerator,patience=opt.earlystop_epoch, delta=-0.001, verbose=True)
    

    trainer = Trainer(opt,accelerator,model,optimizer)
    dl = accelerator.prepare(dl)
    vdl = accelerator.prepare(vdl)

    for epoch in range(opt.niter):
        epoch_start_time = time.time()
        iter_data_time = time.time()
        epoch_iter = 0
        with tqdm(initial = 0,total = len(dl),disable = not accelerator.is_main_process) as pbar:
            for i,data in enumerate(dl):
                trainer.total_steps += 1
                epoch_iter += opt.batch_size

                trainer.set_input(data)
                trainer.optimize_parameters()

                if trainer.total_steps % opt.loss_freq == 0 and accelerator.is_main_process:
                    print("Train loss: {} at step: {}".format(trainer.loss, trainer.total_steps))
                    if not opt.debug:
                        wandb.log({'Train loss':trainer.loss})

                if trainer.total_steps % opt.save_latest_freq == 0 and accelerator.is_main_process:
                    print('saving the latest model %s (epoch %d, model.total_steps %d)' %
                        (opt.name, epoch, trainer.total_steps))
                    trainer.save_networks('latest',accelerator)
                
                pbar.update(1)
                # if i == 30:
                #     accelerator.wait_for_everyone()
                #     break

            if epoch % opt.save_epoch_freq == 0 and accelerator.is_main_process:
                print('saving the model at the end of epoch %d, iters %d' %
                    (epoch, trainer.total_steps))
                trainer.save_networks('latest',accelerator)
                trainer.save_networks(epoch,accelerator)

            # Validation
            trainer.eval()
            acc, ap = Custom_validate(trainer.model, vdl,accelerator)[:2]
            if not opt.debug and accelerator.is_main_process:
                wandb.log({'accuracy':acc,'ap':ap})
            print("(Val @ epoch {}) acc: {}; ap: {}".format(epoch, acc, ap))
            early_stopping(acc, trainer)
            if early_stopping.early_stop:
                cont_train = trainer.adjust_learning_rate()
                if cont_train:
                    print("Learning rate dropped by 10, continue training...")
                    early_stopping = EarlyStopping(accelerator,patience=opt.earlystop_epoch, delta=-0.002, verbose=True)
                else:
                    print("Early stopping.")
                    accelerator.set_trigger()
                    if accelerator.check_trigger():
                        break
            trainer.train()

