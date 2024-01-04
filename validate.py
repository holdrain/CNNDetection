import torch
import numpy as np
from networks.resnet import resnet50
from sklearn.metrics import average_precision_score, precision_recall_curve, accuracy_score,roc_auc_score
from options.test_options import TestOptions
from data import create_dataloader
from tqdm.auto import tqdm


def validate(model, opt):
    data_loader = create_dataloader(opt)

    with torch.no_grad():
        y_true, y_pred = [], []
        for img, label in data_loader:
            in_tens = img.cuda()
            y_pred.extend(model(in_tens).sigmoid().flatten().tolist())
            y_true.extend(label.flatten().tolist())

    y_true, y_pred = np.array(y_true), np.array(y_pred)
    r_acc = accuracy_score(y_true[y_true==0], y_pred[y_true==0] > 0.5)
    f_acc = accuracy_score(y_true[y_true==1], y_pred[y_true==1] > 0.5)
    acc = accuracy_score(y_true, y_pred > 0.5)
    ap = average_precision_score(y_true, y_pred)
    return acc, ap, r_acc, f_acc, y_true, y_pred

# def Custom_validate(model,dl,accelerator):
#     count = 0
#     with torch.no_grad():
#         all_y_true, all_y_pred = [],[]
#         with tqdm(initial=0,total=len(dl),disable=not accelerator.is_main_process) as pbar:
#             for img, label in dl:
#                 in_tens = img
#                 y_pred = torch.softmax(model(in_tens),dim=1)
#                 y_true = label.flatten()
#                 # gather y_pred and y_true
#                 p,t = accelerator.gather_for_metrics((y_pred,y_true))
#                 all_y_pred.extend(p.tolist()) #归一化
#                 all_y_true.extend(t.tolist())
#                 pbar.update(1)
#                 count += 1
#                 if count == 30:
#                     break

#     all_y_pred = torch.Tensor(all_y_pred)
#     all_y_true = torch.Tensor(all_y_true)
#     all_y_true, all_y_pred = np.array(all_y_true), np.array(all_y_pred)
#     r_acc = accuracy_score(all_y_true[all_y_true==0], np.argmax(all_y_pred[all_y_true==0],dim=1))
#     f_acc = accuracy_score(all_y_true[all_y_true==1], np.argmax(all_y_pred[all_y_true==1],dim=1))
#     acc = accuracy_score(all_y_true, np.argmax(all_y_pred,dim=1))
#     ap = average_precision_score(all_y_true, all_y_pred[1])
#     return acc, ap, r_acc, f_acc, y_true, y_pred

def Custom_validate(model,dl,accelerator):
    with torch.inference_mode():
        y_true, y_pred = [], []
        for img, label in tqdm(dl,position=0,desc=f"validating",disable=not accelerator.is_main_process):
            in_tens = img
            y_p = model(in_tens).softmax(1)
            all_y_p,all_label = accelerator.gather_for_metrics((y_p,label))
            y_pred.extend(all_y_p.tolist()) 
            y_true.extend(all_label.flatten().tolist())
        y_true, y_pred = np.array(y_true), np.array(y_pred,dtype=np.float32)
        r_acc = accuracy_score(y_true[y_true==0], np.argmax(y_pred[y_true==0],axis=1))
        f_acc = accuracy_score(y_true[y_true==1], np.argmax(y_pred[y_true==1],axis=1))
        acc = accuracy_score(y_true, np.argmax(y_pred,axis=1))
    try:
        ap = average_precision_score(y_true, y_pred[:,1])
    except Exception as e:
        print(e)
        ap = -1
    try:
        auc = roc_auc_score(y_true,y_pred[:,1])
    except Exception as e:
        print(e)
        auc = -1
    return acc, ap, r_acc, f_acc, auc, y_true, y_pred

if __name__ == '__main__':
    opt = TestOptions().parse(print_options=False)

    model = resnet50(num_classes=1)
    state_dict = torch.load(opt.model_path, map_location='cpu')
    model.load_state_dict(state_dict['model'])
    model.cuda()
    model.eval()

    acc, avg_precision, r_acc, f_acc, y_true, y_pred = validate(model, opt)

    print("accuracy:", acc)
    print("average precision:", avg_precision)

    print("accuracy of real images:", r_acc)
    print("accuracy of fake images:", f_acc)
