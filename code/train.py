from tqdm import tqdm
import torch
from torchmetrics import R2Score
import numpy as np
import pandas as pd
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
loss_fn = torch.nn.MSELoss()
mae_loss_fn = torch.nn.L1Loss()
smoothl1_loss_fn=torch.nn.SmoothL1Loss()
r2_score=R2Score().to(device)




def get_metrics(model, data_loader):
    valid_outputs = []
    valid_labels = []
    valid_loss = []
    valid_mae_loss = []
    valid_r2=[]
    valid_map=[]
    for voc_graphs, oxidant_graphs, voc_lens, oxidant_lens, labels in tqdm(data_loader):
        outputs, i_map = model(
            [voc_graphs.to(device), oxidant_graphs.to(device), torch.tensor(voc_lens).to(device),
             torch.tensor(oxidant_lens).to(device)])
        loss = loss_fn(outputs, torch.tensor(labels).to(device).float())
        mae_loss = mae_loss_fn(outputs, torch.tensor(labels).to(device).float())
        '''
        r2=r2_score(outputs,torch.tensor(labels).to(device).float())
        '''

        valid_outputs += outputs.cpu().detach().numpy().tolist()
        valid_loss.append(loss.cpu().detach().numpy())
        valid_mae_loss.append(mae_loss.cpu().detach().numpy())
        '''
        valid_r2.append(r2.cpu().detach().numpy())
        '''
        valid_labels += labels
        valid_map.append(i_map.cpu().detach().numpy().tolist())

    loss = np.mean(np.array(valid_loss).flatten())
    mae_loss = np.mean(np.array(valid_mae_loss).flatten())
    r2=r2_score(torch.tensor(valid_outputs).to(device).float(),torch.tensor(valid_labels).to(device).float())
    return loss, mae_loss,r2, valid_outputs,valid_labels



def train(max_epochs, model, optimizer, scheduler, train_loader, valid_loader, project_name):
    best_val_loss = 100
    consecutiveepoch_num=0
    now_train_loss=0
    now_mae=0
    now_r2=0
    best_outputs=[]
    for epoch in range(max_epochs):
        
        model.train()
        running_loss = []
        train_mae=[]
        train_r2=[]
        train_output=[]
        tq_loader = tqdm(train_loader)
        o = {}
        for samples in tq_loader:
            optimizer.zero_grad()
            outputs, interaction_map = model(
                [samples[0].to(device), samples[1].to(device), torch.tensor(samples[2]).to(device),
                torch.tensor(samples[3]).to(device)])
            l1_norm = torch.norm(interaction_map, p=2) * 1e-4
            
            loss=smoothl1_loss_fn(outputs,torch.tensor(samples[4]).to(device).float()) + l1_norm
            mae=mae_loss_fn(outputs,torch.tensor(samples[4]).to(device).float())
              
            loss.backward()
            optimizer.step()
            loss = loss - l1_norm
            running_loss.append(loss.cpu().detach())
            train_mae.append(mae.cpu().detach())
            train_output += outputs.cpu().detach().numpy().tolist()
            tq_loader.set_description(
                "Epoch: " + str(epoch + 1) + "  Training loss: " + str(np.mean(np.array(running_loss))))
        model.eval()
        val_loss, mae_loss,r2, outputs,labels = get_metrics(model, valid_loader)
        scheduler.step(val_loss)
        
        with open('train_loss.txt','a') as f:
            f.write(str(epoch+1)+'\t'+str(np.mean(np.array(running_loss)))+'\t'+str(np.mean(np.array(train_mae)))+'\t'+
                    str(val_loss)+'\t'+str(mae_loss)+'\t'+str(r2)+'\n')
            f.close()
        
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            consecutiveepoch_num=0
            now_train_loss=str(np.mean(np.array(running_loss)))
            now_mae=mae_loss 
            now_r2=r2
            best_outputs=outputs
            print("Epoch: " + str(epoch + 1) + "  train_loss " + str(np.mean(np.array(running_loss))) + " Val_loss " + str(val_loss) + " MAE Val_loss " + str(mae_loss)+'Val_R2'+str(r2))
            torch.save(model.state_dict(), "D:/ki/bayesian_seed/best_model/"+str(best_val_loss)+".tar")
            

        else:
            consecutiveepoch_num+=1

        if consecutiveepoch_num>=15:
            break
            
    preds=pd.DataFrame({'labels':labels,'preds':best_outputs})
    preds.to_csv(str(best_val_loss)+'.txt',sep='\t')
       
    return best_val_loss
