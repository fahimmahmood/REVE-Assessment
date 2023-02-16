import time
import torch
import torch.nn.functional as F

def compute_accuracy(model, data_loader, device):

    with torch.no_grad():

        correct_pred, num_examples = 0, 0

        for i, (features, targets) in enumerate(data_loader):

            features = features.to(device)
            targets = targets.float().to(device)

            logits = model(features)
            _, predicted_labels = torch.max(logits, 1)

            num_examples += targets.size(0)
            correct_pred += (predicted_labels == targets).sum()
    return correct_pred.float()/num_examples * 100

def trainer(model,optimizer,train_loader,valid_loader,test_loader,NUM_EPOCHS,DEVICE):
    start_time = time.time()
    best_loss = float('inf')
    for epoch in range(NUM_EPOCHS):
        model.train()
        for batch_idx, batch_data in enumerate(train_loader):
            
            text = batch_data.TEXT_COLUMN_NAME.to(DEVICE)
            labels = batch_data.LABEL_COLUMN_NAME.to(DEVICE)

            ### FORWARD AND BACK PROP
            logits = model(text)
            loss = F.cross_entropy(logits, labels)
            optimizer.zero_grad()
            
            loss.backward()
            
            ### UPDATE MODEL PARAMETERS
            optimizer.step()
            if loss < best_loss:
                best_loss = loss
                torch.save({
                    'epoch':epoch,
                    'model_state_dict':model.state_dict(),
                    'optimizer_state_dict':optimizer.state_dict(),
                    'loss':loss,
                },"model_checkpoint.pt")
            
            ### LOGGING
            if not batch_idx % 50:
                print (f'Epoch: {epoch+1:03d}/{NUM_EPOCHS:03d} | '
                    f'Batch {batch_idx:03d}/{len(train_loader):03d} | '
                    f'Loss: {loss:.4f}')

        with torch.set_grad_enabled(False):
            print(f'training accuracy: '
                f'{compute_accuracy(model, train_loader, DEVICE):.2f}%'
                f'\nvalid accuracy: '
                f'{compute_accuracy(model, valid_loader, DEVICE):.2f}%')
            
        print(f'Time elapsed: {(time.time() - start_time)/60:.2f} min')
        
    print(f'Total Training Time: {(time.time() - start_time)/60:.2f} min')
    print(f'Test accuracy: {compute_accuracy(model, test_loader, DEVICE):.2f}%')