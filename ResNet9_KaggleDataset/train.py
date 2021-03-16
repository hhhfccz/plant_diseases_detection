from model import *
from config import *
import torch


@torch.no_grad()
def evaluate(model, val_loader):
    model.eval()
    outputs = [model.validation_step(batch) for batch in val_loader]
    return model.validation_epoch_end(outputs)


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']
    

def fit_cycles(epochs, max_lr, model, train_loader, val_loader, weight_decay=0,
               grad_clip=None, opt_func=torch.optim.SGD):
    torch.cuda.empty_cache()
    history = []
    
    optimizer = opt_func(model.parameters(), max_lr, weight_decay=weight_decay)
    # scheduler for one cycle learniing rate
    sched = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr, epochs=epochs, steps_per_epoch=len(train_loader))    

    for epoch in range(epochs):
        model.train()
        train_losses = []
        lrs = []
        for batch in train_loader:
            loss = model.training_step(batch)
            train_losses.append(loss)
            loss.backward()
            
            # gradient clipping
            if grad_clip: 
                nn.utils.clip_grad_value_(model.parameters(), grad_clip)
                
            optimizer.step()
            optimizer.zero_grad()
            
            # recording and updating learning rates
            lrs.append(get_lr(optimizer))
            sched.step()
    
        # validation
        result = evaluate(model, val_loader)
        result['train_loss'] = torch.stack(train_losses).mean().item()
        result['lrs'] = lrs
        model.epoch_end(epoch, result)
        history.append(result)
        
    return history


def main():
    train_dir, valid_dir = config()
    train_folder, valid_folder, train_data, valid_data = get_data(train_dir, valid_dir)

    model = to_device(ResNet9(3, len(train_folder.classes)), device=get_device())
    print(model)
    INPUT_SHAPE = (3, 256, 256)
    print(summary(model, (INPUT_SHAPE)))
    
    history = [evaluate(model, valid_data)]
    epochs, max_lr, grad_clip, weight_decay, opt_func = train_config()
    history += fit_cycles(epochs, max_lr, model, train_data, valid_data, grad_clip, weight_decay, opt_func)
    print(history)
    
    PATH = "./ResNet9_KaggleDataset.pth"
    torch.save(model, PATH)


if __name__ == '__main__':
    main()
