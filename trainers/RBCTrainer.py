import os
from glob import glob
import torch
from .BaseTrainer import BaseTrainer


class RBCTrainer(BaseTrainer):
    def __init__(self, arg, G, torch_device, recon_loss, logger):
        super().__init__(arg, torch_device, logger)

        self.G = G
        self.recon_loss = recon_loss
        #self.optim = torch.optim.Adam(G.parameters(), lr=arg.lrG, betas=arg.beta)
        self.optimizer=torch.optim.Adam(G.parameters(),lr=arg.lr_init, betas=arg.beta, weight_decay=arg.l2_coeff)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=100, gamma=0.99)

        self.best_metric = -1

    def save(self, epoch, filename):
        """Save current epoch model

        Save Elements:
            model_type : arg.model
            start_epoch : current epoch
            network : network parameters
            optimizer: optimizer parameters
            best_metric : current best score

        Parameters:
            epoch : current epoch   
            filename : model save file name
        """
        if epoch < 10:
            return
            
        torch.save({"model_type" : self.model_type,
                    "start_epoch" : epoch + 1,
                    "network" : self.G.state_dict(),
                    "optimizer" : self.optimizer.state_dict(),
                    "best_metric": self.best_metric
                    }, self.save_path + "/%s.pth.tar"%(filename))
        print("Model saved %d epoch"%(epoch))

    def load(self, filename=None):
        """ Model load. same with save"""
        if filename is None:
            # load last epoch model
            filename = sorted(glob(self.save_dir + "/*.pth.tar"))[-1]

        if os.path.exists(self.save_path + filename) is True:
            print("Load %s File"%(self.save_path))            
            ckpoint = torch.load(self.save_path + filename)                            
            if ckpoint["model_type"] != self.model_type:
                raise ValueError("Ckpoint Model Type is %s"%(ckpoint["model_type"]))

            self.G.load_state_dict(ckpoint['network'])
            self.optimizer.load_state_dict(ckpoint['optimizer'])
            self.start_epoch = ckpoint['start_epoch']
            self.best_metric = ckpoint["best_metric"]
            print("Load Model Type : %s, epoch : %d acc : %f"%(ckpoint["model_type"], self.start_epoch, self.best_metric))
        else:
            print("Load Failed, not exists file")

    def train(self, train_loader, val_loader=None):
        print("\nStart Train")
        for epoch in range(self.start_epoch, self.epoch):
            for i, (input_, target_) in enumerate(train_loader):
                self.scheduler.step()
                self.G.train()
                input_, target_ = input_.to(self.torch_device), target_.to(self.torch_device)
                output_ = self.G(input_)
                recon_loss = self.recon_loss(output_, target_)
                
                self.optimizer.zero_grad()
                recon_loss.backward()
                self.optimizer.step()
            
                if (i % 40) == 0:
                    self.logger.log_write("train", epoch=epoch, loss=recon_loss.item())

            if val_loader is not None:            
                self.valid(epoch, val_loader)
            else:
                self.save(epoch)
                
    def valid(self, epoch, val_loader):
        self.G.eval()
        with torch.no_grad():
            correct = 0
            for i, (input_, target_) in enumerate(val_loader):
                input_, target_ = input_.to(self.torch_device), target_.to(self.torch_device)                
                output_ = self.G(input_).max(1, keepdim=True)[1]
                correct += output_.eq(target_.view_as(output_)).sum().item()

            acc = correct / len(val_loader.dataset)
            self.logger.log_write("valid", epoch=epoch, acc=acc)

            if acc > self.best_metric:
                self.best_metric = acc
                self.save(epoch, "epoch[%04d]_acc[%f]"%(epoch, acc))

    def test(self, test_loader):
        pass
