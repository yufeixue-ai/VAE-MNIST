import torch
import torch.nn as nn

def loss_function(x_reconst, x, mu, log_var):
      # BCE_loss = nn.BCELoss(reduction='sum')
      # reconstruction_loss = BCE_loss(x_reconst, x)
      reconstruction_loss = nn.functional.binary_cross_entropy(x_reconst, x, reduction='sum')
      # KL_divergence = -0.5 * torch.sum(1 + log_var - torch.exp(log_var) - mu ** 2)
      KL_divergence = - 0.5 * torch.sum(1+ log_var - mu.pow(2) - log_var.exp())
      return reconstruction_loss + KL_divergence, reconstruction_loss, KL_divergence

def train(model, 
          epoch, 
          data_loader, 
          optimizer, 
          log_interval,
          device):
      model.train()
      losses=[]
      recon_losses=[]
      kl_losses=[]
      for batch_idx, (data, _) in enumerate(data_loader):
            batch_size = data.size(0)
            img = data.to(device)
            img_flat = torch.flatten(img, start_dim=1)
            # forward
            optimizer.zero_grad()
            gen_img, mu, log_var = model(img_flat)
            loss, recon_loss, kl_loss = loss_function(gen_img, img_flat, mu, log_var)
            loss = loss/batch_size
            recon_loss = recon_loss/batch_size
            kl_loss = kl_loss/batch_size
            losses.append(loss.item())
            recon_losses.append(recon_loss.item())
            kl_losses.append(kl_loss.item())
            # backward
            loss.backward()
            optimizer.step()
            # infomation
            if batch_idx % log_interval == 0:
                  print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.4f}\tReconLoss: {:.4f}\tKLLoss: {:.4f}'.format(
                        epoch, 
                        batch_idx * len(data), 
                        len(data_loader.dataset),
                        100. * batch_idx / len(data_loader),
                        loss.item(),
                        recon_loss.item(),
                        kl_loss.item()
                        )
                  )
      print('====> Epoch: {} AvrgLoss: {:.4f} ReconLoss: {:.4f} KLLoss: {:.4f}'.format(
            epoch, 
            torch.mean(torch.tensor(losses)),
            torch.mean(torch.tensor(recon_losses)),
            torch.mean(torch.tensor(kl_losses))
            )
      )
      
def test(model, 
         epoch, 
         data_loader,
         device):
      model.eval()
      losses = []
      recon_losses = []
      kl_losses = []
      with torch.no_grad():
            for data, _ in data_loader:
                  batch_size = data.size(0)
                  img = data.to(device)
                  img_flat = torch.flatten(img, start_dim=1)
                  # forward
                  gen_img, mu, log_var = model(img_flat)
                  loss, recon_loss, kl_loss = loss_function(gen_img, img_flat, mu, log_var)
                  loss = loss/batch_size
                  recon_loss = recon_loss/batch_size
                  kl_loss = kl_loss/batch_size  
                  losses.append(loss.item())
                  recon_losses.append(recon_loss.item())
                  kl_losses.append(kl_loss.item())
      
      print(' Test Epoch: {} AvrgLoss: {:.4f} ReconLoss: {:.4f} KLLoss: {:.4f}'.format(
            epoch, 
            torch.mean(torch.tensor(losses)),
            torch.mean(torch.tensor(recon_losses)),
            torch.mean(torch.tensor(kl_losses))
            )
      )
      print("<------------------------------------------------------------->")
    
    