import torch
from torch import nn
from torch.nn import functional as F 


# class VAE(nn.Module):
#     def __init__(self, image_size=784, h_dim=400, z_dim=20):
#         super().__init__()
#         self.fc1 = nn.Linear(image_size, h_dim)
#         self.fc2 = nn.Linear(h_dim, z_dim)
#         self.fc3 = nn.Linear(h_dim, z_dim)
#         self.fc4 = nn.Linear(z_dim, h_dim)
#         self.fc5 = nn.Linear(h_dim, image_size)
#         self.sigmoid = nn.Sigmoid()

#     def encode(self, x):
#         h = F.relu(self.fc1(x))
#         return self.fc2(h), self.fc3(h)

#     def reparameterize(self, mu, log_var):
#         std = torch.exp(log_var/2)
#         eps = torch.rand_like(std)
#         return mu + eps * std

#     def decode(self, z):
#         h = F.relu(self.fc4(z))
#         return self.sigmoid(self.fc5(h))

#     def forward(self, x):
#         mu, log_var = self.encode(x)
#         z = self.reparameterize(mu, log_var)
#         x_reconst = self.decode(z)
#         return x_reconst, mu, log_var
    
# class VAE(nn.Module):
#     def __init__(self, image_size=784, h_dim=400, z_dim=20):
#         super().__init__()
#         self.fc1 = nn.Linear(image_size, h_dim)
#         self.fc2 = nn.Linear(h_dim, z_dim)
#         self.fc3 = nn.Linear(h_dim, z_dim)
#         self.fc4 = nn.Linear(z_dim, h_dim)
#         self.fc5 = nn.Linear(h_dim, image_size)

#     def encode(self, x):
#         h = F.leaky_relu(self.fc1(x), negative_slope=0.2)
#         return self.fc2(h), self.fc3(h)

#     def reparameterize(self, mu, log_var):
#         std = torch.exp(log_var/2)
#         eps = torch.rand_like(std)
#         return mu + eps * std

#     def decode(self, z):
#         h = F.leaky_relu(self.fc4(z), negative_slope=0.2)
#         return F.sigmoid(self.fc5(h))

#     def forward(self, x):
#         mu, log_var = self.encode(x)
#         z = self.reparameterize(mu, log_var)
#         x_reconst = self.decode(z)
#         return x_reconst, mu, log_var

class VAE(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=400, latent_dim=20):
        super(VAE, self).__init__()

        # encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, latent_dim),
            nn.LeakyReLU(0.2)
            )
        
        # latent mean and variance 
        self.mean_layer = nn.Linear(latent_dim, 2)
        self.logvar_layer = nn.Linear(latent_dim, 2)
        
        # decoder
        self.decoder = nn.Sequential(
            nn.Linear(2, latent_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(latent_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid()
            )
     
    def encode(self, x):
        x = self.encoder(x)
        mean, logvar = self.mean_layer(x), self.logvar_layer(x)
        return mean, logvar

    def reparameterization(self, mean, var):
        epsilon = torch.randn_like(var)  
        z = mean + var*epsilon
        return z
    
    def decode(self, x):
        return self.decoder(x)

    def forward(self, x):
        mean, logvar = self.encode(x)
        z = self.reparameterization(mean, logvar)
        x_hat = self.decode(z)
        return x_hat, mean, logvar
