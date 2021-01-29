import torch
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
import torchvision
from torchvision import transforms
import torch.optim as optim
from torch import nn
import matplotlib.pyplot as plt
 
class Encoder(torch.nn.Module):
    def __init__(self, D_in, H, D_out):
        super(Encoder, self).__init__()
        self.linear1 = torch.nn.Linear(D_in, H)
        self.linear2 = torch.nn.Linear(H, D_out)

    def forward(self, x):
        x = torch.relu(self.linear1(x))
        return torch.sigmoid(self.linear2(x))


class Decoder(torch.nn.Module):
    def __init__(self, D_in, H, D_out):
        super(Decoder, self).__init__()
        self.linear1 = torch.nn.Linear(D_in, H)
        self.linear2 = torch.nn.Linear(H, D_out)

    def forward(self, x):
        x = torch.relu(self.linear1(x))
        return  torch.sigmoid(self.linear2(x))


class VAE(torch.nn.Module): 
    def __init__(self, encoder, decoder, h_dim, latent_z):
        super(VAE, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self._enc_mu = torch.nn.Linear(h_dim, latent_z)
        self._enc_log_sigma = torch.nn.Linear(h_dim,latent_z)

    def _sample_latent(self, h_enc):
        """
        Return the latent normal sample z ~ N(mu, sigma^2)
        """
        mu = self._enc_mu(h_enc)
        sigma = self._enc_log_sigma(h_enc)
        sigma = torch.exp(0.5*sigma)

        self.z_mean = mu
        self.z_sigma = sigma

        std_z = torch.from_numpy(np.random.normal(mu, sigma, size=sigma.size())).float()    
        return mu + sigma * Variable(std_z, requires_grad=False)  # Reparameterization trick

    def forward(self, state):
        h_enc = self.encoder(state)
        z = self._sample_latent(h_enc)
        return self.decoder(z)


def ELBO_loss(x, y, z_mean, z_stddev):
    
    reconstruction_loss = torch.sum(x*torch.log(y)+(1-x)*torch.log(1-y),1)
    KL_divergence = 0.5 *torch.sum(torch.square(z_mean) + torch.square(z_stddev) - torch.log(torch.square(z_stddev)) - 1,1)
    
    reconstruction_loss = torch.mean(reconstruction_loss)
    KL_divergence = torch.mean(KL_divergence)

    ELBO = reconstruction_loss - KL_divergence

    loss = -ELBO

    return loss


if __name__ == '__main__':

    input_dim = 28 * 28
    batch_size = 32
    h_dim = 500
    latent_z = 20
    
    
    transform = transforms.Compose([transforms.ToTensor()])
    mnist = torchvision.datasets.MNIST('./', download=True, transform=transform)

    dataloader = torch.utils.data.DataLoader(mnist, batch_size=batch_size,
                                             shuffle=True, num_workers=2)

    print('Number of samples: ', len(mnist))

    encoder = Encoder(input_dim, h_dim, h_dim)
    decoder = Decoder(latent_z, h_dim, input_dim)
    vae = VAE(encoder, decoder,h_dim, latent_z) 

    optimizer = optim.Adam(vae.parameters(), lr=0.0001)
    l = None
    for epoch in range(30):
        for i, data in enumerate(dataloader, 0):
            inputs, classes = data
            inputs, classes = Variable(inputs.resize_(batch_size, input_dim))  , Variable(classes) 
            optimizer.zero_grad()
            dec = vae(inputs)
            loss = ELBO_loss(inputs, dec, vae.z_mean, vae.z_sigma) 
            loss.backward()
            optimizer.step()  
        print(epoch, loss.data )

    plt.imshow(vae(inputs).data[0].numpy().reshape(28, 28), cmap='gray')
    plt.show(block=True)
    plt.imshow(inputs.data[0].numpy().reshape(28, 28), cmap='gray')
    plt.show(block=True)
