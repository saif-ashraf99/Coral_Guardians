import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as dset
import torchvision.utils as vutils

from dcgan_models import DCGANGenerator, DCGANDiscriminator

def train_dcgan(
    real_images_folder='./data/real_images',
    output_folder='./data/synthetic_images',
    batch_size=64, image_size=64, nz=100,
    num_epochs=50, lr=0.0002, beta1=0.5, device='cuda'
):
    """
    Trains a DCGAN on real underwater images (e.g., coral reef images).
    """

    # Create dataset
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))  # for all 3 channels
    ])

    dataset = dset.ImageFolder(root=real_images_folder, transform=transform)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Models
    netG = DCGANGenerator(nz=nz).to(device)
    netD = DCGANDiscriminator().to(device)

    # Loss function and optimizers
    criterion = nn.BCELoss()
    optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))

    fixed_noise = torch.randn(batch_size, nz, 1, 1, device=device)

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    print("Starting Training Loop...")
    for epoch in range(num_epochs):
        for i, (data, _) in enumerate(dataloader):
            netD.zero_grad()

            # Train discriminator with real images
            real_cpu = data.to(device)
            b_size = real_cpu.size(0)
            label = torch.full((b_size,), 1., dtype=torch.float, device=device)
            output = netD(real_cpu).view(-1)
            errD_real = criterion(output, label)
            errD_real.backward()
            D_x = output.mean().item()

            # Train discriminator with fake images
            noise = torch.randn(b_size, nz, 1, 1, device=device)
            fake = netG(noise)
            label.fill_(0.)
            output = netD(fake.detach()).view(-1)
            errD_fake = criterion(output, label)
            errD_fake.backward()
            D_G_z1 = output.mean().item()
            errD = errD_real + errD_fake
            optimizerD.step()

            # Train generator
            netG.zero_grad()
            label.fill_(1.)
            output = netD(fake).view(-1)
            errG = criterion(output, label)
            errG.backward()
            D_G_z2 = output.mean().item()
            optimizerG.step()

            if i % 50 == 0:
                print(f"[{epoch}/{num_epochs}][{i}/{len(dataloader)}] "
                      f"Loss_D: {errD.item():.4f} Loss_G: {errG.item():.4f} "
                      f"D(x): {D_x:.4f} D(G(z)): {D_G_z1:.4f}/{D_G_z2:.4f}")

        # Save fake images each epoch
        fake = netG(fixed_noise)
        vutils.save_image(fake.detach(),
                          f"{output_folder}/fake_samples_epoch_{epoch}.png",
                          normalize=True)

    # Save final models
    torch.save(netG.state_dict(), os.path.join(output_folder, 'dcgan_generator.pth'))
    torch.save(netD.state_dict(), os.path.join(output_folder, 'dcgan_discriminator.pth'))
    print("Training complete and models saved.")
