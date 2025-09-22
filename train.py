import torch
from dataset import RealGhibliDataset
import sys
from utils import save_checkpoint, load_checkpoint
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import config
from tqdm import tqdm
from torchvision.utils import save_image
from discriminator_model import Discriminator
from generator_model import Generator


def train_fn(
    disc_R, disc_G, gen_G, gen_R, loader, opt_disc, opt_gen, l1, mse, d_scaler, g_scaler
):
    R_reals = 0
    R_fakes = 0
    loop = tqdm(loader, leave=True)

    for idx, (ghibli, real) in enumerate(loop):
        ghibli = ghibli.to(config.DEVICE)
        real = real.to(config.DEVICE)

        # Train Discriminators R and G
        with torch.cuda.amp.autocast():
            fake_real = gen_R(ghibli)
            D_R_real = disc_R(real)
            D_R_fake = disc_R(fake_real.detach())
            D_R_real_loss = mse(D_R_real, torch.ones_like(D_R_real))
            D_R_fake_loss = mse(D_R_fake, torch.zeros_like(D_R_fake))
            D_R_loss = D_R_real_loss + D_R_fake_loss

            fake_ghibli = gen_G(real)
            D_G_real = disc_G(ghibli)
            D_G_fake = disc_G(fake_ghibli.detach())
            D_G_real_loss = mse(D_G_real, torch.ones_like(D_G_real))
            D_G_fake_loss = mse(D_G_fake, torch.zeros_like(D_G_fake))
            D_G_loss = D_G_real_loss + D_G_fake_loss

            # put it togethor
            D_loss = (D_R_loss + D_G_loss) / 2

        opt_disc.zero_grad()
        d_scaler.scale(D_loss).backward()
        d_scaler.step(opt_disc)
        d_scaler.update()

        # Train Generators R and G
        with torch.cuda.amp.autocast():
            # adversarial loss for both generators
            D_R_fake = disc_R(fake_real)
            D_G_fake = disc_G(fake_ghibli)
            loss_G_R = mse(D_R_fake, torch.ones_like(D_R_fake))
            loss_G_G = mse(D_G_fake, torch.ones_like(D_G_fake))

            # cycle loss
            cycle_ghibli = gen_G(fake_real)
            cycle_real = gen_R(fake_ghibli)
            cycle_ghibli_loss = l1(ghibli, cycle_ghibli)
            cycle_real_loss = l1(real, cycle_real)

            # add all togethor
            G_loss = (
                loss_G_G
                + loss_G_R
                + cycle_ghibli_loss * config.LAMBDA_CYCLE
                + cycle_real_loss * config.LAMBDA_CYCLE
            )

        opt_gen.zero_grad()
        g_scaler.scale(G_loss).backward()
        g_scaler.step(opt_gen)
        g_scaler.update()

        if idx % 100 == 0:
            save_image(fake_real * 0.5 + 0.5, f"saved_images/real_{idx}.png")
            save_image(fake_ghibli * 0.5 + 0.5, f"saved_images/ghibli_{idx}.png")



def main():
    disc_R = Discriminator(in_channels=3).to(config.DEVICE)
    disc_G = Discriminator(in_channels=3).to(config.DEVICE)
    gen_G = Generator(img_channels=3, num_residuals=9).to(config.DEVICE)
    gen_R = Generator(img_channels=3, num_residuals=9).to(config.DEVICE)
    opt_disc = optim.Adam(
        list(disc_R.parameters()) + list(disc_G.parameters()),
        lr=config.LEARNING_RATE,
        betas=(0.5, 0.999),
    )

    opt_gen = optim.Adam(
        list(gen_G.parameters()) + list(gen_R.parameters()),
        lr=config.LEARNING_RATE,
        betas=(0.5, 0.999),
    )

    L1 = nn.L1Loss()
    mse = nn.MSELoss()

    if config.LOAD_MODEL:
        load_checkpoint(
            config.CHECKPOINT_GEN_R, gen_R, opt_gen, config.LEARNING_RATE,
        )
        load_checkpoint(
            config.CHECKPOINT_GEN_G, gen_G, opt_gen, config.LEARNING_RATE,
        )
        load_checkpoint(
            config.CHECKPOINT_CRITIC_R, disc_R, opt_disc, config.LEARNING_RATE,
        )
        load_checkpoint(
            config.CHECKPOINT_CRITIC_G, disc_G, opt_disc, config.LEARNING_RATE,
        )

    dataset = RealGhibliDataset(
        root_dir=config.TRAIN_DIR,
        transform=config.transforms,
    )

    loader = DataLoader(
        dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
    )
    g_scaler = torch.cuda.amp.GradScaler()
    d_scaler = torch.cuda.amp.GradScaler()

    for epoch in range(config.NUM_EPOCHS):
        train_fn(
            disc_R,
            disc_G,
            gen_G,
            gen_R,
            loader,
            opt_disc,
            opt_gen,
            L1,
            mse,
            d_scaler,
            g_scaler,
        )

        if config.SAVE_MODEL:
            save_checkpoint(gen_R, opt_gen, filename=config.CHECKPOINT_GEN_R)
            save_checkpoint(gen_G, opt_gen, filename=config.CHECKPOINT_GEN_G)
            save_checkpoint(disc_R, opt_disc, filename=config.CHECKPOINT_CRITIC_R)
            save_checkpoint(disc_G, opt_disc, filename=config.CHECKPOINT_CRITIC_G)


if __name__ == "__main__":
    main()