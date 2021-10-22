import torch 
import torch.nn as nn
import torch.optim as optim
import config
from dataset import MapDataset
from generator import Generator
from discriminator import Discriminator
from losses import GeneratorLoss
from torch.utils.data import DataLoader 
from tqdm import tqdm
from torchvision.utils import save_image
from torch.autograd import Variable

torch.autograd.set_detect_anomaly(True)

def train_fn(disc, gen, loader, generator_criterion, opt_disc, opt_gen, epoch):
    train_bar = tqdm(loader)
    running_results = {'batch_sizes': 0, 'd_loss': 0, 'g_loss': 0, 'd_score': 0, 'g_score': 0}        
    
    gen.train()
    disc.train()

    for data, target in train_bar:
        g_update_first = True
        batch_size = data.size(0)
        running_results['batch_sizes'] += batch_size

        real_img = Variable(target)
        if torch.cuda.is_available():
            real_img = real_img.cuda()
        z = Variable(data)
        if torch.cuda.is_available():
            z = z.cuda()

        # Update Discriminator Network
        fake_img = gen(z)

        disc.zero_grad()
        real_out = disc(real_img).mean()
        fake_out = disc(fake_img).mean()
        d_loss = 1 - real_out + fake_out
        d_loss.backward(retain_graph=True)
        opt_disc.step()

         # Update Generator network
        fake_img = gen(z)
        fake_out = disc(fake_img).mean()

        gen.zero_grad()
        g_loss = generator_criterion(fake_out, fake_img, real_img)
        g_loss.backward()

        fake_img = gen(z)
        fake_out = disc(fake_img).mean()

        opt_gen.step()

        running_results['g_loss'] += g_loss.item() * batch_size
        running_results['d_loss'] += d_loss.item() * batch_size
        running_results['d_score'] += real_out.item() * batch_size
        running_results['g_score'] += fake_out.item() * batch_size

        train_bar.set_description(desc='[%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f' % (
            epoch, config.NUM_EPOCHS, running_results['d_loss'] / running_results['batch_sizes'],
            running_results['g_loss'] / running_results['batch_sizes'],
            running_results['d_score'] / running_results['batch_sizes'],
            running_results['g_score'] / running_results['batch_sizes']))

    gen.eval()
    out_path = 'training_results/SRF_' + str(UPSCALE_FACTOR) + '/'
    if not os.path.exists(out_path):
        os.makedirs(out_path)


def main():
    disc = Discriminator().to(config.DEVICE)
    gen = Generator(config.UPSCALE_FACTOR).to(config.DEVICE)
    generator_criterion = GeneratorLoss().to(config.DEVICE)

    opt_disc = optim.Adam(disc.parameters(), lr=config.LEARNING_RATE)
    opt_gen = optim.Adam(gen.parameters(), lr=config.LEARNING_RATE)

    results = {'d_loss': [], 'g_loss': [], 'd_score': [], 'g_score': [], 'psnr': [], 'ssim': []}

    # if config.LOAD_MODEL:
    #     load_checkpoint(
    #         config.CHECKPOINT_GEN, gen, opt_gen, config.LEARNING_RATE,
    #     )
    #     load_checkpoint(
    #         config.CHECKPOINT_DISC, disc, opt_disc, config.LEARNING_RATE,
    #     )

    train_dataset = MapDataset(config.INPUT_DIR, config.TARGET_DIR)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
    )
    
    val_dataset = MapDataset(input_dir=config.INPUT_DIR_TEST, target_dir=config.TARGET_DIR_TEST)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    for epoch in range(config.NUM_EPOCHS):
        train_fn(disc, gen, train_loader, generator_criterion, opt_disc, opt_gen, epoch)

        # if config.SAVE_MODEL and epoch % 5 == 0:
        #     save_checkpoint(gen, opt_gen, filename=config.CHECKPOINT_GEN)
        #     save_checkpoint(disc, opt_disc, filename=config.CHECKPOINT_DISC)

        # save_some_examples(gen, val_loader, epoch, folder="../evaluation")


if __name__ == "__main__":
    main()