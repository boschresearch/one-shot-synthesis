import config
from core import dataloading, models, utils, losses, tracking
from core.differentiable_augmentation import diff_augm
import torch


# --- read options --- #
opt = config.read_arguments(train=True)

# --- create dataloader and recommended model config --- #
dataloader, model_config = dataloading.prepare_dataloading(opt)

# --- create models, losses, and optimizers ---#
netG, netD, netEMA = models.create_models(opt, model_config)
losses_computer = losses.losses_computer(opt, netD.num_blocks)
optimizerG, optimizerD = models.create_optimizers(netG, netD, opt)

# --- create utils --- #
utils.fix_seed(opt.seed)
timer = utils.timer(opt)
visualizer = tracking.visualizer(opt)
diff_augment = diff_augm.augment_pipe(opt)

# --- training loop --- #
for epoch, batch in enumerate(dataloader, start=opt.continue_epoch):
    batch = utils.preprocess_real(batch, netD.num_blocks_ll, opt.device)
    logits, losses = dict(), dict()

    # --- generator update --- #
    netG.zero_grad()
    z = utils.sample_noise(opt.noise_dim, opt.batch_size).to(opt.device)
    out_G = netG.generate(z, get_feat=not opt.no_DR)
    out_G = diff_augment(out_G)
    logits["G"] = netD.discriminate(out_G, for_real=False, epoch=epoch)
    losses["G"] = losses_computer(logits["G"], out_G, real=True, forD=False)
    loss = sum(losses["G"].values())
    loss.backward(retain_graph=True)
    optimizerG.step()

    # --- discriminator update --- #
    netD.zero_grad()
    batch = diff_augment(batch)
    logits["Dreal"] = netD.discriminate(batch, for_real=True, epoch=epoch)
    losses["Dreal"] = losses_computer(logits["Dreal"], batch, real=True, forD=True)
    loss = sum(losses["Dreal"].values())
    loss.backward(retain_graph=True)

    z = utils.sample_noise(opt.noise_dim, opt.batch_size).to(opt.device)
    with torch.no_grad():
        out_G = netG.generate(z)  # fake
    out_G = diff_augment(out_G)
    logits["Dfake"] = netD.discriminate(out_G, for_real=False, epoch=epoch)
    losses["Dfake"] = losses_computer(logits["Dfake"], out_G, real=False, forD=True)
    loss = sum(losses["Dfake"].values())
    loss.backward(retain_graph=True)
    optimizerD.step()

    # --- stats tracking --- #
    visualizer.track_losses_logits(logits, losses)
    if not opt.no_EMA:
        netEMA = utils.update_EMA(netEMA, netG, opt.EMA_decay)
    if epoch % opt.freq_save_ckpt == 0 or epoch == opt.num_epochs:
        visualizer.save_networks(netG, netD, netEMA, epoch)
    if epoch % opt.freq_print == 0 or epoch == opt.num_epochs:
        timer(epoch)
        z = utils.sample_noise(opt.noise_dim, 8).to(opt.device)
        fake = netEMA.generate(z) if not opt.no_EMA else netG.generate(z)
        visualizer.save_batch(fake, epoch)
    if (epoch % opt.freq_save_loss == 0 or epoch == opt.num_epochs) and epoch > 0 :
        visualizer.save_losses_logits(epoch)

    # --- exit if reached the end --- #
    if epoch >= opt.num_epochs:
        break

# --- after training ---#
print("Succesfully finished")
