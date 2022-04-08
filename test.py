import config
from core import dataloading, models, utils, tracking


# --- read options --- #
opt = config.read_arguments(train=False)

# --- create dataloader and recommended model config --- #
dataloader, model_config = dataloading.prepare_dataloading(opt)

# --- create models, losses, and optimizers ---#
netG, netD, netEMA = models.create_models(opt, model_config)

# --- create utils --- #
visualizer = tracking.visualizer(opt)

# --- generate images and masks --- #
for i in range(opt.num_generated):
    z = utils.sample_noise(opt.noise_dim, 1).to(opt.device)
    fake = netEMA.generate(z) if not opt.no_EMA else netG.generate(z)
    visualizer.save_batch(fake, opt.continue_epoch, i=str(i))