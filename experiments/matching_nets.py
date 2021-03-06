"""
Reproduce Matching Network results of Vinyals et al
"""
import argparse
from torch.utils.data import DataLoader
from torch.optim import Adam

from few_shot.datasets import OmniglotDataset, MiniImageNet
from few_shot.core import NShotTaskSampler, prepare_nshot_task, EvaluateFewShot
from few_shot.matching import matching_net_episode
from few_shot.train import fit
from few_shot.callbacks import *
from few_shot.utils import setup_dirs
from config import PATH
import globals


setup_dirs()
assert torch.cuda.is_available()
device = torch.device('cuda')
torch.backends.cudnn.benchmark = True


##############
# Parameters #
##############

evaluation_episodes = 1000
episodes_per_epoch = 100

if globals.DATASET == 'omniglot':
    n_epochs = 100
    dataset_class = OmniglotDataset
    num_input_channels = 1
    lstm_input_size = 64
elif globals.DATASET == 'miniImageNet':
    n_epochs = 200
    dataset_class = MiniImageNet
    num_input_channels = 3
    lstm_input_size = 1600
else:
    raise(ValueError, 'Unsupported dataset')

param_str = f'{globals.DATASET}_n={globals.N_TRAIN}_k={globals.K_TRAIN}_q={globals.Q_TRAIN}_' \
            f'nv={globals.N_TEST}_kv={globals.K_TEST}_qv={globals.Q_TEST}_'\
            f'dist={globals.DISTANCE}_fce={globals.FCE}'


#########
# Model #
#########
from few_shot.models import MatchingNetwork
model = MatchingNetwork(globals.N_TRAIN, globals.K_TRAIN, globals.Q_TRAIN, globals.FCE, num_input_channels,
                        lstm_layers=globals.LSTM_LAYERS,
                        lstm_input_size=lstm_input_size,
                        unrolling_steps=globals.UNROLLING_STEPS,
                        device=device)
model.to(device, dtype=torch.double)


###################
# Create datasets #
###################
background = dataset_class('background')
background_taskloader = DataLoader(
    background,
    batch_sampler=NShotTaskSampler(background, episodes_per_epoch, globals.N_TRAIN, globals.K_TRAIN, globals.Q_TRAIN),
    num_workers=4
)
evaluation = dataset_class('evaluation')
evaluation_taskloader = DataLoader(
    evaluation,
    batch_sampler=NShotTaskSampler(evaluation, episodes_per_epoch, globals.N_TEST, globals.K_TEST, globals.Q_TEST),
    num_workers=4
)


############
# Training #
############
print(f'Training Matching Network on {globals.DATASET}...')
optimiser = Adam(model.parameters(), lr=1e-3)
loss_fn = torch.nn.NLLLoss().cuda()


callbacks = [
    EvaluateFewShot(
        eval_fn=matching_net_episode,
        num_tasks=evaluation_episodes,
        n_shot=globals.N_TEST,
        k_way=globals.K_TEST,
        q_queries=globals.Q_TEST,
        taskloader=evaluation_taskloader,
        prepare_batch=prepare_nshot_task(globals.N_TEST, globals.K_TEST, globals.Q_TEST),
        fce=globals.FCE,
        distance=globals.DISTANCE
    ),
    ModelCheckpoint(
        filepath=PATH + f'/models/matching_nets/{param_str}.pth',
        monitor=f'val_{globals.N_TEST}-shot_{globals.K_TEST}-way_acc',
        # monitor=f'val_loss',
    ),
    ReduceLROnPlateau(patience=20, factor=0.5, monitor=f'val_{globals.N_TEST}-shot_{globals.K_TEST}-way_acc'),
    CSVLogger(PATH + f'/logs/matching_nets/{param_str}.csv'),
]

fit(
    model,
    optimiser,
    loss_fn,
    epochs=n_epochs,
    dataloader=background_taskloader,
    prepare_batch=prepare_nshot_task(globals.N_TRAIN, globals.K_TRAIN, globals.Q_TRAIN),
    callbacks=callbacks,
    metrics=['categorical_accuracy'],
    fit_function=matching_net_episode,
    fit_function_kwargs={'n_shot': globals.N_TRAIN, 'k_way': globals.K_TRAIN, 'q_queries': globals.Q_TRAIN, 'train': True,
                         'fce': globals.FCE, 'distance': globals.DISTANCE}
)
