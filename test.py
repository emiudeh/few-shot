'''
references:

experiments/matching_nets.py


'''
import time 
import torch
import globals
from torch.utils.data import DataLoader
from few_shot.eval import evaluate
from few_shot.datasets import OmniglotDataset, MiniImageNet
from few_shot.core import NShotTaskSampler
from few_shot.matching import matching_net_episode
from few_shot.models import MatchingNetwork
from few_shot.core import NShotTaskSampler, prepare_nshot_task, EvaluateFewShot
from few_shot.eval import evaluate



if globals.DATASET == 'omniglot':
    dataset_class = OmniglotDataset
    num_input_channels = 1
    lstm_input_size = 64
elif globals.DATASET == 'miniImageNet':
    dataset_class = MiniImageNet
    num_input_channels = 3
    lstm_input_size = 1600
else:
    raise(ValueError, 'Unsupported dataset')

assert torch.cuda.is_available()
device = torch.device('cuda')
torch.backends.cudnn.benchmark = True
evaluation_episodes = 1000
episodes_per_epoch = 100


#####
# experiments/matching_nets.py
model = MatchingNetwork(globals.N_TRAIN, globals.K_TRAIN, globals.Q_TRAIN, globals.FCE, 
                        num_input_channels,
                        lstm_layers=globals.LSTM_LAYERS,
                        lstm_input_size=lstm_input_size,
                        unrolling_steps=globals.UNROLLING_STEPS,
                        device=device)

model_path = 'models/matching_nets/omniglot_n=1_k=5_q=15_nv=1_kv=5_qv=1_dist=l2_fce=False.pth'
loaded_model = torch.load(model_path)
model.load_state_dict(loaded_model)


model.to(device)
model.double()
# print("###########################")
# for param in model.parameters():
#     print(param.data)
# print("###########################")


evaluation = dataset_class('evaluation')
dataloader = DataLoader(
    evaluation,
    batch_sampler=NShotTaskSampler(evaluation, episodes_per_epoch, n=globals.N_TEST, k=globals.K_TEST, 
                        q=globals.Q_TEST),
                        num_workers=4
)
prepare_batch = prepare_nshot_task(globals.N_TEST, globals.K_TEST, globals.Q_TEST)

for batch_index, batch in enumerate(dataloader):
    batch_logs = dict(batch=batch_index, size=(dataloader.batch_size or 1))
    x, y = prepare_batch(batch)
    # print(type(x))
    # time.sleep(55)
    loss, y_pred = matching_net_episode(model, None, torch.nn.NLLLoss().cuda(), x, y, globals.N_TEST, globals.K_TEST, 
                    globals.Q_TEST, 'l2', False, False )

    _, predicted = torch.max(y_pred.data, 1)
    # print(predicted)
    # print(y_pred.argmax(dim=-1))
    # print(loss.item())
    print(torch.eq(y_pred.argmax(dim=-1), y).sum().item() / y_pred.shape[0])
    print(y_pred.argmax(dim=-1))
    print(y)
    # print(y_pred.shape)

    # time.sleep(55)

print("FINISHED")



