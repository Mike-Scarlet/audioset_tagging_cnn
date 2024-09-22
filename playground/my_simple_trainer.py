
from local_dataset import *
from split_model import *
from base_config import *
import torch.optim as optim
import torch.utils.data
import time
from pytorch.losses import *

def __CheckDataset():
  sampler = TrainSampler(4)
  dataset = LocalH5Dataset()
  for sample in sampler:
    data_list = []
    for item in sample:
      data_list.append(dataset[item])
    coll = collate_fn(data_list)

def move_data_to_device(x, device):
    if 'float' in str(x.dtype):
        x = torch.Tensor(x)
    elif 'int' in str(x.dtype):
        x = torch.LongTensor(x)
    else:
        return x

    return x.to(device)

class Transfer_Cnn14(nn.Module):
    def __init__(self, sample_rate, window_size, hop_size, mel_bins, fmin, 
        fmax, classes_num, freeze_base):
        """Classifier for a new task using pretrained Cnn14 as a sub module.
        """
        super(Transfer_Cnn14, self).__init__()
        audioset_classes_num = 527
        
        self.base = Cnn14LogMel(sample_rate, window_size, hop_size, mel_bins, fmin, 
            fmax, audioset_classes_num)

        # Transfer to another task layer
        self.fc_transfer = nn.Linear(2048, classes_num, bias=True)

        if freeze_base:
            # Freeze AudioSet pretrained layers
            for param in self.base.parameters():
                param.requires_grad = False

        self.init_weights()

    def init_weights(self):
        init_layer(self.fc_transfer)

    def load_from_pretrain(self, pretrained_checkpoint_path):
        checkpoint = torch.load(pretrained_checkpoint_path)

        model_dict = checkpoint['model']

        remove_names = []
        for k in model_dict:
          if k.startswith("spectrogram") or k.startswith("logmel"):
            remove_names.append(k)
        for k in remove_names:
          del model_dict[k]

        self.base.load_state_dict(model_dict)

    def forward(self, input, mixup_lambda=None):
        """Input: (batch_size, data_length)
        """
        output_dict = self.base(input, mixup_lambda)
        embedding = output_dict['embedding']

        # print(f"embedding: {embedding}")

        print("fc sum", list(self.fc_transfer.parameters())[0].sum().item())
        fc_transter_out = self.fc_transfer(embedding)

        # clipwise_output =  torch.log_softmax(fc_transter_out, dim=-1)
        clipwise_output =  torch.sigmoid(fc_transter_out)
        output_dict['clipwise_output'] = clipwise_output
        # print(f"clipwise_output: {clipwise_output}")

        return output_dict

def main():
  train_bgn_time = time.time()
  
  sampler = TrainSampler(16)
  dataset = LocalH5Dataset()
  device = "cuda"
  
  # prepare model
  model = Transfer_Cnn14(sample_rate=sample_rate, window_size=window_size, 
        hop_size=hop_size, mel_bins=mel_bins, fmin=fmin, fmax=fmax, 
        classes_num=1, freeze_base=True)
  
  # load model state dict
  model.load_from_pretrain(r"F:\Audio\audioset_tagging_cnn\Cnn14_mAP=0.431.pth")
  
  # params_num = count_parameters(model)
  # # flops_num = count_flops(model, clip_samples)
  # logging.info('Parameters num: {}'.format(params_num))
  
  train_loader = torch.utils.data.DataLoader(dataset=dataset, 
        batch_sampler=sampler, collate_fn=collate_fn, 
        num_workers=0, pin_memory=True)
  
  # Optimizer
  optimizer = optim.Adam(model.parameters(), lr=1e-3, 
      betas=(0.9, 0.999), eps=1e-08, weight_decay=0., amsgrad=True)
  
  # Parallel
  print('GPU number: {}'.format(torch.cuda.device_count()))
  model = torch.nn.DataParallel(model)

  if 'cuda' in str(device):
    model.to(device)

  time1 = time.time()

  print("prepare before train: {}".format(time1 - train_bgn_time))

  # for name, param in model.named_parameters():
  #   print(f"{name}: requires_grad={param.requires_grad}")

  # train iter
  iter = 0
  for data in train_loader:
    for key in data.keys():
      data[key] = move_data_to_device(data[key], device)

    # transpose mel feature
    data['mel_feature'] = data['mel_feature'].transpose(2, 3)

    # forward
    model.train()

    batch_output_dict = model(data['mel_feature'], None)
    batch_target_dict = {'target': data['target']}

    # print(f"Model output: {batch_output_dict['clipwise_output']}")
    # print(batch_output_dict["clipwise_output"])
    # print(data['target'])

    loss = clip_bce(batch_output_dict, batch_target_dict)
    loss.backward()

    print(f"Loss: {loss.item()}")
    # print(loss)

    # for name, param in model.named_parameters():
    #   if not param.requires_grad:
    #     continue
    #   if param.grad is not None:
    #     print(f"Gradient of {name}: {param.grad}, sum: {param.grad.sum()}")

    if iter % 50 == 0:
      print("batch:", batch_output_dict["clipwise_output"].T.cpu().detach().numpy())
      print("data:", data['target'].T.cpu().detach().numpy())

    optimizer.step()
    optimizer.zero_grad()
    iter += 1
    # require_grad_items = [p for p in model.parameters() if p.requires_grad]
    # optimized_param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # print("opt cnt: {}".format(optimized_param_count))

if __name__ == "__main__":
  main()