import json

import lib
import numpy as np
import torch
from tqdm import tqdm


class Prediction(object):
    def __init__(self, model, use_cuda, k=100):
        self.model = model
        self.topk = k
        self.device = torch.device('cuda' if use_cuda else 'cpu')

    def pred(self, eval_data, batch_size):
        self.model.eval()
        outputs = []
        dataloader = lib.DataLoader(eval_data, batch_size, if_predict=True)
        item_map_index = dataloader.dataset.itemmap.set_index(['item_idx']).to_dict()['ItemID']
        user_map_index = dataloader.dataset.usermap.set_index(['user_idx']).to_dict()['SessionID']
        with torch.no_grad():
            hidden = self.model.init_hidden()
            for ii, (input, mask, user_id) in tqdm(enumerate(dataloader),
                                                   total=len(dataloader.dataset.df) // dataloader.batch_size,
                                                   miniters=10):
                # for input, target, mask in dataloader:
                input = input.to(self.device)
                logit, hidden = self.model(input, hidden)
                score, output = lib.infer_output(logit, k=self.topk)
                output = output.cpu().numpy().tolist()
                score = score.cpu().numpy().tolist()
                for idx in mask:
                    for i in range(len(output[idx])):
                        output[idx][i] = item_map_index[output[idx][i]]
                    output_with_score_temp = []
                    for i in zip(output[idx], score[idx]):
                        output_with_score_temp.append(list(i))
                    outputs.append(json.dumps([user_map_index[user_id[idx]], output_with_score_temp]))

        return outputs
