from backend.response.direct import DirectHandler
from transformers.modeling_gpt2 import GPT2Config, GPT2LMHeadModel
from transformers import BertTokenizer
from typing import List
from backend.utils import ConfigParser
import copy
import torch
import torch.nn.functional as F


class ConversationHandler(DirectHandler):
    def __init__(self, device='cuda', history_len=0, batch_size=5, max_len=25,
                 penalty=1.0, temperature=1):
        super(ConversationHandler, self).__init__()
        self.device = device
        self.max_len = max_len  # max length of each utterance
        self.history_len = history_len
        self.config = ConfigParser.get_dict()['conversation']
        self.tokenizer = BertTokenizer(vocab_file=self.config['voca_path'])
        self.model = GPT2LMHeadModel.from_pretrained(self.config['dialogue_model'])
        self.mmi_model = GPT2LMHeadModel.from_pretrained(self.config['mmi_model'])
        # move both models to specific device
        self.model.to(device)
        self.model.eval()
        self.mmi_model.to(device)
        self.mmi_model.eval()
        self.history = []  # for future multi-conversation usage
        self.batch_size = batch_size  # how many response generated for MMI filter
        self.penalty = penalty
        self.temperature = temperature

    @staticmethod
    def top_k_top_p_filtering(logits, top_k=8, top_p=0.0, filter_value=-float('Inf')):
        """ Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
            Args:
                logits: logits distribution shape (vocabulary size)
                top_k > 0: keep only top k tokens with highest probability (top-k filtering).
                top_p > 0.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
                    Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
        """
        assert logits.dim() == 2
        top_k = min(top_k, logits[0].size(-1))  # Safety check
        if top_k > 0:
            # Remove all tokens with a probability less than the last token of the top-k
            for logit in logits:
                indices_to_remove = logit < torch.topk(logit, top_k)[0][..., -1, None]
                logit[indices_to_remove] = filter_value  # 对于topk之外的其他元素的logits值设为负无穷

        if top_p > 0.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)  # 对logits进行递减排序
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

            # Remove tokens with cumulative probability above the threshold
            sorted_indices_to_remove = cumulative_probs > top_p
            # Shift the indices to the right to keep also the first token above the threshold
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            for index, logit in enumerate(logits):
                indices_to_remove = sorted_indices[index][sorted_indices_to_remove[index]]
                logit[indices_to_remove] = filter_value
        return logits

    def get_matched(self, target: List[str], log=None) -> (str, List[str]):
        log_list = [] if not isinstance(log, list) else log

        input_text = ''.join(target)  # simply concat each string into sentence
        log_list.append('对话输入为: {}'.format(input_text))
        self.history.append(self.tokenizer.encode(input_text))
        input_ids = [self.tokenizer.cls_token_id]
        # add history, disable it if history_len = 1
        for history_id, history_utr in enumerate(self.history[-self.history_len:]):
            input_ids.extend(history_utr)
            input_ids.append(self.tokenizer.sep_token_id)

        input_ids = [copy.deepcopy(input_ids) for _ in range(self.batch_size)]

        input_tensors = torch.tensor(input_ids).long().to(self.device)
        generated = []
        finish_set = set()

        for _ in range(self.max_len):   # limit length of each utterance
            outputs = self.model(input_ids=input_tensors)
            next_token_logits = outputs[0][:, -1, :]  # grab the last array at dim 1
            for idx in range(self.batch_size):
                for token_id in set([token_ids[idx] for token_ids in generated]):
                    next_token_logits[idx][token_id] /= self.penalty
            next_token_logits = next_token_logits / self.temperature

            # set [UNK] prob to -Inf to make sure no [UNK] preds are made
            for next_token_logit in next_token_logits:
                next_token_logit[self.tokenizer.convert_tokens_to_ids('[UNK]')] = -float('Inf')
            filtered_logits = self.top_k_top_p_filtering(next_token_logits)

            next_token = torch.multinomial(F.softmax(filtered_logits, dim=-1), num_samples=1)
            # judge if any response already generated [SEP]
            for idx, token_id in enumerate(next_token[:, 0]):
                if token_id == self.tokenizer.sep_token_id:
                    finish_set.add(idx)  # record its index
            finish_flag = all(idx in finish_set for idx in range(self.batch_size))
            if finish_flag:
                break
            generated.append([token.item() for token in next_token[:, 0]])
            input_tensors = torch.cat((input_tensors, next_token), dim=-1)

        candidate_response = []
        for batch_idx in range(self.batch_size):  # parse candidate response
            response = []
            for token_idx in range(len(generated)):
                if generated[token_idx][batch_idx] != self.tokenizer.sep_token_id:
                    response.append(generated[token_idx][batch_idx])
                else:
                    break
            candidate_response.append(response)

        log_list.append("候选回复:")
        min_loss = float('Inf')
        best_response = ""
        for res in candidate_response:
            mmi_input_id = [self.tokenizer.cls_token_id]
            mmi_input_id.extend(res)
            mmi_input_id.append(self.tokenizer.sep_token_id)
            for history_utr in reversed(self.history[-self.history_len: ]):
                mmi_input_id.extend(history_utr)
                mmi_input_id.append(self.tokenizer.sep_token_id)

            mmi_input_tensor = torch.tensor(mmi_input_id).long().to(self.device)
            out = self.mmi_model(input_ids=mmi_input_tensor, labels=mmi_input_tensor)
            loss = out[0].item()
            # for debug log
            response_text = self.tokenizer.convert_ids_to_tokens(res)
            log_list.append("回复 “{}” Loss为 {:.2f}".format(response_text, loss))
            if loss < min_loss:
                best_response = res
                min_loss = loss
        self.history.append(best_response)
        best_response_text = self.tokenizer.convert_ids_to_tokens(best_response)

        # send to DirectHandler to search for memes
        # TODO: May need further tokenizer via HanLP
        return super().get_matched([best_response_text], log_list)
        