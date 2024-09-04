import torch
import transformers

class BERT(torch.nn.Module):
    def __init__(self):
        super(BERT, self).__init__()
        self.l1 = transformers.BertModel.from_pretrained('bert-base-chinese')
        self.l2 = torch.nn.Dropout(0.3)
        self.l3 = torch.nn.Linear(64, 1)
    
    def forward(self, q1, q2):
        _, output1 = self.l1(inputs_embeds=q1, return_dict=False)
        _, output2 = self.l1(inputs_embeds=q2, return_dict=False)

        output1 = self.l2(output1)
        output2 = self.l2(output2)

        v = torch.cat((output1, output2), dim=1)
        return self.l3(v).squeeze(1)