import torch
import torch.nn.functional as F
import torch.nn as nn

class MyAttention(nn.Module):
    def __init__(self,hidden_size,attention_size):
        super(MyAttention, self).__init__()
        self.hidden_size=hidden_size
        self.attention_size = attention_size

        self.W_omega  = nn.Parameter(torch.Tensor(self.attention_size,self.hidden_size))
        self.b_omega  = nn.Parameter(torch.Tensor(self.attention_size))
        self.u_omega  = nn.Parameter(torch.Tensor(1,self.attention_size))

    def forward(self, inputs, time_major=True, return_alphas=False):
        '''

        :param inputs: 这里的inputs 的shape必须是[batch,len,hididen]
        :param time_major:
        :param return_alphas:
        :return:
        '''
        # v = torch.tanh(torch.matmul(torch.reshape(inputs, [-1, hidden_size]), W_omega) + tf.reshape(b_omega, [1, -1]))

        v  = torch.tanh(F.linear(inputs, self.W_omega, self.b_omega)) #[B, L, attention_size]
        print(v.size())
        vu = F.linear(v, self.u_omega)  #[B,L,1]
        print(vu.size())
        vu = torch.squeeze(vu)
        print(vu.size())
        exps = torch.exp(vu)
        print(exps.size())

        alphas = exps /torch.sum(input=exps,dim=1, keepdim=True)
        print(alphas.size())

        # Output of Bi-RNN is reduced with attention vector
        output = torch.sum(inputs * torch.unsqueeze(alphas, 2), 1)
        return output
#
# myAttention = MyAttention(100,50)
#
# x = torch.randn(10,64,100)
# y=x.permute(1,0,2)
# res=myAttention(y)
# print(res.size())

x = torch.randn(2,2)
print(x)
y = torch.randn(2,2)
z = torch.randn(2,2)

a = [x,y,z]

res= torch.cat(a, 0)
res=torch.reshape(res, (-1, 2, 2))
print(res.size())
print(res[0])