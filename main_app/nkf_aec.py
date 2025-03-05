'''
Tencent is pleased to support the open source community by making NKF-AEC available.

Copyright (C) 2022 THL A29 Limited, a Tencent company. All rights reserved.

Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
in compliance with the License. You may obtain a copy of the License at

https://opensource.org/licenses/BSD-3-Clause

Unless required by applicable law or agreed to in writing, software distributed
under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
'''
import torch
import torch.nn as nn


class ComplexGRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, batch_first=True, bias=True, dropout=0,
                 bidirectional=False):
        super().__init__()
        self.gru_r = nn.GRU(input_size, hidden_size, num_layers, bias=bias, batch_first=batch_first,
                            dropout=dropout, bidirectional=bidirectional)
        self.gru_i = nn.GRU(input_size, hidden_size, num_layers, bias=bias, batch_first=batch_first,
                            dropout=dropout, bidirectional=bidirectional)

    def forward(self, x, h_rr=None, h_ir=None, h_ri=None, h_ii=None):
        Frr, h_rr = self.gru_r(x.real, h_rr)
        Fir, h_ir = self.gru_r(x.imag, h_ir)
        Fri, h_ri = self.gru_i(x.real, h_ri)
        Fii, h_ii = self.gru_i(x.imag, h_ii)
        y = torch.complex(Frr - Fii, Fri + Fir)
        return y, h_rr, h_ir, h_ri, h_ii


class ComplexDense(nn.Module):
    def __init__(self, in_channel, out_channel, bias=True):
        super().__init__()
        self.linear_real = nn.Linear(in_channel, out_channel, bias=bias)
        self.linear_imag = nn.Linear(in_channel, out_channel, bias=bias)

    def forward(self, x):
        y_real = self.linear_real(x.real)
        y_imag = self.linear_imag(x.imag)
        return torch.complex(y_real, y_imag)


class ComplexPReLU(nn.Module):
    def __init__(self):
        super().__init__()
        self.prelu = torch.nn.PReLU()

    def forward(self, x):
        return torch.complex(self.prelu(x.real), self.prelu(x.imag))


class KGNet(nn.Module):
    def __init__(self, L, fc_dim, rnn_layers, rnn_dim):
        super().__init__()
        self.L = L
        self.rnn_layers = rnn_layers
        self.rnn_dim = rnn_dim

        self.fc_in = nn.Sequential(
            ComplexDense(2 * self.L + 1, fc_dim, bias=True),
            ComplexPReLU()
        )

        self.complex_gru = ComplexGRU(fc_dim, rnn_dim, rnn_layers, bidirectional=False)

        self.fc_out = nn.Sequential(
            ComplexDense(rnn_dim, fc_dim, bias=True),
            ComplexPReLU(),
            ComplexDense(fc_dim, self.L, bias=True)
        )

    def init_hidden(self, batch_size, device):
        self.h_rr = torch.zeros(self.rnn_layers, batch_size, self.rnn_dim).to(device=device)
        self.h_ir = torch.zeros(self.rnn_layers, batch_size, self.rnn_dim).to(device=device)
        self.h_ri = torch.zeros(self.rnn_layers, batch_size, self.rnn_dim).to(device=device)
        self.h_ii = torch.zeros(self.rnn_layers, batch_size, self.rnn_dim).to(device=device)

    def forward(self, input_feature):
        feat = self.fc_in(input_feature).unsqueeze(1)
        rnn_out, self.h_rr, self.h_ir, self.h_ri, self.h_ii = self.complex_gru(feat, self.h_rr, self.h_ir, self.h_ri, self.h_ii)
        kg = self.fc_out(rnn_out).permute(0, 2, 1)
        return kg


class NKF(nn.Module):
    def __init__(self, L=4):
        super().__init__()
        self.L = L
        self.kg_net = KGNet(L=self.L, fc_dim=18, rnn_layers=1, rnn_dim=18)
        self.stft = lambda x: torch.stft(x, n_fft=1024, hop_length=256, win_length=1024, window=torch.hann_window(1024),
                                         return_complex=True)
        self.istft = lambda X: torch.istft(X, n_fft=1024, hop_length=256, win_length=1024,
                                           window=torch.hann_window(1024), return_complex=False)

    def forward(self, x, y):
        if x.dim() == 1:
            x = x.unsqueeze(0)
        if y.dim() == 1:
            y = y.unsqueeze(0)
        x = self.stft(x)
        y = self.stft(y)
        B, F, T = x.shape
        device = x.device
        h_prior = torch.zeros(B * F, self.L, 1, dtype=torch.complex64, device=device)
        h_posterior = torch.zeros(B * F, self.L, 1, dtype=torch.complex64, device=device)
        self.kg_net.init_hidden(B * F, device)

        x = x.contiguous().view(B * F, T)
        y = y.contiguous().view(B * F, T)
        echo_hat = torch.zeros(B * F, T, dtype=torch.complex64, device=device)

        for t in range(T):
            if t < self.L:
                xt = torch.cat([torch.zeros(B * F, self.L - t - 1, dtype=torch.complex64, device=device), x[:, :t + 1]],
                               dim=-1)
            else:
                xt = x[:, t - self.L + 1:t + 1]
            if xt.abs().mean() < 1e-5:
                continue

            dh = h_posterior - h_prior
            h_prior = h_posterior
            e = y[:, t] - torch.matmul(xt.unsqueeze(1), h_prior).squeeze()

            input_feature = torch.cat([xt, e.unsqueeze(1), dh.squeeze()], dim=1)
            kg = self.kg_net(input_feature)
            h_posterior = h_prior + torch.matmul(kg, e.unsqueeze(-1).unsqueeze(-1))

            echo_hat[:, t] = torch.matmul(xt.unsqueeze(1), h_posterior).squeeze()

        s_hat = self.istft(y - echo_hat).squeeze()

        return s_hat
