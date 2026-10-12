import torch
import torch.nn as nn


class SEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.Tanh(),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()  # SE的输出是权重，所以必须经过sigmoid才能得到0-1之间的值
        )

    def forward(self, x):
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y.expand_as(x)


class base_Model(nn.Module):
    def __init__(self, configs):
        super(base_Model, self).__init__()

        activation_func = nn.Tanh()

        # 第一个卷积块
        self.conv_block1 = nn.Sequential(
            nn.Conv1d(configs.input_channels, 64, kernel_size=11,
                      stride=configs.stride, padding='same', bias=False),
            nn.BatchNorm1d(64),
            activation_func,
            nn.Dropout(configs.dropout),

            nn.Conv1d(64, 64, kernel_size=7,
                      stride=1, padding='same', bias=False),
            nn.BatchNorm1d(64),
            activation_func
        )

        # 第二个卷积块
        self.conv_block2_res = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=7, stride=1, padding='same', bias=False),
            nn.BatchNorm1d(128),
            activation_func,

            nn.Conv1d(128, 128, kernel_size=5, stride=1, padding='same', bias=False),
            nn.BatchNorm1d(128),
        )
        self.attention_block2 = SEBlock(128)
        self.res_conv2 = nn.Conv1d(64, 128, kernel_size=1, stride=1, bias=False)

        # 第三个卷积块
        self.conv_block3_res = nn.Sequential(
            nn.Conv1d(128, configs.final_out_channels, kernel_size=5, stride=1, padding='same', bias=False),
            nn.BatchNorm1d(configs.final_out_channels),
            activation_func,

            nn.Conv1d(configs.final_out_channels, configs.final_out_channels, kernel_size=3, stride=1, padding='same', bias=False),
            nn.BatchNorm1d(configs.final_out_channels),
        )
        self.attention_block3 = SEBlock(configs.final_out_channels)
        self.res_conv3 = nn.Conv1d(128, configs.final_out_channels, kernel_size=1, stride=1, bias=False)

        model_output_dim = configs.features_len

        self.logits = nn.Sequential(
            nn.Linear(model_output_dim * configs.final_out_channels, 512),
            activation_func,
            nn.Dropout(configs.dropout),
            nn.Linear(512, configs.num_classes)
        )
        self.activation_func = activation_func

        # v2: 掩码重建投影头（TimeMAE 式掩码表征回归）
        # 学生（掩码输入）的卷积特征 -> 对齐 EMA 教师的无掩码特征
        self.recon_proj = nn.Sequential(
            nn.Conv1d(configs.final_out_channels, configs.final_out_channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(configs.final_out_channels),
        )

    def forward(self, x_in):
        B, T, C = x_in.shape
        x = x_in.permute(0, 2, 1)

        # 第一阶段
        x1 = self.conv_block1(x)
        # print(x1.grad_fn)

        # 第二阶段: 残差 + 注意力
        x2_res = self.conv_block2_res(x1)
        x2_att = self.attention_block2(x2_res)
        x2 = x2_att + self.res_conv2(x1)
        x2 = self.activation_func(x2)
        # print(x2.grad_fn)

        # 第三阶段: 残差 + 注意力
        x3_res = self.conv_block3_res(x2)
        x3_att = self.attention_block3(x3_res)
        x3 = x3_att + self.res_conv3(x2)
        x3 = self.activation_func(x3)
        # print(x3.grad_fn)

        # 展平并分类
        x_flat = x3.reshape(x3.shape[0], -1)
        logits = self.logits(x_flat)
        # print(logits.grad_fn)

        return logits, x3

