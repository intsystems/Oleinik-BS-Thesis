import torch
import torch.nn as nn
import torch.nn.functional as F
from math import ceil, prod


def calc_new_width(width, kernel_size=1, stride=1, padding=0):
    return (width - kernel_size + 2 * padding) // stride + 1, (
        width - kernel_size + 2 * padding
    ) % stride


def build_3_conv(begin, end):
    if begin < end:
        raise ValueError("Begin less then end")

    step = ceil((begin / end) ** (1 / 3))
    current = begin

    conv_setup = []
    padding = []
    for i in range(3):
        output_padding = 0
        if current // step >= end:
            conv_setup.append({"kernel_size": step, "stride": step})
            current, output_padding = calc_new_width(current, step, step)
        elif current == end:
            conv_setup.append({"kernel_size": 1})
        else:
            diff = current - end
            conv_setup.append({"kernel_size": diff + 1})
            current = end
        padding.append(output_padding)

    return conv_setup, padding


class Mu_Transform(nn.Module):
    def __init__(
        self,
        in_channel,
        out_channel,
        conv_setup={},
        verbose=False,
        transposed=False,
        linear=False,
    ):
        super(Mu_Transform, self).__init__()
        self.verbose = verbose
        self.linear = linear

        if self.linear:
            self.linear_layer = nn.Linear(in_channel, out_channel)
        else:
            if transposed:
                self.conv1 = nn.ConvTranspose2d(
                    in_channel, 2 * in_channel, **conv_setup[0]
                )
                self.conv1_bn = nn.BatchNorm2d(2 * in_channel)

                self.conv2 = nn.ConvTranspose2d(
                    2 * in_channel, 2 * out_channel, **conv_setup[1]
                )
                self.conv2_bn = nn.BatchNorm2d(2 * out_channel)

                self.conv3 = nn.ConvTranspose2d(
                    2 * out_channel, out_channel, **conv_setup[2]
                )
            else:
                self.conv1 = nn.Conv2d(in_channel, 2 * in_channel, **conv_setup[0])
                self.conv1_bn = nn.BatchNorm2d(2 * in_channel)

                self.conv2 = nn.Conv2d(2 * in_channel, 2 * out_channel, **conv_setup[1])
                self.conv2_bn = nn.BatchNorm2d(2 * out_channel)

                self.conv3 = nn.Conv2d(2 * out_channel, out_channel, **conv_setup[2])

    def forward(self, x):
        if self.verbose:
            print(f"{x.shape=}")

        if self.linear:
            out = x.view(x.size(0), -1)
            out = self.linear_layer(out)
        else:
            out = F.relu(self.conv1_bn(self.conv1(x)))
            if self.verbose:
                print(f"{out.shape=}")

            out = F.relu(self.conv2_bn(self.conv2(out)))
            if self.verbose:
                print(f"{out.shape=}")

            out = self.conv3(out)

        if self.verbose:
            print(f"{out.shape=}")

        return out


def get_mu_transform(first: torch.Size, second: torch.Size, verbose=False):
    if len(first) == 2 or len(second) == 2:
        return Mu_Transform(
            prod(first[1:]), prod(second[1:]), linear=True, verbose=verbose
        )
    elif first[2] < second[2]:
        conv_setup, output_padding = build_3_conv(second[2], first[2])
        for i in range(3):
            conv_setup[i]["output_padding"] = output_padding[i]
        conv_setup.reverse()
        if verbose:
            print(f"{conv_setup=}")
        return Mu_Transform(
            first[1], second[1], conv_setup=conv_setup, transposed=True, verbose=verbose
        )
    else:
        conv_setup, output_padding = build_3_conv(first[2], second[2])
        if verbose:
            print(f"{conv_setup=}")
        return Mu_Transform(first[1], second[1], conv_setup=conv_setup, verbose=verbose)


class DownStudent(nn.Module):
    def __init__(self, from_=32, to_=3):
        super(DownStudent, self).__init__()
        self.from_ = from_
        self.to_ = to_

        self.conv1 = nn.Conv2d(32, 16, kernel_size=1, padding=2)
        self.conv1_bn = nn.BatchNorm2d(16)

        self.conv2 = nn.Conv2d(16, 8, kernel_size=2, padding=5)
        self.conv2_bn = nn.BatchNorm2d(8)

        self.conv3 = nn.Conv2d(8, 3, kernel_size=2, padding=9)
        self.conv3_bn = nn.BatchNorm2d(3)

    def forward(self, x):
        out = x
        if self.from_ >= 32:
            out = F.relu(self.conv1_bn(self.conv1(out)))

        if self.to_ == 16:
            return out

        if self.from_ >= 16:
            out = F.relu(self.conv2_bn(self.conv2(out)))

        if self.to_ == 8:
            return out

        out = self.conv2_bn(self.conv3(out))

        return out


class UpStudent(nn.Module):
    def __init__(self, from_=3, to_=64):
        super(UpStudent, self).__init__()
        self.from_ = from_
        self.to_ = to_

        self.conv1 = nn.Conv2d(3, 8, kernel_size=3)
        self.conv1_bn = nn.BatchNorm2d(8)

        self.conv2 = nn.Conv2d(8, 16, kernel_size=3)
        self.conv2_bn = nn.BatchNorm2d(16)

        self.conv3 = nn.Conv2d(16, 32, kernel_size=3)
        self.conv3_bn = nn.BatchNorm2d(32)

        self.fc1 = nn.Linear(in_features=128, out_features=64)

    def forward(self, x):
        out = x
        if self.from_ <= 3:
            out = F.relu(self.conv1_bn(self.conv1(out)))
            out = F.max_pool2d(out, 2)

        if self.to_ == 8:
            return out

        if self.from_ <= 8:
            out = F.relu(self.conv2_bn(self.conv2(out)))
            out = F.max_pool2d(out, 2)

        if self.to_ == 16:
            return out

        if self.from_ <= 16:
            out = self.conv3_bn(self.conv3(out))
            out = F.max_pool2d(out, 2)

        if self.to_ == 32:
            return out

        out = out.view(out.size(0), -1)
        out = self.fc1(out)

        return out


class Student_Teacher(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Student_Teacher, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, 2 * in_channels, kernel_size=1)
        self.conv1_bn = nn.BatchNorm2d(2 * in_channels)

        self.conv2 = nn.Conv2d(2 * in_channels, 2 * out_channels, kernel_size=1)
        self.conv2_bn = nn.BatchNorm2d(2 * out_channels)

        self.conv3 = nn.Conv2d(2 * out_channels, out_channels, kernel_size=1)

    def forward(self, x):
        out = F.relu(self.conv1_bn(self.conv1(x)))

        out = F.relu(self.conv2_bn(self.conv2(out)))

        out = self.conv3(out)

        return out


class Linear_Model(nn.Module):
    def __init__(self, in_size, out_size):
        super(Linear_Model, self).__init__()
        self.lin = nn.Linear(in_size, out_size)

    def forward(self, x):
        out = x.view(x.size(0), -1)
        out = self.lin(out)
        return out


class Mutual_Info(nn.Module):
    def __init__(self, sequence):
        super(Mutual_Info, self).__init__()
        self.sequence = nn.ModuleList(sequence)

    def forward(self, x):
        for module in self.sequence:
            x = module(x)
        return x
