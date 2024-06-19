import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
import matplotlib.pyplot as plt
import wandb
from mut_info import *

from metrics import (
    accuracy,
    FGSM_accuracy_curve,
    Noise_accuracy_curve,
)


class All_Connections_Distillation(L.LightningModule):
    def __init__(self, student, teacher, coeffs, beta=0.5):
        super().__init__()
        self.student = student
        self.teacher = teacher
        self.teacher.requires_grad_(False)
        self.coeffs = coeffs
        self.beta = beta
        self.build_connections()
        self.loss = nn.CrossEntropyLoss()
        self.test_step_outputs = []

    def build_connections(self):
        self.connections = [
            (
                0,
                0,
                self.coeffs[0],
                Mutual_Info([Student_Teacher(8, 16)]),
                torch.nn.Parameter(torch.zeros(1)),
            ),
            (
                0,
                1,
                self.coeffs[1],
                Mutual_Info([UpStudent(8, 16), Student_Teacher(16, 32)]),
                torch.nn.Parameter(torch.zeros(1)),
            ),
            (
                0,
                2,
                self.coeffs[2],
                Mutual_Info([UpStudent(8, 32), Student_Teacher(32, 64)]),
                torch.nn.Parameter(torch.zeros(1)),
            ),
            (
                0,
                3,
                self.coeffs[3],
                Mutual_Info([UpStudent(8, 64), Linear_Model(64, 128)]),
                torch.nn.Parameter(torch.zeros(1)),
            ),
            (
                1,
                0,
                self.coeffs[4],
                Mutual_Info([DownStudent(16, 8), Student_Teacher(8, 16)]),
                torch.nn.Parameter(torch.zeros(1)),
            ),
            (
                1,
                1,
                self.coeffs[5],
                Mutual_Info([Student_Teacher(16, 32)]),
                torch.nn.Parameter(torch.zeros(1)),
            ),
            (
                1,
                2,
                self.coeffs[6],
                Mutual_Info([UpStudent(16, 32), Student_Teacher(32, 64)]),
                torch.nn.Parameter(torch.zeros(1)),
            ),
            (
                1,
                3,
                self.coeffs[7],
                Mutual_Info([UpStudent(16, 64), Linear_Model(64, 128)]),
                torch.nn.Parameter(torch.zeros(1)),
            ),
            (
                2,
                0,
                self.coeffs[8],
                Mutual_Info([DownStudent(32, 8), Student_Teacher(8, 16)]),
                torch.nn.Parameter(torch.zeros(1)),
            ),
            (
                2,
                1,
                self.coeffs[9],
                Mutual_Info([DownStudent(32, 16), Student_Teacher(16, 32)]),
                torch.nn.Parameter(torch.zeros(1)),
            ),
            (
                2,
                2,
                self.coeffs[10],
                Mutual_Info([Student_Teacher(32, 64)]),
                torch.nn.Parameter(torch.zeros(1)),
            ),
            (
                2,
                3,
                self.coeffs[11],
                Mutual_Info([UpStudent(32, 64), Linear_Model(64, 128)]),
                torch.nn.Parameter(torch.zeros(1)),
            ),
            (
                3,
                0,
                self.coeffs[12],
                Mutual_Info([Linear_Model(64, 16 * 15 * 15)]),
                torch.nn.Parameter(torch.zeros(1)),
            ),
            (
                3,
                1,
                self.coeffs[13],
                Mutual_Info([Linear_Model(64, 32 * 6 * 6)]),
                torch.nn.Parameter(torch.zeros(1)),
            ),
            (
                3,
                2,
                self.coeffs[14],
                Mutual_Info([Linear_Model(64, 64 * 2 * 2)]),
                torch.nn.Parameter(torch.zeros(1)),
            ),
            (
                3,
                3,
                self.coeffs[15],
                Mutual_Info([Linear_Model(64, 128)]),
                torch.nn.Parameter(torch.zeros(1)),
            ),
        ]

    def on_fit_start(self):
        self.model_device = next(self.student.parameters()).device
        for i in range(len(self.connections)):
            student_layer, teacher_layer, coeff, mu_info, param = self.connections[i]
            self.connections[i] = (
                student_layer,
                teacher_layer,
                coeff,
                mu_info.to(self.model_device),
                param.to(self.model_device),
            )

    def training_step(self, batch, batch_idx):
        x, y = batch

        *student_feat, student_logits = self.student.get_features(x, [0, 1, 2, 3, 4])
        teacher_feat = self.teacher.get_features(x, [0, 1, 2, 3])

        feat_loss_sum = 0
        for student_idx, teacher_idx, lmbda, mi_model, log_sigma in self.connections:
            my_stud_feat = mi_model(student_feat[student_idx])
            my_stud_feat = my_stud_feat.view(my_stud_feat.size(0), -1)
            my_teacher_feat = teacher_feat[teacher_idx].view(
                teacher_feat[teacher_idx].size(0), -1
            )

            sigma2 = torch.log(1 + torch.exp(log_sigma))
            feat_loss = ((my_teacher_feat - my_stud_feat) ** 2).sum(
                1
            ).mean() / 2 * sigma2 + 0.5 * torch.log(sigma2)
            feat_loss_sum += lmbda * feat_loss

        class_loss = self.loss(student_logits, y)
        loss = class_loss * (1.0 - self.beta) + feat_loss_sum * self.beta / 4

        labels_hat = torch.argmax(self.student(x), dim=1)
        train_acc = torch.sum(y == labels_hat).item() / (len(y) * 1.0)

        self.log_dict({"train_loss": loss, "train_acc": train_acc})
        return loss

    def log_accuracy(self, batches):
        test_acc = accuracy(self.student, batches)
        self.test_acc = test_acc
        self.log("test_acc", test_acc)

    def log_fgsm_accuracy(self, batches):
        fgsm_delta_arr, fgsm_accuracy_arr, auc_fgsm = FGSM_accuracy_curve(
            self.student, batches
        )
        self.log("AUC_FGSM", auc_fgsm)

        table_fgsm = wandb.Table(
            data=[[x, y] for (x, y) in zip(fgsm_delta_arr, fgsm_accuracy_arr)],
            columns=["delta", "fgsm_accuracy"],
        )
        wandb.log(
            {
                "FGSM_accuracy": wandb.plot.line(
                    table_fgsm, "delta", "fgsm_accuracy", title="FGSM_accuracy"
                )
            }
        )

    def log_noise_accuracy(self, batches):
        noise_delta_arr, noise_accuracy_arr, auc_noise = Noise_accuracy_curve(
            self.student, batches
        )
        self.log("AUC_Noise", auc_noise)

        table_noise = wandb.Table(
            data=[[x, y] for (x, y) in zip(noise_delta_arr, noise_accuracy_arr)],
            columns=["delta", "noise_accuracy"],
        )
        wandb.log(
            {
                "Noise_accuracy": wandb.plot.line(
                    table_noise, "delta", "noise_accuracy", title="Noise_accuracy"
                )
            }
        )

    def log_connection_picture(self):
        base_thickness = 14

        teacher_points = [(1, 3), (2, 3), (3, 3), (4, 3)]
        student_points = [(1, 0), (2, 0), (3, 0), (4, 0)]
        for point in teacher_points:
            plt.plot(*point, "ro")
        for point in student_points:
            plt.plot(*point, "go")

        coeffs_sum = sum(self.coeffs)
        for student_idx, teacher_idx, connection_value, _, _ in self.connections:
            thickness = round(connection_value / coeffs_sum, 3) * base_thickness
            x_values = [student_points[student_idx][0], teacher_points[teacher_idx][0]]
            y_values = [student_points[student_idx][1], teacher_points[teacher_idx][1]]
            plt.plot(x_values, y_values, "b", linestyle="-", linewidth=thickness)
        ax = plt.gca()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        wandb.log({"connection_plot": wandb.Image(plt)})

    def test_step(self, batch, batch_idx):
        self.test_step_outputs.append(batch)

    def on_test_epoch_end(self):
        self.log_accuracy(self.test_step_outputs)
        self.log_fgsm_accuracy(self.test_step_outputs)
        self.log_noise_accuracy(self.test_step_outputs)
        self.log_connection_picture()
        self.test_step_outputs.clear()

    def configure_optimizers(self):
        learnable_params = list(self.student.parameters())
        for _, _, _, model, sigma in self.connections:
            learnable_params += list(model.parameters())
            learnable_params += [sigma]

        lr = 0.1
        optimizer = torch.optim.SGD(learnable_params, lr=lr)
        wandb.log({"optimizer": "SGD", "lr": lr})
        return optimizer
