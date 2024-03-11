import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
import wandb

from metrics import (
    accuracy,
    FGSM_accuracy_curve,
    Noise_accuracy_curve,
)


class Hinton_Distillation(L.LightningModule):
    def __init__(self, student, teacher, beta=0.8, temp=2.0):
        super().__init__()
        self.student = student
        self.teacher = teacher
        self.teacher.requires_grad_(False)
        self.beta = beta
        self.temp = temp
        self.loss = nn.CrossEntropyLoss()
        self.test_step_outputs = []

    def distillation_loss(self, out, teacher_out):
        g = nn.Softmax(dim=1)(out / self.temp)
        f = F.log_softmax(teacher_out / self.temp)
        return nn.KLDivLoss(reduction="batchmean")(f, g)

    def training_step(self, batch, batch_idx):
        x, y = batch
        out = self.student(x)
        with torch.no_grad():
            teacher_out = self.teacher(x)

        student_loss = self.loss(out, y)
        distill_loss = self.distillation_loss(out, teacher_out)
        loss = (1 - self.beta) * student_loss + self.beta * distill_loss

        labels_hat = torch.argmax(out, dim=1)
        train_acc = torch.sum(y == labels_hat).item() / (len(y) * 1.0)

        self.log_dict({"train_loss": loss, "train_acc": train_acc})
        return loss

    def log_accuracy(self, batches):
        test_acc = accuracy(self.student, batches)
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

    def test_step(self, batch, batch_idx):
        self.test_step_outputs.append(batch)

    def on_test_epoch_end(self):
        self.log_accuracy(self.test_step_outputs)
        self.log_fgsm_accuracy(self.test_step_outputs)
        self.log_noise_accuracy(self.test_step_outputs)
        self.test_step_outputs.clear()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.student.parameters())
        return optimizer
