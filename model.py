import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class D(nn.Module):
    def __init__(self):
        super(D, self).__init__()

    def forward(self, p, z):
        z = z.detach()

        p = F.normalize(p, p=2, dim=1)
        z = F.normalize(z, p=2, dim=1)
        return -(p * z).sum(dim=1).mean()


class Model(nn.Module):
    def __init__(self, args, downstream=False):
        super(Model, self).__init__()
        resnet50 = models.resnet50(pretrained=False)
        proj_hid, proj_out = args.proj_hidden, args.proj_out
        pred_hid, pred_out = args.pred_hidden, args.pred_out


        self.backbone = nn.Sequential(*list(resnet50.children())[:-1])
        backbone_in_channels = resnet50.fc.in_features

        self.projection = nn.Sequential(
            nn.Linear(backbone_in_channels, proj_hid),
            nn.BatchNorm1d(proj_hid),
            nn.ReLU(inplace=True),
            nn.Linear(proj_hid, proj_hid),
            nn.BatchNorm1d(proj_hid),
            nn.ReLU(inplace=True),
            nn.Linear(proj_hid, proj_out),
            nn.BatchNorm1d(proj_out),
        )

        self.prediction = nn.Sequential(
            nn.Linear(proj_out, pred_hid),
            nn.BatchNorm1d(pred_hid),
            nn.ReLU(inplace=True),
            nn.Linear(pred_hid, pred_out),
        )

        self.d = D()
        if args.checkpoints is not None and downstream:
            self.load_state_dict(torch.load(args.checkpoints)['model_state_dict'])


    def forward(self, x1, x2):
        out1 = self.backbone(x1).squeeze()
        z1 = self.projection(out1)
        p1 = self.prediction(z1)

        out2 = self.backbone(x2).squeeze()
        z2 = self.projection(out2)
        p2 = self.prediction(z2)

        d1 = self.d(p1, z2) / 2.
        d2 = self.d(p2, z1) / 2.

        return d1, d2
    


class DownStreamModel(nn.Module):
    def __init__(self, args, n_classes=6):
        super(DownStreamModel, self).__init__()
        self.simsiam = Model(args, downstream=True)
        hidden = 512

        self.net_backbone = nn.Sequential(
            self.simsiam.backbone,
        )

        for name, param in self.net_backbone.named_parameters():
            param.requires_grad = False

        self.net_projection = nn.Sequential(
            self.simsiam.projection,
        )

        for name, param in self.net_projection.named_parameters():
            param.requires_grad = False

        self.out = nn.Sequential(
            # nn.Flatten(),  # 展平为 [batch_size, 2048] 原作者是故意的还是版本问题?
            nn.Linear(args.proj_out, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_classes),
        )

    def forward(self, x):
        out = self.net_backbone(x)  # torch.Size([16, 2048, 1, 1])
        out = torch.flatten(out, 1)
        # out = self.net_projection(out)
        out = self.out(out)
        return out
