from tqdm import trange
import torch
from torch import nn
from torch.utils.data import DataLoader


class Trainer:
    def train(
        self,
        num_epochs: int,
        device: torch.device,
        net: nn.Module,
        train_dl: DataLoader,
        val_dl: DataLoader,
        optim: torch.optim.Optimizer,
        loss_fn: nn.Module,
        *args,
        **kwargs,
    ):
        tloss = 0
        vloss = 0
        for i in trange(num_epochs, desc="Epoch"):
            net.train()
            tloss += self.train_once(i + 1, device, net, train_dl, optim, loss_fn)
            net.eval()
            vloss += self.validate_once(i + 1, device, net, val_dl, loss_fn)
            print(f"Avg Train Loss: {tloss/(i+1)}, Avg Val Loss: {vloss/(i+1)}")

    def train_once(
        self,
        epoch,
        device,
        net,
        train_dl,
        optim,
        loss_fn,
        *args,
        **kwargs,
    ):
        batch_loss = 0
        num_ites = len(train_dl)
        for i, batch in enumerate(train_dl):
            a, p, n = batch  # anchor, positive, negative
            a = a.to(device)
            p = p.to(device)
            n = n.to(device)
            optim.zero_grad()
            a_emb = net(a)
            p_emb = net(p)
            n_emb = net(n)
            loss = loss_fn(a_emb, p_emb, n_emb)  # must be triplet loss
            loss.backward()  # backprop
            optim.step()
            batch_loss += loss.item()
            print(
                f"[Epoch {epoch} Batch {i+1}/{num_ites} Train]: loss: {loss.item():.4f}"
            )
        return batch_loss / (i + 1)

    def validate_once(
        self,
        epoch,
        device,
        net,
        val_dl,
        loss_fn,
        *args,
        **kwargs,
    ):
        batch_loss = 0
        num_ites = len(val_dl)
        with torch.no_grad():
            for i, batch in enumerate(val_dl):
                a, p, n = batch
                a = a.to(device)
                p = p.to(device)
                n = n.to(device)
                a_emb = net(a)
                p_emb = net(p)
                n_emb = net(n)
                loss = loss_fn(a_emb, p_emb, n_emb)
                batch_loss += loss.item()
                print(
                    f"[Epoch {epoch} Batch {i+1}/{num_ites} Val]: loss: {loss.item():.4f}"
                )
        return batch_loss / (i + 1)
