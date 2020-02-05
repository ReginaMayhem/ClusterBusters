import pytorch_lightning as pl
import torch.nn as nn
import torch
import torch.nn.functional as F


class MeanEmbeddings_pl(pl.LightningModule):
    def __init__(self, n_features, n_hidden, hidden_size, dropout=0.1):
        super().__init__()
        
        self.n_features = n_features
        self.n_hidden = n_hidden
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.learning_rate = learning_rate

        for i in range(n_hidden):
            if i == 0:
                setattr(self, f"hidden_{i}", nn.Linear(self.n_features*2, self.hidden_size))
            else:
                setattr(self, f"hidden_{i}", nn.Linear(self.hidden_size, self.hidden_size))
        
        self.projection = nn.Linear(self.hidden_size, 2)
        
    def forward(self, target, reference):
        reference_mean = torch.mean(reference, dim=0)
        
        x = torch.cat((reference_mean, target))
        
        for i in range(self.n_hidden):
            x = F.dropout(torch.relu(getattr(self, f"hidden_{i}")(x)), self.dropout)
            
        output = self.projection(x)
        
        return output
    
    def training_step(self, batch, batch_idx):
        # REQUIRED
        x, y, ref = batch
        y_hat = self.forward(torch.Tensor(x), torch.Tensor(ref))
        loss = F.cross_entropy(y_hat, y)
        tensorboard_logs = {'train_loss': loss}
        return {'loss': loss, 'log': tensorboard_logs}

    def validation_step(self, batch, batch_idx):
        # OPTIONAL
        x, y, ref = batch
        y_hat = self.forward(torch.Tensor(x), torch.Tensor(ref))
        
        acc = torch.Tensor(y == y_hat.item())
        return {'val_loss': F.cross_entropy(y_hat, y), "val_acc": acc}

    def validation_end(self, outputs):
        # OPTIONAL
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        avg_acc = np.stack([x['val_acc'] for x in outputs]).mean()
        tensorboard_logs = {'val_loss': avg_loss}
        return {'avg_val_loss': avg_loss, "val_acc": avg_acc, 'log': tensorboard_logs}

    def configure_optimizers(self):
        # REQUIRED
        # can return multiple optimizers and learning_rate schedulers
        # (LBFGS it is automatically supported, no need for closure function)
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

    @pl.data_loader
    def train_dataloader(self):
        # REQUIRED
        return train

    @pl.data_loader
    def val_dataloader(self):
        # OPTIONAL
        return test
    

   
class SeparateEmbeddings(nn.Module):
    def __init__(self, options):
        super().__init__()
        
        self.n_features = options["n_features"]
        self.n_hidden = options["n_hidden"]
        self.hidden_size = options["hidden_size"]
        self.dropout = options["dropout"]

        
        for i in range(self.n_hidden):
            if i == 0:
                setattr(self, f"hidden_{i}", nn.Linear(self.n_features*2, self.hidden_size))
            else:
                setattr(self, f"hidden_{i}", nn.Linear(self.hidden_size, self.hidden_size))
        
        self.projection = nn.Linear(self.hidden_size, 2)
        
    def forward(self, target, reference):
        x = torch.cat((reference, target.unsqueeze(0).repeat(reference.size(0), 1)), dim=1)
        
        
        for i in range(self.n_hidden):
            x = F.dropout(torch.relu(getattr(self, f"hidden_{i}")(x)), self.dropout)
            
        x = torch.mean(x, dim=0)
        output = self.projection(x)
        
        return output
    
class TransformerModel(nn.Module):
    def __init__(self, options):
        super().__init__()
        
        self.n_features = options["n_features"]
        self.n_layers = options["n_layers"]
        self.dim_ff = options["dim_ff"]
        self.n_heads = options["n_heads"]
        self.dropout = options["dropout"]

        
        self.enc_layer = nn.TransformerEncoderLayer(d_model=self.n_features + 1, nhead=self.n_heads, dim_feedforward = self.dim_ff, dropout=self.dropout)
        self.enc = nn.TransformerEncoder(self.enc_layer, num_layers = self.n_layers)
        self.projection = nn.Linear(self.n_features + 1, 2)
        
    def forward(self, target, reference):
        x = torch.cat((reference, target.unsqueeze(0).repeat(reference.size(0), 1)), dim=1)
        
        reference_in = torch.cat((reference, torch.zeros((reference.size(0),1)).to(reference.device)), dim=1)
        target_in = torch.cat((target, torch.Tensor([1]).to(target.device)), dim=0)
        x = torch.cat((reference_in, target_in.unsqueeze(0)), dim=0)
        out = self.enc(x.unsqueeze(1))
        output = self.projection(out[-1])
        
        return output.squeeze()
            
            
class SeparateEmbeddings_pl(pl.LightningModule):
    def __init__(self, n_features, n_hidden, hidden_size, learning_rate, train, test, dropout=0.1):
        super().__init__()
        
        self.n_features = n_features
        self.n_hidden = n_hidden
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.train_dl = train
        self.test_dl = test
        
        for i in range(n_hidden):
            if i == 0:
                setattr(self, f"hidden_{i}", nn.Linear(self.n_features*2, self.hidden_size))
            else:
                setattr(self, f"hidden_{i}", nn.Linear(self.hidden_size, self.hidden_size))
        
        self.projection = nn.Linear(self.hidden_size, 2)
        
    def forward(self, target, reference):
        x = torch.cat((reference, target.unsqueeze(0).repeat(reference.size(0), 1)), dim=1)
        
        
        for i in range(self.n_hidden):
            x = F.dropout(torch.relu(getattr(self, f"hidden_{i}")(x)), self.dropout)
            
        x = torch.mean(x, dim=0)
        output = self.projection(x)
        
        return output
    
    def training_step(self, batch, batch_idx):
        # REQUIRED
        x, y, ref = batch
        device = next(self.parameters()).device
        y_hat = self.forward(torch.Tensor(x).to(device), torch.Tensor(ref).to(device))
        loss = F.cross_entropy(y_hat.unsqueeze(0), torch.LongTensor([y]).to(device))
        tensorboard_logs = {'train_loss': loss}
        return {'loss': loss, 'log': tensorboard_logs}

    def validation_step(self, batch, batch_idx):
        # OPTIONAL
        x, y, ref = batch
        device = next(self.parameters()).device

        y_hat = self.forward(torch.Tensor(x).to(device), torch.Tensor(ref).to(device))
        acc = torch.Tensor([y == torch.argmax(y_hat).item()]).to(device)
        return {'val_loss': F.cross_entropy(y_hat.unsqueeze(0), torch.LongTensor([y]).to(device)), "val_acc": acc}

    def validation_end(self, outputs):
        # OPTIONAL
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        avg_acc = torch.stack([x['val_acc'] for x in outputs]).mean()
        tensorboard_logs = {'val_loss': avg_loss}
        return {'avg_val_loss': avg_loss, "val_acc": avg_acc, 'log': tensorboard_logs}

    def configure_optimizers(self):
        # REQUIRED
        # can return multiple optimizers and learning_rate schedulers
        # (LBFGS it is automatically supported, no need for closure function)
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

    @pl.data_loader
    def train_dataloader(self):
        # REQUIRED
        return self.train_dl

    @pl.data_loader
    def val_dataloader(self):
        # OPTIONAL
        return self.test_dl
        
        
        