from transformers import get_linear_schedule_with_warmup
from transformers import GPT2LMHeadModel
import pytorch_lightning as pl
from torch.optim import AdamW

class GPT2PreTrained(pl.LightningModule):
    def __init__(self, lr=5e-5, max_epochs=15, warmup_steps=1000):
        super().__init__()
        self.model = GPT2LMHeadModel.from_pretrained("gpt2")
        self.save_hyperparameters()
        self.val_dataloader_ = None
        self.test_dataloader_ = None
        self.train_dataloader_ = None

    def forward(self, input_ids, labels, attention_mask):     
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        return outputs
    
    def common_step(self, batch, batch_idx):
        outputs = self(**batch)
        loss = outputs.loss

        return loss
      
    def training_step(self, batch, batch_idx):
        loss = self.common_step(batch, batch_idx)     
        # logs metrics for each training_step,
        # and the average across the epoch
        self.log("training_loss", loss)

        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.common_step(batch, batch_idx)     
        self.log("validation_loss", loss, on_epoch=True)

        return loss

    def test_step(self, batch, batch_idx):
        loss = self.common_step(batch, batch_idx)     

        return loss

    def configure_optimizers(self):
        # create optimizer
        optimizer = AdamW(self.model.parameters(), lr=self.hparams.lr)
        # create learning rate scheduler
        training_steps = self.hparams.max_epochs * len(self.train_dataloader_)
        lr_scheduler = {'scheduler': get_linear_schedule_with_warmup(optimizer,
                                                    num_warmup_steps=self.hparams.warmup_steps,
                                                    num_training_steps=training_steps),
                        'name': 'learning_rate',
                        'interval':'step',
                        'frequency': 1}
        
        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler}

    def set_train_dataloader(self, train_dataloader):
        self.train_dataloader_ = train_dataloader

    def set_valid_dataloader(self, valid_dataloader):
        self.val_dataloader_ = valid_dataloader

    def set_test_dataloader(self, test_dataloader):
        self.test_dataloader_ = test_dataloader

    def train_dataloader(self):
        return self.train_dataloader_
    
    def val_dataloader(self):
        return self.val_dataloader_
    
    def test_dataloader(self):
        return self.test_dataloader_
