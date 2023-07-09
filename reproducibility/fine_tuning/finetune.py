from torch import nn
from torch import optim
import clip
import tqdm
import numpy as np
import torch
from reproducibility.embedders.internal_datasets import CLIPImageLabelDataset
from reproducibility.embedders.scheduler import cosine_lr
import pandas as pd
from torch.utils.data import DataLoader
from datetime import datetime
from torch.cuda.amp import autocast

from sklearn.metrics import f1_score

# Define a linear classifier
class LinearClassifier(nn.Module):
    def __init__(self, input_size, num_classes):
        super(LinearClassifier, self).__init__()
        self.fc = nn.Linear(input_size, num_classes)
        
    def forward(self, x):
        # Convert input matrix to the same data type as self.weight
        x = x.to(self.fc.weight.dtype)
        out = self.fc(x)
        return out
        
def convert_models_to_fp32(model):
    for p in model.parameters():
        p.data = p.data.float()
        if p.grad is not None:
            p.grad.data = p.grad.data.float()

class FineTuner:

    def __init__(self,
                args=None,
                logging=None,
                backbone=None,
                num_classes=None,
                lr=5e-5,
                weight_decay=0.2,
                warmup=0,
                comet_tracking=None,
                comet_tags=None
                ):
                
        self.args = args
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.logging = logging
        self.warmup = warmup
        self.hyper_params = {
            "lr": lr,
            "weight_decay": weight_decay
        }
        

        ##########################
        # Step 1. Model switch
        ##########################
        # Get preprocess regardless if it is CLIP backbone or not
        model_type = args.PC_CLIP_ARCH
        self.model, self.preprocess = clip.load(model_type,
                                                device=self.device,
                                                jit=False)  # Must set jit=False for training


        if self.args.model_name in ['plip', 'clip']:
            # TODO this is hard coded
            input_size = 512
            self.linear_classifier = LinearClassifier(input_size, num_classes)
            self.linear_classifier = self.linear_classifier.to(self.device)

            if backbone is not None:
                print('Load pre-trained PLIP model')
                if self.args.model_name == 'clip':
                    raise Exception('This is wrong.')
                self._load_plip_checkpoint(path=backbone)

            # parameters to be back-propagated.
            bp_params = list(self.model.parameters()) + list(self.linear_classifier.parameters())
        elif self.args.model_name.startswith('resnet'):
            model_version = int(self.args.model_name.split('resnet')[1])
            self.model = None
            if model_version == 18:
                from torchvision.models import resnet18, ResNet18_Weights
                self.model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
            elif model_version == 50:
                from torchvision.models import resnet50, ResNet50_Weights
                self.model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
            elif model_version == 101:
                from torchvision.models import resnet101, ResNet101_Weights
                self.model = resnet101(weights=ResNet101_Weights.IMAGENET1K_V1)
            # Modify the last fully connected layer
            self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
            self.model.to(self.device)
            # parameters to be back-propagated.
            bp_params = self.model.parameters()

        elif self.args.model_name.startswith('vit'):
            self.model = None
            if self.args.model_name == 'vit_b_16':
                from torchvision.models import vit_b_16, ViT_B_16_Weights
                self.model = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
            elif self.args.model_name == 'vit_b_32':
                from torchvision.models import vit_b_32, ViT_B_32_Weights
                self.model = vit_b_32(weights=ViT_B_32_Weights.IMAGENET1K_V1)
            # Modify the last fully connected layer
            self.model.heads.head = nn.Linear(self.model.heads.head.in_features, num_classes)
            self.model.to(self.device)
            # parameters to be back-propagated.
            bp_params = self.model.parameters()
        else:
            raise Exception('No such model.')

        if self.device == "cpu":
            self.model.float()
        else:
            if self.args.model_name in ['plip', 'clip']:
                clip.model.convert_weights(self.model)
        
        ##########################
        # Step 2. Optimizer
        ##########################
        
        self.classification_criterion = nn.CrossEntropyLoss()
        
        if self.args.optimizer == 'AdamW':
            self.optimizer = optim.AdamW(bp_params,
                                            lr=self.hyper_params["lr"],
                                            weight_decay=self.hyper_params["weight_decay"])
        elif self.args.optimizer == 'Adagrad':
            self.optimizer = optim.Adagrad(bp_params,
                                            lr=self.hyper_params["lr"],
                                            weight_decay=self.hyper_params["weight_decay"])
        elif self.args.optimizer == 'Adam':
            self.optimizer = optim.Adagrad(bp_params,
                                            lr=self.hyper_params["lr"],
                                            weight_decay=self.hyper_params["weight_decay"])
        elif self.args.optimizer == 'SGD':
            self.optimizer = optim.SGD(bp_params,
                                            lr=self.hyper_params["lr"],
                                            weight_decay=self.hyper_params["weight_decay"])



    def _load_plip_checkpoint(self,
                                path=None,
                                ):
        if path is None:
            raise Exception('No path provided.')
        self.model.load_state_dict(torch.load(path))


    def calculate_f1_score(self, outputs, labels, average='weighted'):
        # Convert tensor outputs and labels to numpy arrays
        outputs = outputs.cpu().numpy()
        labels = labels.cpu().numpy()
        # Convert outputs to predicted labels by selecting the index of the maximum value
        predicted_labels = np.argmax(outputs, axis=1)
        # Calculate the F1 score
        f1 = f1_score(labels, predicted_labels, average=average)
        return f1

    def forward_pass(self, images):
        if self.args.model_name in ['plip', 'clip']:
            image_features = self.model.encode_image(images)
            outputs = self.linear_classifier(image_features)
        else:
            with autocast():
                outputs = self.model(images)
        return outputs


    def valid_evaluation(self, dataloader, pbar, pbar_description="Currently Validating"):
        valid_loss_this_epoch = 0
        
        outputs_list = []
        labels_list = []
        
        self.model.eval()
        if self.args.model_name in ['plip', 'clip']:
            self.linear_classifier.eval()

        for batch in dataloader:
            pbar.set_description(pbar_description)

            with torch.no_grad():
                images, labels = batch
                images = images.to(self.device)
                labels = labels.to(self.device)

                # Forward pass
                outputs = self.forward_pass(images)
                
                # Append the output and label tensors to the lists
                outputs_list.append(outputs)
                labels_list.append(labels)

                # Compute the loss
                total_loss = self.classification_criterion(outputs, labels)
                valid_loss_this_epoch += total_loss.cpu().data.numpy()

        # Concatenate output and label tensors
        outputs_all = torch.cat(outputs_list, dim=0)
        labels_all = torch.cat(labels_list, dim=0)
        f1_weighted = self.calculate_f1_score(outputs_all, labels_all, average='weighted')
        f1_macro = self.calculate_f1_score(outputs_all, labels_all, average='macro')

        self.model.train()
        if self.args.model_name in ['plip', 'clip']:
            self.linear_classifier.train()

        return valid_loss_this_epoch, f1_weighted, f1_macro
    
        

    def tuner(self,
                train_dataframe,
                validation_dataframe,
                test_dataframe=None,
                save_directory='',
                batch_size=4,
                epochs=5,
                evaluation_steps=500,
                num_workers=1
                ):

        start_time = str(datetime.now())


        # Regardless the model_type, we will use the same CLIP Image Label Dataset loader.
        train_dataset = CLIPImageLabelDataset(train_dataframe, self.preprocess)
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers)
        validation_dataset = CLIPImageLabelDataset(validation_dataframe, self.preprocess)
        validation_dataloader = DataLoader(validation_dataset, batch_size=batch_size, num_workers=num_workers)
        if test_dataframe is not None:
            test_dataset = CLIPImageLabelDataset(test_dataframe, self.preprocess)
            test_dataloader = DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers)

        num_batches_per_epoch = len(train_dataloader)
        total_steps = len(train_dataloader) * epochs
        scheduler = cosine_lr(self.optimizer, self.hyper_params["lr"], self.warmup, total_steps)

        self.model.train()
        if self.args.model_name in ['plip', 'clip']:
            self.linear_classifier.train()

        performance_df = pd.DataFrame(index=np.arange(epochs), columns=['epoch','loss','f1_weighted','f1_macro'])

        for epoch in range(epochs):
            pbar = tqdm.tqdm(position=0, total=len(train_dataloader))
            pbar.set_description(f"{epoch}/{epochs}")

            train_loss_this_epoch = 0
            for i, batch in enumerate(train_dataloader):
                self.optimizer.zero_grad()
                step = num_batches_per_epoch * epoch + i
                scheduler(step)

                images, labels = batch
                images = images.to(self.device)
                labels = labels.to(self.device)

                # Forward pass
                outputs = self.forward_pass(images)
                
                # TODO delete soon: Verify the back-propagation is working.
                #print(self.model.visual.conv1.weight)
                #print(self.linear_classifier.fc.weight)

                # Compute the loss
                
                # Check if the tensor has one dimension
                if len(outputs.shape) == 1:
                    #print("Tensor has one dimension, unsqueeze it.")
                    outputs = outputs.unsqueeze(0)
                else:
                    pass
                total_loss = self.classification_criterion(outputs, labels)
                
                total_loss.backward()
                new_lr = scheduler(step)

                train_loss_this_epoch += total_loss.cpu().data.numpy()
                self.logging.info(f'[Train - this batch] epoch: {epoch}, batch: {i}, new learning rate: {new_lr}')
                #self.experiment.log_metric("learning_rate", new_lr, step=step)

                if self.device == "cpu":
                    self.optimizer.step() 
                else:
                    convert_models_to_fp32(self.model)
                    self.optimizer.step()
                    clip.model.convert_weights(self.model)
                pbar.update(1)

                if evaluation_steps == 0:
                    pass
                else:
                    if step % evaluation_steps == 0:
                        valid_loss_this_epoch, f1_weighted, f1_macro = self.valid_evaluation(validation_dataloader, pbar, pbar_description="Currently Validating")
                        pbar.set_description(f"{epoch}/{epochs}")
                        self.logging.info(f'[Validation - this batch] epoch: {epoch}, batch: {i}, total loss: {valid_loss_this_epoch}, f1_weighted: {f1_weighted}, f1_macro: {f1_macro}')

            self.logging.info(f'[Train - final] epoch: {epoch}, total loss: {train_loss_this_epoch}')

            # Validation at the end of each epoch
            valid_loss_this_epoch, f1_weighted, f1_macro = self.valid_evaluation(validation_dataloader, pbar, pbar_description="Currently Validating")
            pbar.set_description(f"{epoch}/{epochs}")
            self.logging.info(f'[Validation - final] epoch: {epoch}, total loss: {valid_loss_this_epoch}, f1_weighted: {f1_weighted}, f1_macro: {f1_macro}')

            performance_df.loc[epoch, 'epoch'] = epoch
            performance_df.loc[epoch, 'loss'] = valid_loss_this_epoch
            performance_df.loc[epoch, 'f1_weighted'] = f1_weighted
            performance_df.loc[epoch, 'f1_macro'] = f1_macro

            #torch.save(self.model.state_dict(), f"{save_directory}/epoch_{epoch}_{start_time}_model.pt")

            # This is just for debug only:
            # TODO: remove it.
            if test_dataframe is not None:
                valid_loss_this_epoch, f1_weighted, f1_macro = self.valid_evaluation(test_dataloader, pbar, pbar_description="Currently Testing")
                performance_df.loc[epoch, 'f1_test_weighted'] = f1_weighted
                performance_df.loc[epoch, 'f1_test_macro'] = f1_macro


            pbar.close()

        performance_df['f1_weighted'] = performance_df['f1_weighted'].astype(float)
        performance_df['f1_macro'] = performance_df['f1_macro'].astype(float)
        return performance_df
