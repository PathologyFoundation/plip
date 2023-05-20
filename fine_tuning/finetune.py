from comet_ml import Experiment
from torch import nn
from torch import optim
import clip
import tqdm
import numpy as np
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
import torch
import math
from torch.optim.lr_scheduler import LinearLR
from embedders.internal_datasets import CLIPImageCaptioningDataset, CLIPCaptioningDataset, CLIPImageDataset, CLIPImageLabelDataset
from embedders.transform import _train_transform
from embedders.scheduler import cosine_lr
import pandas as pd
from torch.utils.data import DataLoader
from PIL import Image
from datetime import datetime
from torch.cuda.amp import autocast

from sklearn.metrics import f1_score
from .linear_classifier import LinearClassifier


def unwrap_model(model):
    if hasattr(model, 'module'):
        return model.module
    else:
        return model

def zero_shot_classification(model, preprocess, images, labels, device, num_workers=1, batch_size=32):
    image_embeddings = image_embedder(model, preprocess, images, device, num_workers, batch_size)
    text_embeddings = text_embedder(model, labels, device, num_workers, batch_size)

    score = image_embeddings.dot(text_embeddings.T)
    predictions = [labels[np.argmax(i)] for i in score]

    return predictions


def image_embedder(model, preprocess, list_of_images, device="cuda", num_workers=1, batch_size=32):
    train_dataset = CLIPImageDataset(list_of_images, preprocess)
    dataloader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers)

    image_embeddings = []

    total = len(list_of_images) // batch_size
    pbar = tqdm.tqdm(total=total, position=0)
    with torch.no_grad():
        for images in dataloader:
            images = images.to(device)

            image_embeddings.extend(model.encode_image(images).detach().cpu().numpy())

            pbar.update(1)
        pbar.close()

    image_embeddings = np.array(image_embeddings)
    image_embeddings = image_embeddings / np.linalg.norm(image_embeddings, axis=1, keepdims=True)
    return image_embeddings

def text_embedder(model, list_of_labels, device="cuda", num_workers=1, batch_size=32):
    train_dataset = CLIPCaptioningDataset(list_of_labels)
    dataloader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers)
    text_embeddings = []
    total = len(list_of_labels) // batch_size

    pbar = tqdm.tqdm(total=total, position=0)
    with torch.no_grad():
        for captions in dataloader:
            idx = clip.tokenize(captions, truncate=True).to(device)
            text_embeddings.extend(model.encode_text(idx).detach().cpu().numpy())

            pbar.update(1)

        pbar.close()

    text_embeddings = np.array(text_embeddings)
    text_embeddings = text_embeddings / np.linalg.norm(text_embeddings, axis=1, keepdims=True)

    return text_embeddings

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
        # Get preprocess regardless if it is CLIP backbone or not (or like EfficientNet)
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

        elif self.args.model_name.startswith('EfficientNet_b'):
            model_version = int(self.args.model_name.split('_b')[1])
            
            self.model = None

            if model_version == 0:
                from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
                self.model = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
            elif model_version == 1:
                from torchvision.models import efficientnet_b1, EfficientNet_B1_Weights
                self.model = efficientnet_b1(weights=EfficientNet_B1_Weights.IMAGENET1K_V1)
            elif model_version == 2:
                from torchvision.models import efficientnet_b2, EfficientNet_B2_Weights
                self.model = efficientnet_b2(weights=EfficientNet_B2_Weights.IMAGENET1K_V1)
            elif model_version == 3:
                from torchvision.models import efficientnet_b3, EfficientNet_B3_Weights
                self.model = efficientnet_b3(weights=EfficientNet_B3_Weights.IMAGENET1K_V1)
            elif model_version == 4:
                from torchvision.models import efficientnet_b4, EfficientNet_B4_Weights
                self.model = efficientnet_b4(weights=EfficientNet_B4_Weights.IMAGENET1K_V1)
            elif model_version == 5:
                from torchvision.models import efficientnet_b5, EfficientNet_B5_Weights
                self.model = efficientnet_b5(weights=EfficientNet_B5_Weights.IMAGENET1K_V1)
            elif model_version == 6:
                from torchvision.models import efficientnet_b6, EfficientNet_B6_Weights
                self.model = efficientnet_b6(weights=EfficientNet_B6_Weights.IMAGENET1K_V1)
            elif model_version == 7:
                from torchvision.models import efficientnet_b7, EfficientNet_B7_Weights
                self.model = efficientnet_b7(weights=EfficientNet_B7_Weights.IMAGENET1K_V1)
            
            # Modify the last fully connected layer
            self.model.classifier[1] = nn.Linear(self.model.classifier[1].in_features, num_classes)
            for param in self.model.parameters():
                param.data = param.data.float()
            self.model.to(self.device)
            # parameters to be back-propagated.
            bp_params = self.model.parameters()
        
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

    def valid_evaluation(self, dataloader, pbar, pbar_description="Currently Validating"):
        valid_loss_this_epoch = 0
        
        outputs_list = []
        labels_list = []

        for batch in dataloader:
            pbar.set_description(pbar_description)

            with torch.no_grad():
                images, labels = batch
                images = images.to(self.device)
                labels = labels.to(self.device)

                # Forward pass
                if self.args.model_name in ['plip', 'clip']:
                    image_features = self.model.encode_image(images)
                    outputs = self.linear_classifier(image_features)
                else: # EfficientNet
                    with autocast():
                        outputs = self.model(images)

                
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
                if self.args.model_name in ['plip', 'clip']:
                    image_features = self.model.encode_image(images)
                    outputs = self.linear_classifier(image_features)
                else: # EfficientNet
                    with autocast():
                        outputs = self.model(images)

                # TODO delete soon: Verify the back-propagation is working.
                #print(self.model.visual.conv1.weight)
                #print(self.linear_classifier.fc.weight)

                # Compute the loss
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
