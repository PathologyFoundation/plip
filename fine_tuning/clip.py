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

class CLIPTuner:

    def __init__(self,
                args=None,
                logging=None,
                model_type="ViT-B/32",
                backbone=None,
                num_classes=None,
                lr=5e-5,
                weight_decay=0.2,
                warmup=50,
                comet_tracking=None,
                px_size=224,
                comet_tags=None
                ):
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"

        self.args = args
        self.logging = logging

        self.model, self.preprocess = clip.load(model_type,
                                                device=self.device,
                                                jit=False)  # Must set jit=False for training
        if backbone is not None:
            print('Load pre-trained PLIP model')
            self._load_checkpoint(path=backbone)
        
        self.warmup = warmup
        self.train_preprocess = _train_transform(first_resize = self.args.first_resize,
                                                n_px = self.args.pxsize
                                                )
        if self.device == "cpu":
            self.model.float()
        else:
            clip.model.convert_weights(self.model)
        
        self.hyper_params = {
            "lr": lr,
            "weight_decay": weight_decay
        }

        #self.experiment.log_parameters(self.hyper_params)

        self.loss_img = nn.CrossEntropyLoss()
        self.loss_txt = nn.CrossEntropyLoss()
        if self.args.optimizer == 'AdamW':
            self.optimizer = optim.AdamW(self.model.parameters(),
                                        lr=self.hyper_params["lr"],
                                        weight_decay=self.hyper_params["weight_decay"])
        elif self.args.optimizer == 'Adagrad':
            self.optimizer = optim.Adagrad(self.model.parameters(),
                                        lr=self.hyper_params["lr"],
                                        weight_decay=self.hyper_params["weight_decay"])
        elif self.args.optimizer == 'Adam':
            self.optimizer = optim.Adagrad(self.model.parameters(),
                                        lr=self.hyper_params["lr"],
                                        weight_decay=self.hyper_params["weight_decay"])

        # TODO this is hard coded
        input_size = 512
        self.linear_classifier = LinearClassifier(input_size, num_classes)
        self.linear_classifier = self.linear_classifier.to(self.device)
        
        self.classification_criterion = nn.CrossEntropyLoss()


    def _load_checkpoint(self,
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

    def valid_evaluation(self, clip, dataloader, pbar, pbar_description="Currently Validating"):
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
                image_features = self.model.encode_image(images)
                outputs = self.linear_classifier(image_features)
                
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
                save_directory,
                batch_size=4,
                epochs=5,
                evaluation_steps=500,
                num_workers=1
                ):

        start_time = str(datetime.now())
        train_dataset = CLIPImageLabelDataset(train_dataframe, self.train_preprocess)
        validation_dataset = CLIPImageLabelDataset(validation_dataframe, self.preprocess)
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers)
        validation_dataloader = DataLoader(validation_dataset, batch_size=batch_size, num_workers=num_workers)
        num_batches_per_epoch = len(train_dataloader)
        total_steps = len(train_dataloader) * epochs
        scheduler = cosine_lr(self.optimizer, self.hyper_params["lr"], self.warmup, total_steps)

        #with self.experiment.train():

        self.model.train()

        performance_df = pd.DataFrame(index=np.arange(epochs+1), columns=['epoch','loss','f1_weighted','f1_macro'])

        for epoch in range(epochs+1):
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
                image_features = self.model.encode_image(images)
                outputs = self.linear_classifier(image_features)

                # Compute the loss
                # TODO confirming if logit is needed.
                total_loss = self.classification_criterion(outputs, labels)
                
                logit_scale = self.model.logit_scale.exp()
                #self.experiment.log_metric("logit_scale", logit_scale.item(), step=step)

                ground_truth = torch.arange(len(images), dtype=torch.long, device=self.device)

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

                with torch.no_grad():
                    unwrap_model(self.model).logit_scale.clamp_(0, math.log(100))

                if evaluation_steps == 0:
                    pass
                else:
                    if step % evaluation_steps == 0:
                        valid_loss_this_epoch, f1_weighted, f1_macro = self.valid_evaluation(clip, validation_dataloader, pbar, pbar_description="Currently Validating")
                        pbar.set_description(f"{epoch}/{epochs}")
                        self.logging.info(f'[Validation - this batch] epoch: {epoch}, batch: {i}, total loss: {valid_loss_this_epoch}, f1_weighted: {f1_weighted}, f1_macro: {f1_macro}')

            self.logging.info(f'[Train - final] epoch: {epoch}, total loss: {train_loss_this_epoch}')

            # Validation at the end of each epoch
            valid_loss_this_epoch, f1_weighted, f1_macro = self.valid_evaluation(clip, validation_dataloader, pbar, pbar_description="Currently Validating")
            pbar.set_description(f"{epoch}/{epochs}")
            self.logging.info(f'[Validation - final] epoch: {epoch}, total loss: {valid_loss_this_epoch}, f1_weighted: {f1_weighted}, f1_macro: {f1_macro}')

            performance_df.loc[epoch, 'epoch'] = epoch
            performance_df.loc[epoch, 'loss'] = valid_loss_this_epoch
            performance_df.loc[epoch, 'f1_weighted'] = f1_weighted
            performance_df.loc[epoch, 'f1_macro'] = f1_macro

            #torch.save(self.model.state_dict(), f"{save_directory}/epoch_{epoch}_{start_time}_model.pt")

            pbar.close()

        performance_df['f1_weighted'] = performance_df['f1_weighted'].astype(float)
        performance_df['f1_macro'] = performance_df['f1_macro'].astype(float)
        return performance_df
