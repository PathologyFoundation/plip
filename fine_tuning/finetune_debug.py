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
import torch.nn.functional as F
from torch.cuda.amp import autocast
from sklearn.metrics import f1_score

from metrics import eval_metrics


# Define a linear classifier
class LinearClassifier(nn.Module):
    def __init__(self, input_size, num_classes):
        super(LinearClassifier, self).__init__()
        self.relu = nn.ReLU()
        self.fc = nn.Linear(input_size, num_classes)
        
    def forward(self, x):
        # Convert input matrix to the same data type as self.weight
        x = x.to(self.fc.weight.dtype)
        # Apply normalization
        #x = F.normalize(x, p=2, dim=1)
        #x = self.relu(x) # Do not add non-linear, cuz that will be different from linear probing arch.
        out = self.fc(x)
        return out

    def l2_penalty(self):
        l2_norm = torch.tensor(0.0).to(self.fc.weight.device)
        for param in self.parameters():
            l2_norm += torch.norm(param, p=2)
        return l2_norm
        
def convert_models_to_fp32(model):
    for p in model.parameters():
        p.data = p.data.float()
        if p.grad is not None:
            p.grad.data = p.grad.data.float()

class FineTuner_debug:

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
        # Update on 5/22: self.preprocess is the same as linear probing preprocess.
        # But the embedding carried out from PLIP with same backbone is different on Kather test.
        # Both linear probing and fine-tuning have the same model weights for ViT for both PLIP and CLIP.
        if self.args.model_name in ['plip', 'clip']:
            # TODO this is hard coded
            input_size = 512
            self.linear_classifier = LinearClassifier(input_size, num_classes)
            self.linear_classifier = self.linear_classifier.to(self.device)

            '''
            print(self.linear_classifier)
            # replace weight from linear probing
            import pickle
            with open('/oak/stanford/groups/jamesz/pathtweets/results/fine_tuning/linclassifier_temp.pkl', 'rb') as f:
                lin_classifier = pickle.load(f)
            self.linear_classifier.fc.weight.data = torch.FloatTensor(lin_classifier.coef_).to(self.device)
            self.linear_classifier.fc.bias.data = torch.FloatTensor(lin_classifier.intercept_).to(self.device)

            print('Success')
            #exit()
            '''


            if backbone is not None:
                print('Load pre-trained PLIP model')
                if self.args.model_name == 'clip':
                    raise Exception('This is wrong.')
                self._load_plip_checkpoint(path=backbone)

            #print(self.model.visual.conv1.weight)
            #exit()

            # parameters to be back-propagated.
            bp_params = list(self.model.parameters()) + list(self.linear_classifier.parameters())
            '''
            # Freeze the model
            for param in self.model.parameters():
                param.requires_grad = False
            bp_params = self.linear_classifier.parameters()
            '''
        
        elif self.args.model_name == 'MuDiPath':
            # TODO this is hard coded
            input_size = 1024
            self.linear_classifier = LinearClassifier(input_size, num_classes)
            self.linear_classifier = self.linear_classifier.to(self.device)

            from embedders.mudipath import build_densenet
            self.model = build_densenet(download_dir="/oak/stanford/groups/jamesz/pathtweets/models/",
                                        pretrained="mtdp")  # TODO fixed path
            # Modify the last fully connected layer
            #for param in self.model.parameters():
            #    param.data = param.data.float()
            self.model.to(self.device)
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
            image_features = F.normalize(image_features, p=2, dim=1)
            outputs = self.linear_classifier(image_features)
        elif self.args.model_name == 'MuDiPath':
            with autocast():
                image_features = self.model(images).squeeze()
                outputs = self.linear_classifier(image_features)
        elif self.args.model_name.startswith('EfficientNet'):
            with autocast():
                outputs = self.model(images)
        else:
            with autocast():
                outputs = self.model(images)
        return outputs


    def valid_evaluation(self, dataloader, pbar, pbar_description="Currently Validating"):
        valid_loss_this_epoch = 0
        
        outputs_list = []
        labels_list = []

        #'''
        self.model.eval()
        if self.args.model_name in ['plip', 'clip', 'MuDiPath']:
            self.linear_classifier.eval()
        #'''
        
        #tensor_list = []
        for batch in dataloader:
            pbar.set_description(pbar_description)

            with torch.no_grad():
                images, labels = batch
                images = images.to(self.device)
                labels = labels.to(self.device)

                # Forward pass
                '''
                if pbar_description == 'Currently Testing':
                    if self.args.model_name in ['plip', 'clip']:
                        image_features = self.model.encode_image(images)
                        image_features = F.normalize(image_features, p=2, dim=1)
                        tensor_list.append(image_features)
                        outputs = self.linear_classifier(image_features)
                    elif self.args.model_name == 'MuDiPath':
                        with autocast():
                            image_features = self.model(images).squeeze()
                            outputs = self.linear_classifier(image_features)
                    elif self.args.model_name.startswith('EfficientNet'):
                        with autocast():
                            outputs = self.model(images)
                    else:
                        with autocast():
                            outputs = self.model(images)
                else:
                '''
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


        # method 1
        f1_weighted = self.calculate_f1_score(outputs_all, labels_all, average='weighted')
        f1_macro = self.calculate_f1_score(outputs_all, labels_all, average='macro')

        # Concatenate tensors along dimension 0
        #concatenated_tensor = torch.cat(tensor_list, dim=0)
        #print(concatenated_tensor.shape)
        #print(concatenated_tensor)
        
        '''
        # method 2
        outputs_all = outputs_all.cpu().numpy()
        labels_all = labels_all.cpu().numpy()
        # Convert outputs to predicted labels by selecting the index of the maximum value
        predicted_labels = np.argmax(outputs_all, axis=1)
        perf = eval_metrics(y_true=labels_all, y_pred=predicted_labels, y_pred_proba=outputs_all, average_method='weighted')
        print(perf)
        '''


        #'''
        self.model.train()
        if self.args.model_name in ['plip', 'clip', 'MuDiPath']:
            self.linear_classifier.train()
        #'''

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
        
        #pbar = tqdm.tqdm(position=0, total=len(train_dataloader))
        #test_loss_this_epoch, f1_weighted_test, f1_macro_test = self.valid_evaluation(test_dataloader, pbar, pbar_description="Currently Testing")
        #exit()


        num_batches_per_epoch = len(train_dataloader)
        total_steps = len(train_dataloader) * epochs
        scheduler = cosine_lr(self.optimizer, self.hyper_params["lr"], self.warmup, total_steps)

        self.model.train()
        if self.args.model_name in ['plip', 'clip', 'MuDiPath']:
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
                '''
                if i == 10:
                    print(outputs.shape)
                    print(outputs)
                    print(labels.shape)
                    print(labels)
                    exit()
                '''

                total_loss = self.classification_criterion(outputs, labels)
                #l2_loss = self.linear_classifier.l2_penalty()
                #print(total_loss, l2_loss)
                #total_loss = total_loss + l2_loss

                total_loss.backward()
                new_lr = scheduler(step)

                train_loss_this_epoch += total_loss.cpu().data.numpy()
                #self.logging.info(f'[Train - this batch] epoch: {epoch}, batch: {i}, new learning rate: {new_lr}')
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
                        test_loss_this_epoch, f1_weighted_test, f1_macro_test = self.valid_evaluation(test_dataloader, pbar, pbar_description="Currently Testing")
                        print(f'[epoch: {epoch}, batch: {i}] val loss: {valid_loss_this_epoch:.4f}, test loss: {test_loss_this_epoch:.4f}, f1_weighted_val: {f1_weighted:.4f}, f1_macro_val: {f1_macro:.4f}, f1_weighted_test: {f1_weighted_test:.4f}, f1_macro_test: {f1_macro_test:.4f}')

            self.logging.info(f'[Train - final] epoch: {epoch}, total loss: {train_loss_this_epoch}')

            # Validation at the end of each epoch
            valid_loss_this_epoch, f1_weighted, f1_macro = self.valid_evaluation(validation_dataloader, pbar, pbar_description="Currently Validating")
            pbar.set_description(f"{epoch}/{epochs}")
            test_loss_this_epoch, f1_weighted_test, f1_macro_test = self.valid_evaluation(test_dataloader, pbar, pbar_description="Currently Testing")
            print(f'[epoch: {epoch}, batch: {i}] val loss: {valid_loss_this_epoch:.4f}, test loss: {test_loss_this_epoch:.4f}, f1_weighted_val: {f1_weighted:.4f}, f1_macro_val: {f1_macro:.4f}, f1_weighted_test: {f1_weighted_test:.4f}, f1_macro_test: {f1_macro_test:.4f}')

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
