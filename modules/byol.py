"""
MIT License

Copyright (c) 2020 Phil Wang
https://github.com/lucidrains/byol-pytorch/

Adjusted to de-couple for data loading, parallel training
"""

import copy
import random
from functools import wraps

import torch
from torch import nn
import torch.nn.functional as F
import torchvision.models as models
from einops.layers.torch import Rearrange # Import for ViT processing
# helper functions


def default(val, def_val):
    return def_val if val is None else val


def flatten(t):
    return t.reshape(t.shape[0], -1)


def singleton(cache_key):
    def inner_fn(fn):
        @wraps(fn)
        def wrapper(self, *args, **kwargs):
            instance = getattr(self, cache_key)
            if instance is not None:
                return instance

            instance = fn(self, *args, **kwargs)
            setattr(self, cache_key, instance)
            return instance

        return wrapper

    return inner_fn


# loss fn


def loss_fn(x, y):
    x = F.normalize(x, dim=-1, p=2)
    y = F.normalize(y, dim=-1, p=2)
    return 2 - 2 * (x * y).sum(dim=-1)


# augmentation utils


class RandomApply(nn.Module):
    def __init__(self, fn, p):
        super().__init__()
        self.fn = fn
        self.p = p

    def forward(self, x):
        if random.random() > self.p:
            return x
        return self.fn(x)


# exponential moving average


class EMA:
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new


def update_moving_average(ema_updater, ma_model, current_model):
    for current_params, ma_params in zip(
        current_model.parameters(), ma_model.parameters()
    ):
        old_weight, up_weight = ma_params.data, current_params.data
        ma_params.data = ema_updater.update_average(old_weight, up_weight)


# MLP class for projector and predictor


class MLP(nn.Module):
    def __init__(self, dim, projection_size, hidden_size=4096):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, projection_size),
        )

    def forward(self, x):
        return self.net(x)


# a wrapper class for the base neural network
# will manage the interception of the hidden layer output
# and pipe it into the projecter and predictor nets


class NetWrapper(nn.Module):
    def __init__(self, net, projection_size, projection_hidden_size, layer=-2):
        super().__init__()
        self.net = net
        self.layer = layer

        self.projector = None
        self.projection_size = projection_size
        self.projection_hidden_size = projection_hidden_size

        self.hidden = None
        self.hook_registered = False

    def _find_layer(self):
        if type(self.layer) == str:
            modules = dict([*self.net.named_modules()])
            return modules.get(self.layer, None)
        elif type(self.layer) == int:
            children = [*self.net.children()]
            return children[self.layer]
        return None

    def _hook(self, _, __, output):
        self.hidden = flatten(output)

    def _register_hook(self):
        layer = self._find_layer()
        assert layer is not None, f"hidden layer ({self.layer}) not found"
        handle = layer.register_forward_hook(self._hook)
        self.hook_registered = True

    @singleton("projector")
    def _get_projector(self, hidden):
        _, dim = hidden.shape
        projector = MLP(dim, self.projection_size, self.projection_hidden_size)
        return projector.to(hidden)

    def get_representation(self, x):
        if not self.hook_registered:
            self._register_hook()

        if self.layer == -1:
            return self.net(x)

        _ = self.net(x)
        hidden = self.hidden
        self.hidden = None
        assert hidden is not None, f"hidden layer {self.layer} never emitted an output"
        return hidden

    def forward(self, x):
        representation = self.get_representation(x)
        projector = self._get_projector(representation)
        projection = projector(representation)
        return projection


# main class


class BYOL(nn.Module):
    def __init__(
        self,
        net,
        image_size,
        hidden_layer=-2,
        projection_size=256,
        projection_hidden_size=4096,
        augment_fn=None,
        moving_average_decay=0.99,
        loss_type="byol_loss",
    ):
        super().__init__()

        self.online_encoder = NetWrapper(
            net, projection_size, projection_hidden_size, layer=hidden_layer
        )
        self.target_encoder = None
        self.target_ema_updater = EMA(moving_average_decay)

        self.online_predictor = MLP(
            projection_size, projection_size, projection_hidden_size
        )
        self.loss_type = "byol_loss"

        # send a mock image tensor to instantiate singleton parameters
        self.forward(torch.randn(2, 3, image_size, image_size), torch.randn(2, 3, image_size, image_size), self.loss_type)

    @singleton("target_encoder")
    def _get_target_encoder(self):
        target_encoder = copy.deepcopy(self.online_encoder)
        return target_encoder

    def reset_moving_average(self):
        del self.target_encoder
        self.target_encoder = None

    def update_moving_average(self):
        assert (
            self.target_encoder is not None
        ), "target encoder has not been created yet"
        update_moving_average(
            self.target_ema_updater, self.target_encoder, self.online_encoder
        )

    def forward(self, image_one, image_two, loss_type):
        # BYOL loss
        online_proj_one = self.online_encoder(image_one)
        online_proj_two = self.online_encoder(image_two)
        online_pred_one = self.online_predictor(online_proj_one)
        online_pred_two = self.online_predictor(online_proj_two)

        with torch.no_grad():
            target_encoder = self._get_target_encoder()
            target_proj_one = target_encoder(image_one)
            target_proj_two = target_encoder(image_two)

        loss_one = loss_fn(online_pred_one, target_proj_two.detach())
        loss_two = loss_fn(online_pred_two, target_proj_one.detach())
        byol_loss = (loss_one + loss_two).mean()

        if loss_type == "dimcl_loss":
            # DimCL loss
            z_a = self.online_encoder.get_representation(image_one)  # (N, D)
            z_b = self.online_encoder.get_representation(image_two)
            
            z_a = F.normalize(z_a, p=2, dim=0)  # Нормализация по батчу
            z_b = F.normalize(z_b, p=2, dim=0)
            
            D = z_a.size(1)
            z_a_t = z_a.T  # (D, N)
            z_b_t = z_b.T  # (D, N)
            all_keys = torch.cat([z_a_t, z_b_t], dim=0)  # (2D, N)
            
            similarity = torch.mm(z_a_t, all_keys.T) / 0.1  # Температура τ=0.1
            labels = torch.arange(D, 2 * D, device=z_a.device)
            
            dimcl_loss = F.cross_entropy(similarity, labels)
            total_loss = 0.9 * byol_loss + 0.1 * dimcl_loss  # λ=0.1
            
            return total_loss, byol_loss 
        
        return byol_loss


class NetWrapperViT(nn.Module):
    def __init__(self, net, projection_size, projection_hidden_size, layer=-2):
        super().__init__()
        self.net = net
        self.layer = layer  # Not used for ViT, but keep for compatibility

        self.projector = None
        self.projection_size = projection_size
        self.projection_hidden_size = projection_hidden_size

        self.hidden = None  # Not strictly needed, but good for consistency
        # self.hook_registered = False # Not needed, simplified logic

    # No _find_layer, _hook, _register_hook needed for ViT

    @singleton("projector")
    def _get_projector(self, hidden):
        _, dim = hidden.shape
        projector = MLP(dim, self.projection_size, self.projection_hidden_size)
        return projector.to(hidden)

    def get_representation(self, x):
        # Adapted from the example project's ByolNet
        x = self.net._process_input(x)
        n = x.shape[0]

        # Expand the class token to the full batch
        batch_class_token = self.net.class_token.expand(n, -1, -1)
        x = torch.cat([batch_class_token, x], dim=1)

        x = self.net.encoder(x)

        # Classifier "token" as used by standard language architectures
        x = x[:, 0]
        return x

    def forward(self, x):
        representation = self.get_representation(x)
        projector = self._get_projector(representation)
        projection = projector(representation)
        return projection



class BYOLViT(nn.Module):
    def __init__(
        self,
        net,
        image_size,
        hidden_layer=-2, 
        projection_size=256,
        projection_hidden_size=4096,
        augment_fn=None,
        moving_average_decay=0.99,
        pretrained=True,
        loss_type="byol_loss"
    ):
        super().__init__()
        
        self.pretrained = pretrained
        if self.pretrained:
            self.online_encoder = NetWrapperViT(
                net, projection_size, projection_hidden_size, layer=hidden_layer
             )
        else:
            raise NotImplementedError("Non-pretrained ViT not implemented")

        self.target_encoder = None
        self.target_ema_updater = EMA(moving_average_decay)

        self.online_predictor = MLP(
            projection_size, projection_size, projection_hidden_size
        )
        
        # Get dimension
        with torch.no_grad():
          mock_input = torch.randn(2, 3, image_size, image_size)
          rep = self.online_encoder.get_representation(mock_input)
          in_features = rep.shape[-1]

        self.loss_type = "byol_loss"
        # send a mock image tensor to instantiate singleton parameters
        self.forward(torch.randn(2, 3, image_size, image_size), torch.randn(2, 3, image_size, image_size), self.loss_type)

    @singleton("target_encoder")
    def _get_target_encoder(self):
        target_encoder = copy.deepcopy(self.online_encoder)
        return target_encoder

    def reset_moving_average(self):
        del self.target_encoder
        self.target_encoder = None

    def update_moving_average(self):
        assert (
            self.target_encoder is not None
        ), "target encoder has not been created yet"
        update_moving_average(
            self.target_ema_updater, self.target_encoder, self.online_encoder
        )

    def forward(self, image_one, image_two, loss_type):
        # BYOL loss
        online_proj_one = self.online_encoder(image_one)
        online_proj_two = self.online_encoder(image_two)
        online_pred_one = self.online_predictor(online_proj_one)
        online_pred_two = self.online_predictor(online_proj_two)

        with torch.no_grad():
            target_encoder = self._get_target_encoder()
            target_proj_one = target_encoder(image_one)
            target_proj_two = target_encoder(image_two)

        loss_one = loss_fn(online_pred_one, target_proj_two.detach())
        loss_two = loss_fn(online_pred_two, target_proj_one.detach())
        byol_loss = (loss_one + loss_two).mean()

        if loss_type == "dimcl_loss":
            # DimCL loss
            z_a = self.online_encoder.get_representation(image_one)  # (N, D)
            z_b = self.online_encoder.get_representation(image_two)
            
            z_a = F.normalize(z_a, p=2, dim=0)  
            z_b = F.normalize(z_b, p=2, dim=0)
            
            D = z_a.size(1)
            z_a_t = z_a.T  # (D, N)
            z_b_t = z_b.T  # (D, N)
            all_keys = torch.cat([z_a_t, z_b_t], dim=0)  # (2D, N)
            
            similarity = torch.mm(z_a_t, all_keys.T) / 0.1 
            labels = torch.arange(D, 2 * D, device=z_a.device)
            
            dimcl_loss = F.cross_entropy(similarity, labels)
            total_loss = 0.9 * byol_loss + 0.1 * dimcl_loss 
            return total_loss, byol_loss 
        
        return byol_loss
