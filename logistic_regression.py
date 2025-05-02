import os
import argparse
import pickle
from collections import defaultdict
import numpy as np
import torch
import torch.nn as nn
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import torch.nn.functional as F
from sklearn.manifold import TSNE
from modules.transformations import TransformsSimCLR
from process_features import get_features, create_data_loaders_from_arrays

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True, type=str, help="Path to pre-trained backbone model")
    parser.add_argument(
        "--model_type",
        required=True,
        type=str,
        choices=['resnet18', 'vit'],
        help="Type of backbone model: resnet18/vit",
    )
    parser.add_argument(
        "--dataset",
        required=True,
        type=str,
        choices=['cifar10', 'cifar100'],
        help="Dataset to use: cifar10/cifar100",
    )
    parser.add_argument("--image_size", default=224, type=int, help="Image size")
    parser.add_argument(
        "--learning_rate", default=3e-3, type=float, help="Initial learning rate."
    )
    parser.add_argument(
        "--batch_size", default=768, type=int, help="Batch size for feature extraction."
    )
    parser.add_argument(
        "--logreg_batch_size", default=2048, type=int, help="Batch size for linear classifier training."
    )
    parser.add_argument(
        "--num_epochs", default=300, type=int, help="Number of epochs to train linear classifier."
    )
    parser.add_argument(
        "--dataset_dir",
        default="./datasets",
        type=str,
        help="Directory where dataset is stored.",
    )
    parser.add_argument(
        "--num_workers",
        default=8,
        type=int,
        help="Number of data loading workers",
    )
    args = parser.parse_args()

    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

    if args.dataset == "cifar10":
        train_dataset_fn = datasets.CIFAR10
        test_dataset_fn = datasets.CIFAR10
        n_classes = 10
        tsne_n_samples = 5000
        tsne_cmap = 'tab10'
        tsne_perplexity = 30
        tsne_n_iter = 1000
        tsne_title = 't-SNE Visualization of Test Data Representations (CIFAR-10)'
        tsne_use_legend = True

    elif args.dataset == "cifar100":
        train_dataset_fn = datasets.CIFAR100
        test_dataset_fn = datasets.CIFAR100
        n_classes = 100
        tsne_n_samples = 10000 
        tsne_cmap = 'viridis' 
        tsne_perplexity = 50 
        tsne_n_iter = 1000
        tsne_title = 't-SNE Visualization of Test Data Representations (CIFAR-100)'
        tsne_use_legend = False
    else:
        raise NotImplementedError(f"Dataset {args.dataset} not supported.")


    train_transform = TransformsSimCLR(size=args.image_size).test_transform
    test_transform = TransformsSimCLR(size=args.image_size).test_transform

    train_dataset = train_dataset_fn(
        args.dataset_dir,
        train=True,
        download=True,
        transform=train_transform,
    )

    test_dataset = test_dataset_fn(
        args.dataset_dir,
        train=False,
        download=True,
        transform=test_transform,
    )

    train_loader_extract = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=False, 
        drop_last=False, 
        num_workers=args.num_workers,
    )

    test_loader_extract = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=args.num_workers,
    )

    if args.model_type == "resnet18":
        backbone = models.resnet18(weights=None)
        num_features = backbone.fc.in_features
        backbone.fc = nn.Identity() # remove final layer
    elif args.model_type == "vit":
        backbone = models.vit_b_16(weights=None) 
        num_features = backbone.heads.head.in_features
        backbone.heads.head = nn.Identity() # remove final classification head
    else:
        raise NotImplementedError(f"Model type {args.model_type} not implemented")

    backbone.load_state_dict(torch.load(args.model_path, map_location='cpu'), strict=False)
    backbone = backbone.to(device)
    backbone.eval()

    logreg = nn.Linear(num_features, n_classes)
    logreg = logreg.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(params=logreg.parameters(), lr=args.learning_rate)

    feature_cache_file = f"features_{args.model_type}_{args.dataset}.p"

    if not os.path.exists(feature_cache_file):
        print(f"### Creating features from pre-trained {args.model_type} on {args.dataset} ###")
        (train_X, train_y, test_X, test_y) = get_features(
            backbone, train_loader_extract, test_loader_extract, device
        )
        print(f"### Saving features to {feature_cache_file} ###")
        with open(feature_cache_file, "wb") as f:
             pickle.dump((train_X, train_y, test_X, test_y), f, protocol=4)
    else:
        print(f"### Loading features from {feature_cache_file} ###")
        with open(feature_cache_file, "rb") as f:
            (train_X, train_y, test_X, test_y) = pickle.load(f)

    train_loader_logreg, test_loader_logreg = create_data_loaders_from_arrays(
        train_X, train_y, test_X, test_y, args.logreg_batch_size
    )

    print(f"### Training linear classifier on top of {args.model_type} features for {args.dataset} ###")
    for epoch in range(args.num_epochs):
        metrics = defaultdict(list)
        logreg.train()
        for step, (h, y) in enumerate(train_loader_logreg):
            h = h.to(device)
            y = y.to(device)

            outputs = logreg(h)

            loss = criterion(outputs, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            accuracy = (outputs.argmax(1) == y).sum().item() / y.size(0)
            metrics["Loss/train"].append(loss.item())
            metrics["Accuracy/train"].append(accuracy)

        print(f"Epoch [{epoch+1}/{args.num_epochs}]: " + "\t".join([f"{k}: {np.array(v).mean():.4f}" for k, v in metrics.items()]))

    print("### Calculating final testing performance ###")
    metrics = defaultdict(list)
    logreg.eval()
    with torch.no_grad():
        for step, (h, y) in enumerate(test_loader_logreg):
            h = h.to(device)
            y = y.to(device)

            outputs = logreg(h)

            accuracy = (outputs.argmax(1) == y).sum().item() / y.size(0)
            metrics["Accuracy/test"].append(accuracy)

    print(f"Final test performance: " + "\t".join([f"{k}: {np.array(v).mean():.4f}" for k, v in metrics.items()]))

    print("Generating t-SNE visualization...")
    tsne_output_file = f"test_tsne_{args.model_type}_{args.dataset}.png"

    indices = np.random.choice(len(test_X), tsne_n_samples, replace=False)
    test_features_subset = test_X[indices]
    test_labels_subset = test_y[indices]

    tsne = TSNE(n_components=2, random_state=42, perplexity=tsne_perplexity, n_iter=tsne_n_iter, init='pca', learning_rate='auto')
    embedded_features = tsne.fit_transform(test_features_subset)

    plt.figure(figsize=(12, 10))
    scatter = plt.scatter(embedded_features[:, 0], embedded_features[:, 1], c=test_labels_subset, cmap=tsne_cmap, alpha=0.6, s=10)

    if tsne_use_legend:
        plt.legend(*scatter.legend_elements(), title='Classes', bbox_to_anchor=(1.05, 1), loc='upper left')
    else:
        plt.colorbar(label='Class Label')

    plt.title(tsne_title)
    plt.xlabel("t-SNE Component 1")
    plt.ylabel("t-SNE Component 2")
    plt.tight_layout()
    plt.savefig(tsne_output_file)
    print(f"t-SNE plot saved to {tsne_output_file}")
    plt.close()
