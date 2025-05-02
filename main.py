import os
import argparse
import torch
from torch.utils.tensorboard import SummaryWriter
from torchvision import models, datasets
import numpy as np
from collections import defaultdict

from modules.byol import BYOL, BYOLViT
from modules.transformations import TransformsSimCLR

# distributed training
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm


def cleanup():
    dist.destroy_process_group()


def main(gpu, args):
    rank = args.nr * args.gpus + gpu
    dist.init_process_group("nccl", rank=rank, world_size=args.world_size)

    torch.manual_seed(0)
    torch.cuda.set_device(gpu)

    # dataset
    if args.dataset == "cifar10":
        train_dataset = datasets.CIFAR10(
            args.dataset_dir,
            download=True,
            transform=TransformsSimCLR(size=args.image_size), # paper 224
        )
    else:
        train_dataset = datasets.CIFAR100(
            args.dataset_dir,
            download=True,
            transform=TransformsSimCLR(size=args.image_size), # paper 224
        )

    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset, num_replicas=args.world_size, rank=rank
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        drop_last=True,
        num_workers=args.num_workers,
        pin_memory=True,
        sampler=train_sampler,
    )

    # model
    if args.model_type == "resnet18":
        resnet = models.resnet18(pretrained=False)
        model = BYOL(resnet, image_size=args.image_size, hidden_layer="avgpool", loss_type=args.loss_type)
    else:
        resnet = models.vision_transformer.vit_b_16(image_size=args.image_size, weights='DEFAULT')
        model = BYOLViT(resnet, image_size=args.image_size, hidden_layer="avgpool", loss_type=args.loss_type)

    model = model.cuda(gpu)

    # distributed data parallel
    model = DDP(model, device_ids=[gpu], find_unused_parameters=True)

    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    # TensorBoard writer

    if gpu == 0:
        writer = SummaryWriter()

    if args.loss_type =="dimcl_loss":
        # solver
        global_step = 0
        for epoch in range(args.num_epochs):
            metrics = defaultdict(list)
            total_loss, total_byol_loss, n = 0, 0, 0
            scaler = torch.amp.GradScaler("cuda")
            for step, ((x_i, x_j), _) in tqdm(enumerate(train_loader)):
                x_i = x_i.cuda(non_blocking=True)
                x_j = x_j.cuda(non_blocking=True)

                with torch.amp.autocast("cuda"):
                    loss, byol_loss = model(x_i, x_j, "dimcl_loss")
                    # loss = model(x_i, x_j)
                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                model.module.update_moving_average()  # update moving average of target encoder
                
                n += 1
                total_loss += loss.item()
                total_byol_loss += byol_loss
                # if step % 1 == 0 and gpu == 0:
                #     print(f"Step [{step}/{len(train_loader)}]:\tLoss: {loss.item()} BYOL_Loss: {byol_loss}")
                if gpu == 0:
                    writer.add_scalar("Loss/train_step", byol_loss.item(), global_step)
                    metrics["Loss/train"].append(byol_loss.item())
                    # writer.add_scalar("Loss/train_step", loss.item(), global_step)
                    # metrics["Loss/train"].append(loss.item())
                    global_step += 1
            print(f"Epoch {epoch}:\tLoss: {total_loss / n} BYOL_Loss: {total_byol_loss / n}")

            
            if gpu == 0:
                # write metrics to TensorBoard
                for k, v in metrics.items():
                    writer.add_scalar(k, np.array(v).mean(), epoch)

                if epoch % args.checkpoint_epochs == 0:
                    if gpu == 0:
                        print(f"Saving model at epoch {epoch}")
                        torch.save(resnet.state_dict(), f"./model-{epoch}.pt")

                    # let other workers wait until model is finished
                    # dist.barrier()
    else:
                # solver
        global_step = 0
        for epoch in range(args.num_epochs):
            metrics = defaultdict(list)
            total_loss, n = 0, 0
            scaler = torch.amp.GradScaler("cuda")
            for step, ((x_i, x_j), _) in tqdm(enumerate(train_loader)):
                x_i = x_i.cuda(non_blocking=True)
                x_j = x_j.cuda(non_blocking=True)

                with torch.amp.autocast("cuda"):
                    # loss, byol_loss = model(x_i, x_j)
                    loss = model(x_i, x_j, "byol_loss")
                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                model.module.update_moving_average()  # update moving average of target encoder
                
                n += 1
                total_loss += loss.item()
                # total_byol_loss += byol_loss
                # if step % 1 == 0 and gpu == 0:
                #     print(f"Step [{step}/{len(train_loader)}]:\tLoss: {loss.item()} BYOL_Loss: {byol_loss}")
                if gpu == 0:
                    # writer.add_scalar("Loss/train_step", byol_loss.item(), global_step)
                    # metrics["Loss/train"].append(byol_loss.item())
                    writer.add_scalar("Loss/train_step", loss.item(), global_step)
                    metrics["Loss/train"].append(loss.item())
                    global_step += 1
            print(f"Epoch {epoch}:\tLoss: {total_loss / n}")

            
            if gpu == 0:
                # write metrics to TensorBoard
                for k, v in metrics.items():
                    writer.add_scalar(k, np.array(v).mean(), epoch)

                if epoch % args.checkpoint_epochs == 0:
                    if gpu == 0:
                        print(f"Saving model at epoch {epoch}")
                        torch.save(resnet.state_dict(), f"./model-{epoch}.pt")

                    # let other workers wait until model is finished
                    # dist.barrier()

    # save your improved network
    if gpu == 0:
        torch.save(resnet.state_dict(), "./model-final.pt")

    cleanup()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_size", default=224, type=int, help="Image size")
    parser.add_argument(
        "--learning_rate", default=3e-4, type=float, help="Initial learning rate."
    )
    parser.add_argument(
        "--batch_size", default=64, type=int, help="Batch size for training."
    )
    parser.add_argument(
        "--num_epochs", default=100, type=int, help="Number of epochs to train for."
    )
    parser.add_argument(
        "--resnet_version", default="resnet18", type=str, help="ResNet version."
    )
    parser.add_argument(
        "--checkpoint_epochs",
        default=5,
        type=int,
        help="Number of epochs between checkpoints/summaries.",
    )
    parser.add_argument(
        "--loss_type",
        default="byol_loss",
        type=str,
        help="byol_loss/dimcl_loss",
    )
    parser.add_argument(
        "--model_type",
        default="resnet18",
        type=str,
        help="resnet18/vit",
    )
    parser.add_argument(
        "--dataset",
        default="cifar10",
        type=str,
        help="cifar10/cifar100",
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
        help="Number of data loading workers (caution with nodes!)",
    )
    parser.add_argument(
        "--nodes", default=1, type=int, help="Number of nodes",
    )
    parser.add_argument("--gpus", default=1, type=int, help="number of gpus per node")
    parser.add_argument("--nr", default=0, type=int, help="ranking within the nodes")
    args = parser.parse_args()

    # Master address for distributed data parallel
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "8010"
    args.world_size = args.gpus * args.nodes

    # Initialize the process and join up with the other processes.
    # This is “blocking,” meaning that no process will continue until all processes have joined.
    mp.spawn(main, args=(args,), nprocs=args.gpus, join=True)
