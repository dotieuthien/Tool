import os
import glob
import torch
import torch.utils.data
import torch.optim as optim
import torch.nn.functional
import neptune
from natsort import natsorted

from loader.self_augment_loader import PairAnimeDataset, loader_collate
from models.residual_models.pyramid_net import UNet
from loss.triplet import compute_triplet_three_branches, compute_triplet_loss
from loss.infoNCE import compute_infoNCE_loss
from training.tools import AverageMeter, get_full_class_name


def load_existing_checkpoint(model, optimizer, config):
    weight_paths = natsorted(glob.glob(os.path.join(config.spot_checkpoint, "*.pth")))

    if len(weight_paths) == 0:
        return 0

    weight_path = weight_paths[-1]
    weight_data = torch.load(weight_path)

    model.load_state_dict(weight_data["model"])
    optimizer.load_state_dict(weight_data["optimizer"])

    epoch = int(os.path.basename(weight_path)[6:9]) + 1
    return epoch


def save_checkpoint(model, optimizer, config, epoch):
    state = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    path = os.path.join(config.spot_checkpoint, "model_%03d.pth" % epoch)
    torch.save(state, path)


def train(config):
    device = config.device
    get_z = False
    image_size = (768, 512)
    dropout = 0.0
    base_weight_path = None
    unsupervised_loss = True
    use_color = True

    # Dataset
    dataset = PairAnimeDataset(config.data_folder, image_size, config)
    dataset_name = get_full_class_name(dataset)
    print("Dataset size:", len(dataset))
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=config.batch_size, shuffle=True, collate_fn=loader_collate)

    # Model
    model = UNet(base_weight_path, dropout)
    model.to(device)
    model_name = get_full_class_name(model)

    # Optimization
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, config.step_size, gamma=0.1)

    # Load the checkpoint
    start_epoch = load_existing_checkpoint(model, optimizer, config)

    # neptune.ai init
    neptune.init(
        project_qualified_name="cinnamon/cobra-team",
        api_token=config.neptune_token)

    params = {
        "model_name": model_name,
        "dataset_name": dataset_name,
        "get_z": get_z,
        "image_size": image_size,
        "learning_rate": config.learning_rate,
        "dropout": dropout,
        "unsupervised_loss": unsupervised_loss,
        "use_color": use_color,
    }

    neptune.create_experiment(
        name="colorization-ucn",
        params=params,
        upload_source_files=["*/*.py", "requirements.txt"])
    neptune.append_tags(["colorization", "hades", "component-based"])

    # Training loop
    for epoch in range(start_epoch, config.num_epochs):
        print("===============")
        print("Epoch:", epoch)
        model.train()
        bug_count = 0
        mean_loss = AverageMeter()

        for step, data in enumerate(loader):
            print(step)
            list_a, list_b, positive_pairs = data
            sketch_a, mask_a, graph_a, kpts_a, colors_a = list_a
            sketch_b, mask_b, graph_b, kpts_b, colors_b = list_b

            sketch_a = sketch_a.to(device)
            mask_a = mask_a.to(device)
            graph_a = graph_a.to(device)
            sketch_b = sketch_b.to(device)
            mask_b = mask_b.to(device)
            graph_b = graph_b.to(device)
            positive_pairs = positive_pairs.to(device)

            # check for bug
            if positive_pairs.shape[1] == 0:
                bug_count += 1
                continue

            # run the model
            if get_z:
                output_a, x_a, y_a = model(sketch_a, mask_a)
                output_b, x_b, y_b = model(sketch_b, mask_b)

                loss, loss_out, loss_x, loss_y = compute_triplet_three_branches(
                    output_a, output_b, x_a, x_b, y_a, y_b,
                    positive_pairs, colors_a, colors_b)
                
                if unsupervised_loss:
                    loss_infoNCE = compute_infoNCE_loss(output_a, output_b, colors_a, colors_b, use_color=use_color)
                    loss = loss + loss_infoNCE
                mean_loss.update(loss.item())

                if step % config.print_freq == 0:
                    print("Loss: %.4f    X: %.4f    Y: %.4f" % (loss_out.item(), loss_x.item(), loss_y.item()))

            else:
                output_a = model(sketch_a, mask_a)
                output_b = model(sketch_b, mask_b)
                loss = compute_triplet_loss(output_a, output_b, positive_pairs, colors_a, colors_b)
                if unsupervised_loss:
                    loss_infoNCE = compute_infoNCE_loss(output_a, output_b, colors_a, colors_b, use_color=use_color)
                    loss = loss + loss_infoNCE
                mean_loss.update(loss.item())

                if step % config.print_freq == 0:
                    print("Loss: %.4f" % (loss.item()))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if epoch % config.save_freq == 0:
            save_checkpoint(model, optimizer, config, epoch)
        scheduler.step()

        neptune.log_metric("train/loss", mean_loss.get(), timestamp=epoch)
        neptune.log_metric("train/bug_count", bug_count, timestamp=epoch)

    print("Finished")


def main():
    from loader.config_utils import load_config, convert_to_object

    config_path = "/home/hades/UCN/gcp_toei_config.ini"

    main_config = convert_to_object(load_config(config_path))
    main_config.device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    print('Device ', main_config.device)
    main_config.print_freq = 10
    train(main_config)


if __name__ == "__main__":
    main()
