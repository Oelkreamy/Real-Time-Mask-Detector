import os
import argparse
import yaml
import torch
from tqdm import trange
from tqdm import tqdm
from model.models.detection_model import DetectionModel
from model.data.dataset import Dataset

from torch.utils.data import DataLoader


def get_args():
    parser = argparse.ArgumentParser(description='YOLOv8 model training')
    parser.add_argument(
        '--model-config',
        type=str,
        default='model/config/models/yolov8n.yaml',
        help='path to model config file'
    )
    parser.add_argument(
        '--weights',
        type=str,
        help='path to weights file'
    )

    parser.add_argument(
        '--train-config',
        type=str,
        default='model/config/training/fine_tune.yaml',
        help='path to training config file'
    )

    dataset_args = parser.add_argument_group('Dataset')
    dataset_args.add_argument(
        '--dataset',
        type=str,
        required=True,
        help='path to dataset config file'
    )

    dataset_args.add_argument(
        '--dataset-mode',
        type=str,
        default='train',
        help='dataset mode'
    )

    parser.add_argument(
        '--device',
        '-d',
        type=str,
        default='cuda',
        help='device to model on'
    )

    parser.add_argument(
        '--save',
        '-s',
        action='store_true',
        help='save trained model weights'
    )

    return parser.parse_args()


def main(args):
    train_config = yaml.safe_load(open(args.train_config, 'r', encoding='utf-8'))


    device = torch.device(args.device)
    model = DetectionModel(args.model_config, device=device)
    if args.weights is not None:
        state_dict = torch.load(args.weights)
        missing = model.load(state_dict, strict=False)
        print("Loaded weights with non-strict mode.")
        print("Missing/unexpected keys:", missing)

    # Freeze backbone parameters
    if hasattr(model, 'backbone'):
        for param in model.backbone.parameters():
            param.requires_grad = False
        print("Backbone frozen. Only head will be trained.")
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=train_config['lr'])
    else:
        print("Warning: model has no attribute 'backbone'. Training all parameters.")
        optimizer = torch.optim.Adam(model.parameters(), lr=train_config['lr'])

    dataset = Dataset(args.dataset, mode=args.dataset_mode, batch_size=train_config['batch_size'])
    dataloader = DataLoader(dataset, batch_size=dataset.batch_size, shuffle=True, collate_fn=Dataset.collate_fn)

    if args.save:
        save_path = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                 train_config['save_dir'],
                                 os.path.splitext(os.path.basename(args.model_config))[0])
        os.makedirs(save_path, exist_ok=True)

    best_loss = float('inf')
    best_model_path = None
    for epoch in range(train_config['epochs']):
        epoch_loss = 0.0
        num_batches = len(dataloader)
        with tqdm(dataloader, desc=f"Epoch {epoch+1}/{train_config['epochs']}", unit="batch") as pbar:
            for batch in pbar:
                loss = model.loss(batch)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
        avg_loss = epoch_loss / num_batches
        print(f"Epoch [{epoch+1}/{train_config['epochs']}] completed. Average Loss: {avg_loss:.4f}")

        # Save best model (lowest avg_loss)
        if args.save and avg_loss < best_loss:
            best_loss = avg_loss
            best_model_path = os.path.join(save_path, 'best.pt')
            model.save(best_model_path)
            print(f"Best model updated and saved to {best_model_path}")

    # Save last model at the end
    if args.save:
        last_model_path = os.path.join(save_path, 'last.pt')
        model.save(last_model_path)
        print(f"Last model saved to {last_model_path}")


if __name__ == '__main__':
    args = get_args()
    main(args)
    