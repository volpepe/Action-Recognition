import torch
import sys
import numpy as np
import itertools
from models import *
from dataset import *
from torch.utils.data import DataLoader
from torch.autograd import Variable
import argparse
import time
import datetime

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_train_path", type=str, default='/home/datasets/KineticsImgTest/dataset/train_mult', help="Path to train dataset")
    parser.add_argument("--dataset_val_path", type=str, default='/home/datasets/KineticsImgTest/dataset/val_mult', help="Path to validation dataset")
    parser.add_argument('--saved_model_path', type=str, default='/home/datasets/KineticsImgTest/results')
    parser.add_argument("--num_epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=64, help="Size of each training batch")
    parser.add_argument("--sequence_length", type=int, default=1, help="Number of frames in each sequence")
    parser.add_argument("--latent_dim", type=int, default=512, help="Dimensionality of the latent representation")
    parser.add_argument('--num_train_threshold', type=int, default=8)
    parser.add_argument("--checkpoint_interval", type=int, default=5, help="Interval between saving model checkpoints")
    parser.add_argument("--checkpoint_model", type=str, default="", help="Optional path to checkpoint model")
    parser.add_argument("--test_dataset", action="store_true", default=False, help="Only load the dataset")
    opt = parser.parse_args()
    print(opt)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Define training set
    train_dataset = Dataset(
        dataset_path=opt.dataset_train_path,
        sequence_length=opt.sequence_length,
        training=True,
    )
    train_dataloader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True, num_workers=16)

    # Define test set
    test_dataset = Dataset(
        dataset_path=opt.dataset_val_path,
        sequence_length=opt.sequence_length,
        training=False,
    )
    test_dataloader = DataLoader(test_dataset, batch_size=opt.batch_size, shuffle=False, num_workers=16)

    if opt.test_dataset:
        print("Dataset tests:")
        for i in range(20):
            (X, y) = train_dataset[i]
            (X_2, y_2) = test_dataset[i]
            print("Train batch {}: {}, {}".format(str(i), str(X), str(y)))
            print("Val batch {}: {}, {}".format(str(i), str(X_2), str(y_2)))
        sys.exit()

    # Classification criterion
    cls_criterion = nn.CrossEntropyLoss().to(device)

    # Define network
    model = ConvLSTM(
        num_classes=train_dataset.num_classes,
        latent_dim=opt.latent_dim,
        lstm_layers=1,
        hidden_dim=1024,
        bidirectional=True,
        attention=True,
    )

    if torch.cuda.device_count() > 1:
        print("Using", torch.cuda.device_count(), "GPUs!")
        model = ParallelConvLSTM(model)

    model = model.to(device)

    # Add weights from checkpoint model if specified
    if opt.checkpoint_model:
        model.load_state_dict(torch.load(opt.checkpoint_model))

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

    def test_model(epoch):
        """ Evaluate the model on the test set """
        print("")
        model.eval()
        test_metrics = {"loss": [], "acc": []}
        for batch_i, (X, y) in enumerate(test_dataloader):
            image_sequences = Variable(X.to(device), requires_grad=False)
            labels = Variable(y, requires_grad=False).to(device)
            with torch.no_grad():
                # Reset LSTM hidden state
                model.lstm.reset_hidden_state()
                # Get sequence predictions
                predictions = model(image_sequences)
            # Compute metrics
            acc = 100 * (predictions.detach().argmax(1) == labels).cpu().numpy().mean()
            loss = cls_criterion(predictions, labels).item()
            # Keep track of loss and accuracy
            test_metrics["loss"].append(loss)
            test_metrics["acc"].append(acc)
            # Log test performance
            sys.stdout.write(
                "\rTesting -- [Batch %d/%d] [Loss: %f (%f), Acc: %.2f%% (%.2f%%)]"
                % (
                    batch_i,
                    len(test_dataloader),
                    loss,
                    np.mean(test_metrics["loss"]),
                    acc,
                    np.mean(test_metrics["acc"]),
                )
            )
        model.train()
        print("")
        return acc

    starting_time = time.time()

    #training procedure
    for epoch in range(opt.num_epochs):
        best_accuracy = 0
        keep_training_value = 0
        epoch_metrics = {"loss": [], "acc": []}
        prev_time = time.time()
        print("--- Epoch {} ---".format(str(epoch)))
        for batch_i, (X, y) in enumerate(train_dataloader):

            if X.size(0) == 1:
                continue

            image_sequences = Variable(X.to(device), requires_grad=True)
            labels = Variable(y.to(device), requires_grad=False)

            optimizer.zero_grad()

            # Reset LSTM hidden state
            model.lstm.reset_hidden_state()

            # Get sequence predictions
            predictions = model(image_sequences)

            # Compute metrics
            loss = cls_criterion(predictions, labels)
            acc = 100 * (predictions.detach().argmax(1) == labels).cpu().numpy().mean()

            loss.backward()
            optimizer.step()

            # Keep track of epoch metrics
            epoch_metrics["loss"].append(loss.item())
            epoch_metrics["acc"].append(acc)

            # Determine approximate time left
            batches_done = epoch * len(train_dataloader) + batch_i
            batches_left = opt.num_epochs * len(train_dataloader) - batches_done
            time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time))
            prev_time = time.time()

            # Print log
            sys.stdout.write(
                "\r[Epoch %d/%d] [Batch %d/%d] [Loss: %f (%f), Acc: %.2f%% (%.2f%%)] ETA: %s"
                % (
                    epoch,
                    opt.num_epochs,
                    batch_i,
                    len(train_dataloader),
                    loss.item(),
                    np.mean(epoch_metrics["loss"]),
                    acc,
                    np.mean(epoch_metrics["acc"]),
                    time_left,
                )
            )

            # Empty cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # Evaluate the model on the test set
        validation_accuracy = test_model(epoch)

        # Save model checkpoint
        if epoch % opt.checkpoint_interval == 0:
            os.makedirs(os.path.join(opt.saved_model_path, "model_checkpoints"), exist_ok=True)
            torch.save(model.state_dict(), "model_checkpoints/{}_{}.pth".format(model.__class__.__name__, str(epoch)))

        if validation_accuracy < best_accuracy:
            keep_training_value += 1
            print("Accuracy has gotten worse... best result was {} epochs ago".format(str(keep_training_value)))
            if keep_training_value >= opt.num_train_threshold:
                print("Interrupting training at epoch {} for lack of improvements".format(str(epoch)))
                break
        else:
            keep_training_value = 0
            best_accuracy = validation_accuracy
            print("Accuracy is better than last epoch.")

    print("Training ended in {}".format(str(time.time() - starting_time)))