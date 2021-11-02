#!/usr/bin/env python

from argparse import ArgumentParser
from utils import get_mnist_data_loaders, DataLoaderProgress, NN_FC_CrossEntropy
from fastprogress.fastprogress import master_bar, progress_bar
import torch

# check


def train_one_epoch(dataloader, model, criterion, optimizer, device, mb):

    # Put the model into training mode
    model.train()

    # Loop over the data using the progress_bar utility
    for _, (X, Y) in progress_bar(DataLoaderProgress(dataloader), parent=mb):
        X, Y = X.to(device), Y.to(device)

        # Compute model output and then loss
        output = model(X)
        loss = criterion(output, Y)

        # TODO:
        # - zero-out gradients
        model.zero_grad()
        # - compute new gradients
        loss.backward()

        # Ask question
        learning_rate = 1e-2
        weight_decay = 1e-3
        momentum = 0.9

        """
        params = ()
        opmizer = torch.optim.SGD(params, lr=learning_rate)
        """

        # - update paramaters
        with torch.no_grad():
            for param, grad_ema in zip(model.parameters(), model.grad_emas):
                grad_ema.set_(momentum * grad_ema +
                              (1 - momentum) * param.grad)
                param -= learning_rate * grad_ema + weight_decay * param


def validate(dataloader, model, criterion, device, epoch, num_epochs, mb):

    # Put the model into validation/evaluation mode
    model.eval()

    N = len(dataloader.dataset)
    num_batches = len(dataloader)

    loss, num_correct = 0, 0

    # Tell pytorch to stop updating gradients when executing the following
    with torch.no_grad():

        for X, Y in dataloader:
            X, Y = X.to(device), Y.to(device)

            # Compute the model output
            output = model(X)

            # TODO:
            # - compute loss
            loss += criterion(output, Y).item()

            # - compute the number of correctly classified examples
            num_correct += (output.argmax(1) ==
                            Y).type(torch.float).sum().item()

        loss /= num_batches
        accuracy = num_correct / N

    message = "Initial" if epoch == 0 else f"Epoch {epoch:>2}/{num_epochs}:"
    message += f" accuracy={100*accuracy:5.2f}%"
    message += f" and loss={loss:.3f}"
    mb.write(message)


def train(model, criterion, optimizer, train_loader, valid_loader, device, num_epochs):

    mb = master_bar(range(num_epochs))

    validate(valid_loader, model, criterion, device, 0, num_epochs, mb)

    for epoch in mb:
        train_one_epoch(train_loader, model, criterion, optimizer, device, mb)
        validate(valid_loader, model, criterion,
                 device, epoch + 1, num_epochs, mb)


def main():

    aparser = ArgumentParser("Train a neural network on the MNIST dataset.")
    aparser.add_argument(
        "mnist", type=str, help="Path to store/find the MNIST dataset")
    aparser.add_argument("--num_epochs", type=int, default=10)
    aparser.add_argument("--batch_size", type=int, default=128)
    aparser.add_argument("--learning_rate", type=float, default=0.1)
    aparser.add_argument("--seed", action="store_true")
    aparser.add_argument("--gpu", action="store_true")

    args = aparser.parse_args()

    # Set the random number generator seed if one is provided
    if args.seed:
        torch.manual_seed(args.seed)

    # Use GPU if requested and available
    device = "cuda" if args.gpu and torch.cuda.is_available() else "cpu"

    # Get data loaders
    train_loader, valid_loader = get_mnist_data_loaders(
        args.mnist, args.batch_size, 0)

    # TODO: create a new model
    # Your model can be as complex or simple as you'd like. It must work
    # with the other parts of this script.
    nx = train_loader.dataset.data.shape[1:].numel()
    ny = len(train_loader.dataset.classes)
    layer_sizes = (nx, 10, 10, ny)
    model = NN_FC_CrossEntropy(layer_sizes).to(device)

    # TODO:
    # - create a CrossEntropyLoss criterion
    # - create an optimizer of your choice
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD

    train(
        model, criterion, optimizer, train_loader, valid_loader, device, args.num_epochs
    )


if __name__ == "__main__":
    main()
