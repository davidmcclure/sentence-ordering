

import click

from sorder.models import pick_next as model


@click.group()
def cli():
    pass


@cli.command()
@click.argument('train_path', type=click.Path())
@click.argument('model_path', type=click.Path())
@click.option('--train_skim', type=int, default=10000)
@click.option('--lr', type=float, default=1e-3)
@click.option('--epochs', type=int, default=1000)
@click.option('--epoch_size', type=int, default=100)
@click.option('--batch_size', type=int, default=20)
@click.option('--lstm_dim', type=int, default=500)
@click.option('--lin_dim', type=int, default=500)
def train(*args, **kwargs):
    model.train(*args, **kwargs)


if __name__ == '__main__':
    cli()
