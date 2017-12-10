

import click

from sorder.models import pairs


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
@click.option('--batch_size', type=int, default=10)
@click.option('--lstm_dim', type=int, default=1000)
@click.option('--lin_dim', type=int, default=500)
def train(*args, **kwargs):
    pairs.train(*args, **kwargs)


@cli.command()
@click.argument('test_path', type=click.Path())
@click.argument('m1_path', type=click.Path())
@click.argument('m2_path', type=click.Path())
@click.option('--test_skim', type=int, default=10000)
@click.option('--map_source', default='cpu')
@click.option('--map_target', default='cpu')
def predict(*args, **kwargs):
    pairs.predict(*args, **kwargs)


if __name__ == '__main__':
    cli()