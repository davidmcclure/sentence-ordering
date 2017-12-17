

import click

from sorder.models import context_regression as model
from sorder import cuda


@click.group()
def cli():
    pass


@cli.command()
@click.argument('train_path', type=click.Path())
@click.argument('model_path', type=click.Path())
@click.option('--train_skim', type=int, default=1000000)
@click.option('--lr', type=float, default=1e-3)
@click.option('--epochs', type=int, default=1000)
@click.option('--epoch_size', type=int, default=1000)
@click.option('--batch_size', type=int, default=20)
@click.option('--lstm_dim', type=int, default=500)
@click.option('--lin_dim', type=int, default=500)
@click.option('--gpu', type=int, default=0)
def train(*args, **kwargs):
    with cuda.gpu(kwargs.pop('gpu')):
        model.train(*args, **kwargs)


@cli.command()
@click.argument('test_path', type=click.Path())
@click.argument('sent_encoder_path', type=click.Path())
@click.argument('graf_encoder_path', type=click.Path())
@click.argument('regressor_path', type=click.Path())
@click.argument('gp_path', type=click.Path())
@click.option('--test_skim', type=int, default=10000)
@click.option('--map_source', default='cuda:0')
@click.option('--map_target', default='cuda:0')
@click.option('--gpu', type=int, default=0)
def predict(*args, **kwargs):
    with cuda.gpu(kwargs.pop('gpu')):
        model.predict(*args, **kwargs)


if __name__ == '__main__':
    cli()
