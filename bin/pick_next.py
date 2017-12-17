

import click

from sorder.models import pick_next as model
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
@click.option('--gpu', type=int, default=2)
def train(*args, **kwargs):
    with cuda.gpu(kwargs.pop('gpu')):
        model.train(*args, **kwargs)


@cli.command()
@click.argument('test_path', type=click.Path())
@click.argument('s_encoder_path', type=click.Path())
@click.argument('r_encoder_path', type=click.Path())
@click.argument('classifier_path', type=click.Path())
@click.argument('gp_path', type=click.Path())
@click.option('--test_skim', type=int, default=10000)
@click.option('--map_source', default='cuda:2')
@click.option('--map_target', default='cuda:2')
@click.option('--gpu', type=int, default=2)
def predict(*args, **kwargs):
    with cuda.gpu(kwargs.pop('gpu')):
        model.predict(*args, **kwargs)


if __name__ == '__main__':
    cli()
