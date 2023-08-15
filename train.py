#!/usr/bin/env python3 -u
# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.
"""
Train a new model on one or across multiple GPUs.
"""

import collections
from collections import OrderedDict
import itertools
import os
import math
import torch

from fairseq import distributed_utils, options, progress_bar, tasks, utils
from fairseq.data import iterators
from fairseq.trainer import Trainer
from fairseq.meters import AverageMeter, StopwatchMeter


def main(args):
    if args.max_tokens is None:
        args.max_tokens = 6000
    print(args)

    if not torch.cuda.is_available():
        raise NotImplementedError('Training on CPU is not supported')
    torch.cuda.set_device(args.device_id)
    torch.manual_seed(args.seed)

    # Setup task, e.g., translation, language modeling, etc.
    task = tasks.setup_task(args)

    # Load dataset splits
    load_dataset_splits(task, ['train', 'valid'])

    # Build model and criterion
    model = task.build_model(args)
    criterion = task.build_criterion(args)
    print(f'| model {args.arch}, criterion {criterion.__class__.__name__}')
    print(f'| num. model params: {sum(p.numel() for p in model.parameters())}')

    # Make a dummy batch to (i) warm the caching allocator and (ii) as a
    # placeholder DistributedDataParallel when there's an uneven number of
    # batches per worker.
    max_positions = utils.resolve_max_positions(
        task.max_positions(),
        model.max_positions(),
    )
    dummy_batch = task.dataset('train').get_dummy_batch(args.max_tokens, max_positions)
    oom_batch = task.dataset('train').get_dummy_batch(1, max_positions)

    # Build trainer
    trainer = Trainer(args, task, model, criterion, dummy_batch, oom_batch)
    print(f'| training on {args.distributed_world_size} GPUs')
    print(
        f'| max tokens per GPU = {args.max_tokens} and max sentences per GPU = {args.max_sentences}'
    )

    # Initialize dataloader
    epoch_itr = task.get_batch_iterator(
        dataset=task.dataset(args.train_subset),
        max_tokens=args.max_tokens,
        max_sentences=args.max_sentences,
        max_positions=max_positions,
        ignore_invalid_inputs=True,
        required_batch_size_multiple=8,
        seed=args.seed,
        num_shards=args.distributed_world_size,
        shard_id=args.distributed_rank,
    )
    if args.load_lm:
        load_lm_state(args, model)
    for param in model.srclmdecoder.parameters():
        param.requires_grad = False
    for param in model.tgtlmdecoder.parameters():
        param.requires_grad = False
    # Load the latest checkpoint if one is available
    if not load_checkpoint(args, trainer, epoch_itr):
        trainer.dummy_train_step([dummy_batch])

    # Train until the learning rate gets too small
    max_epoch = args.max_epoch or math.inf
    max_update = args.max_update or math.inf
    lr = trainer.get_lr()
    train_meter = StopwatchMeter()
    train_meter.start()
    valid_losses = [None]
    valid_subsets = args.valid_subset.split(',')
    if args.load_nmt:
        save_checkpoint(args, trainer, epoch_itr, valid_losses[0], nmt=True)
    while lr > args.min_lr and epoch_itr.epoch < max_epoch and trainer.get_num_updates() < max_update:
        # train for one epoch
        train(args, trainer, task, epoch_itr)

        if epoch_itr.epoch % args.validate_interval == 0:
            valid_losses = validate(args, trainer, task, epoch_itr, valid_subsets)

        # only use first validation loss to update the learning rate
        lr = trainer.lr_step(epoch_itr.epoch, valid_losses[0])

        # save checkpoint
        if epoch_itr.epoch % args.save_interval == 0:
            save_checkpoint(args, trainer, epoch_itr, valid_losses[0])
    train_meter.stop()
    print('| done training in {:.1f} seconds'.format(train_meter.sum))


def train(args, trainer, task, epoch_itr):
    """Train the model for one epoch."""

    # Update parameters every N batches
    if epoch_itr.epoch <= len(args.update_freq):
        update_freq = args.update_freq[epoch_itr.epoch - 1]
    else:
        update_freq = args.update_freq[-1]

    # Initialize data iterator
    itr = epoch_itr.next_epoch_itr(fix_batches_to_gpus=args.fix_batches_to_gpus)
    itr = iterators.GroupedIterator(itr, update_freq)
    progress = progress_bar.build_progress_bar(
        args, itr, epoch_itr.epoch, no_progress_bar='simple',
    )

    extra_meters = collections.defaultdict(lambda: AverageMeter())
    first_valid = args.valid_subset.split(',')[0]
    max_update = args.max_update or math.inf
    for i, samples in enumerate(progress, start=epoch_itr.iterations_in_epoch):
        log_output = trainer.train_step(samples)
        if log_output is None:
            continue

        # log mid-epoch stats
        stats = get_training_stats(trainer)
        for k, v in log_output.items():
            if k in ['loss', 'nll_loss', 'ntokens', 'nsentences', 'sample_size']:
                continue  # these are already logged above
            if 'loss' in k:
                extra_meters[k].update(v, log_output['sample_size'])
            else:
                extra_meters[k].update(v)
            stats[k] = extra_meters[k].avg
        progress.log(stats)

        # ignore the first mini-batch in words-per-second calculation
        if i == 0:
            trainer.get_meter('wps').reset()

        num_updates = trainer.get_num_updates()
        if args.save_interval_updates > 0 and num_updates % args.save_interval_updates == 0 and num_updates > 0:
            valid_losses = validate(args, trainer, task, epoch_itr, [first_valid])
            save_checkpoint(args, trainer, epoch_itr, valid_losses[0])

        if num_updates >= max_update:
            break

    # log end-of-epoch stats
    stats = get_training_stats(trainer)
    for k, meter in extra_meters.items():
        stats[k] = meter.avg
    progress.print(stats)

    # reset training meters
    for k in [
        'train_loss', 'train_nll_loss', 'wps', 'ups', 'wpb', 'bsz', 'gnorm', 'clip',
    ]:
        meter = trainer.get_meter(k)
        if meter is not None:
            meter.reset()


def get_training_stats(trainer):
    stats = collections.OrderedDict()
    stats['loss'] = '{:.3f}'.format(trainer.get_meter('train_loss').avg)
    if trainer.get_meter('train_nll_loss').count > 0:
        nll_loss = trainer.get_meter('train_nll_loss').avg
        stats['nll_loss'] = '{:.3f}'.format(nll_loss)
    else:
        nll_loss = trainer.get_meter('train_loss').avg
    stats['ppl'] = get_perplexity(nll_loss)
    stats['wps'] = round(trainer.get_meter('wps').avg)
    stats['ups'] = '{:.1f}'.format(trainer.get_meter('ups').avg)
    stats['wpb'] = round(trainer.get_meter('wpb').avg)
    stats['bsz'] = round(trainer.get_meter('bsz').avg)
    stats['num_updates'] = trainer.get_num_updates()
    stats['lr'] = trainer.get_lr()
    # stats['tradeoff'] = trainer.get_tradeoff()
    stats['gnorm'] = '{:.3f}'.format(trainer.get_meter('gnorm').avg)
    stats['clip'] = '{:.0%}'.format(trainer.get_meter('clip').avg)
    stats['oom'] = trainer.get_meter('oom').avg
    if trainer.get_meter('loss_scale') is not None:
        stats['loss_scale'] = '{:.3f}'.format(trainer.get_meter('loss_scale').avg)
    stats['wall'] = round(trainer.get_meter('wall').elapsed_time)
    stats['train_wall'] = round(trainer.get_meter('train_wall').sum)
    return stats


def validate(args, trainer, task, epoch_itr, subsets):
    """Evaluate the model on the validation set(s) and return the losses."""
    valid_losses = []
    for subset in subsets:
        # Initialize data iterator
        itr = task.get_batch_iterator(
            dataset=task.dataset(subset),
            max_tokens=args.max_tokens,
            max_sentences=args.max_sentences_valid,
            max_positions=utils.resolve_max_positions(
                task.max_positions(),
                trainer.get_model().max_positions(),
            ),
            ignore_invalid_inputs=args.skip_invalid_size_inputs_valid_test,
            required_batch_size_multiple=8,
            seed=args.seed,
            num_shards=args.distributed_world_size,
            shard_id=args.distributed_rank,
        ).next_epoch_itr(shuffle=False)
        progress = progress_bar.build_progress_bar(
            args,
            itr,
            epoch_itr.epoch,
            prefix=f"valid on \'{subset}\' subset",
            no_progress_bar='simple',
        )

        # reset validation loss meters
        for k in ['valid_loss', 'valid_nll_loss']:
            meter = trainer.get_meter(k)
            if meter is not None:
                meter.reset()
        extra_meters = collections.defaultdict(lambda: AverageMeter())

        for sample in progress:
            log_output = trainer.valid_step(sample)

            for k, v in log_output.items():
                if k in ['loss', 'nll_loss', 'ntokens', 'nsentences', 'sample_size']:
                    continue
                extra_meters[k].update(v)

        # log validation stats
        stats = get_valid_stats(trainer)
        for k, meter in extra_meters.items():
            stats[k] = meter.avg
        progress.print(stats)

        valid_losses.append(stats['valid_loss'])
    return valid_losses


def get_valid_stats(trainer):
    stats = collections.OrderedDict()
    stats['valid_loss'] = trainer.get_meter('valid_loss').avg
    if trainer.get_meter('valid_nll_loss').count > 0:
        nll_loss = trainer.get_meter('valid_nll_loss').avg
        stats['valid_nll_loss'] = nll_loss
    else:
        nll_loss = trainer.get_meter('valid_loss').avg
    stats['valid_ppl'] = get_perplexity(nll_loss)
    stats['num_updates'] = trainer.get_num_updates()
    if hasattr(save_checkpoint, 'best'):
        stats['best'] = min(save_checkpoint.best, stats['valid_loss'])
    return stats


def get_perplexity(loss):
    try:
        return '{:.2f}'.format(math.pow(2, loss))
    except OverflowError:
        return float('inf')


def save_checkpoint(args, trainer, epoch_itr, val_loss, nmt=False):
    if args.no_save or not distributed_utils.is_master(args):
        return
    epoch = epoch_itr.epoch
    end_of_epoch = epoch_itr.end_of_epoch() if not nmt else True
    updates = trainer.get_num_updates()

    checkpoint_conds = collections.OrderedDict()
    checkpoint_conds[f'checkpoint{epoch}.pt'] = (
        end_of_epoch
        and not args.no_epoch_checkpoints
        and epoch % args.save_interval == 0
    )
    checkpoint_conds[f'checkpoint_{epoch}_{updates}.pt'] = (
        not end_of_epoch
        and args.save_interval_updates > 0
        and updates % args.save_interval_updates == 0
    )
    checkpoint_conds['checkpoint_best.pt'] = (
            val_loss is not None and
            (not hasattr(save_checkpoint, 'best') or val_loss < save_checkpoint.best)
    )
    checkpoint_conds['checkpoint_last.pt'] = True  # keep this last so that it's a symlink

    prev_best = getattr(save_checkpoint, 'best', val_loss)
    if val_loss is not None:
        save_checkpoint.best = min(val_loss, prev_best)
    extra_state = {
        'train_iterator': epoch_itr.state_dict(),
        'val_loss': val_loss,
    }
    if hasattr(save_checkpoint, 'best'):
        extra_state['best'] = save_checkpoint.best

    if checkpoints := [
        os.path.join(args.save_dir, fn)
        for fn, cond in checkpoint_conds.items()
        if cond
    ]:
        for cp in checkpoints:
            trainer.save_checkpoint(cp, extra_state)

    if not end_of_epoch and args.keep_interval_updates > 0:
        # remove old checkpoints; checkpoints are sorted in descending order
        checkpoints = utils.checkpoint_paths(args.save_dir, pattern=r'checkpoint_\d+_(\d+)\.pt')
        for old_chk in checkpoints[args.keep_interval_updates:]:
            os.remove(old_chk)


def load_checkpoint(args, trainer, epoch_itr):
    """Load a checkpoint and replay dataloader to match."""
    os.makedirs(args.save_dir, exist_ok=True)
    checkpoint_path = os.path.join(args.save_dir, args.restore_file)
    if args.load_nmt:
        checkpoint_path = os.path.join(args.save_dir, args.load_nmt_file)
        assert os.path.exists(checkpoint_path), 'You have specified --load-nmt flag, but there is no nmt model checkpoint.'
        print("Notice that your model will load from a NMT ckt,\nIf you do not want it to happen, please cancel the --load-nmt flag")
    if os.path.isfile(checkpoint_path):
        extra_state = trainer.load_checkpoint(
            checkpoint_path,
            args.reset_optimizer,
            args.reset_lr_scheduler,
            eval(args.optimizer_overrides),
            strict=not args.load_nmt,
        )
        if extra_state is not None:
            # replay train iterator to match checkpoint
            epoch_itr.load_state_dict(extra_state['train_iterator'])

            print(
                f'| loaded checkpoint {checkpoint_path} (epoch {epoch_itr.epoch} @ {trainer.get_num_updates()} updates)'
            )

            trainer.lr_step(epoch_itr.epoch)
            trainer.lr_step_update(trainer.get_num_updates())
            if 'best' in extra_state:
                save_checkpoint.best = extra_state['best']
        return True
    return False


def load_dataset_splits(task, splits):
    for split in splits:
        if split == 'train':
            task.load_dataset(split, combine=True)
        else:
            for k in itertools.count():
                split_k = split + (str(k) if k > 0 else '')
                try:
                    task.load_dataset(split_k, combine=False)
                except FileNotFoundError as e:
                    if k > 0:
                        break
                    raise e

def load_lm_state(args, model):
    os.makedirs(args.save_dir, exist_ok=True)
    checkpoint_path = os.path.join(args.save_dir, args.load_srclm_file)
    if os.path.isfile(checkpoint_path):
        print('load language model...')
        from torch.serialization import default_restore_location
        state = torch.load(checkpoint_path, map_location=lambda s, l: default_restore_location(s, 'cpu'))
        def upgrade(obj):
            if not isinstance(obj, OrderedDict):
                return obj
            oldkeys = list(obj.keys())
            for k in oldkeys:
                if k.startswith('decoder') and k != 'decoder':
                    newkey = k.split('.', 1)[1]
                else:
                    newkey = k
                obj[newkey] = upgrade(obj[k])
                if k.startswith('decoder'):
                    del obj[k]

        # state = _upgrade_state_dict(state)
        # model.upgrade_state_dict(state['model'])
        upgrade(state['model'])
        # upgrade(state['model']._metadata)
    # load model parameters
        try:
            model.srclmdecoder.load_state_dict(state['model'], strict=True)
        except Exception:
            raise Exception('Cannot load model parameters from source language model checkpoint, '
                            'please ensure that the architectures match')
        checkpoint_path = os.path.join(args.save_dir, args.load_tgtlm_file)
        assert os.path.exists(checkpoint_path)
        state = torch.load(checkpoint_path, map_location=lambda s, l: default_restore_location(s, 'cpu'))
        # state = _upgrade_state_dict(state)
        # model.upgrade_state_dict(state['model'])
        upgrade(state['model'])
        # upgrade(state['model']._metadata)
        # load model parameters
        try:
            model.tgtlmdecoder.load_state_dict(state['model'], strict=True)
        except Exception:
            raise Exception('Cannot load model parameters from target language model checkpoint, '
                            'please ensure that the architectures match')
        return True
    return False

def load_nmt_state(args, model):
    checkpoint_path = os.path.join(args.save_dir, args.load_nmt_file)
    if os.path.isfile(checkpoint_path):
        from torch.serialization import default_restore_location
        state = torch.load(checkpoint_path, map_location=lambda s, l: default_restore_location(s, 'cpu'))
        try:
            model.load_state_dict(state['model'], strict=False)
            print('Load nmt model successfully')
        except Exception:
            raise Exception('Cannot load model parameters from NMT model checkpoint, '
                            'please ensure that the architectures match')


if __name__ == '__main__':
    parser = options.get_training_parser()
    args = options.parse_args_and_arch(parser)

    if args.distributed_port > 0 or args.distributed_init_method is not None:
        from distributed_train import main as distributed_main

        distributed_main(args)
    elif args.distributed_world_size > 1:
        from multiprocessing_train import main as multiprocessing_main

        multiprocessing_main(args)
    else:
        main(args)
