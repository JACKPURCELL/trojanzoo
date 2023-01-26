#!/usr/bin/env python3

import re
from trojanzoo.utils.fim import KFAC, EKFAC
from trojanzoo.utils.logger import MetricLogger
from trojanzoo.utils.memory import empty_cache
from trojanzoo.utils.model import accuracy, activate_params
from trojanzoo.utils.output import ansi, get_ansi_len, output_iter, prints
from trojanzoo.environ import env

import torch
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm

from trojanzoo.utils.model import ExponentialMovingAverage
from collections.abc import Callable
from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
import torch.utils.data




def distillation(module: nn.Module, num_classes: int,
          epochs: int, optimizer: Optimizer, lr_scheduler: _LRScheduler = None,
          adv_train: bool = None,
          lr_warmup_epochs: int = 0,
          model_ema: ExponentialMovingAverage = None,
          model_ema_steps: int = 32,
          grad_clip: float = None, pre_conditioner: None | KFAC | EKFAC = None,
          print_prefix: str = 'Train', start_epoch: int = 0, resume: int = 0,
          validate_interval: int = 10, save: bool = False, amp: bool = False,
          loader_train: torch.utils.data.DataLoader = None,
          loader_valid: torch.utils.data.DataLoader = None,
          epoch_fn: Callable[..., None] = None,
          get_data_fn: Callable[..., tuple[torch.Tensor, torch.Tensor]] = None,
          forward_fn: Callable[..., torch.Tensor] = None,
          loss_fn: Callable[..., torch.Tensor] = None,
          after_loss_fn: Callable[..., None] = None,
          validate_fn: Callable[..., tuple[float, float]] = None,
          save_fn: Callable[..., None] = None, file_path: str = None,
          folder_path: str = None, suffix: str = None,
          writer=None, main_tag: str = 'train', tag: str = '',
          accuracy_fn: Callable[..., list[float]] = None,
          verbose: bool = True, output_freq: str = 'iter', indent: int = 0,
          change_train_eval: bool = True, lr_scheduler_freq: str = 'epoch',
          backward_and_step: bool = True, 
          mixmatch: bool = False,
          tea_forward_fn: Callable[..., torch.Tensor] = None,
          interleave_fn = None,
          **kwargs):
    r"""Train the model"""
    if epochs <= 0:
        return
    get_data_fn = get_data_fn or (lambda x: x)
    forward_fn = forward_fn or module.__call__
    loss_fn = loss_fn or (lambda _input, _label, _output=None: F.cross_entropy(_output or forward_fn(_input), _label))

    
    validate_fn = validate_fn or dis_validate 
    accuracy_fn = accuracy_fn or accuracy

    scaler: torch.cuda.amp.GradScaler = None
    if not env['num_gpus']:
        amp = False
    if amp:
        scaler = torch.cuda.amp.GradScaler()
    best_validate_result = (0.0, float('inf'))
    if validate_interval != 0:
        best_validate_result = validate_fn(loader=loader_valid, get_data_fn=get_data_fn,
                                           forward_fn=forward_fn, loss_fn=loss_fn,
                                           writer=None, tag=tag, _epoch=start_epoch,
                                           verbose=verbose, indent=indent, tea_forward_fn=tea_forward_fn, **kwargs)
        best_acc = best_validate_result[0]

    params: list[nn.Parameter] = []
    for param_group in optimizer.param_groups:
        params.extend(param_group['params'])
    len_loader_train = len(loader_train)
    total_iter = (epochs - resume) * len_loader_train

    logger = MetricLogger()
    if mixmatch:
        logger.create_meters(loss=None)
    else:
        logger.create_meters(  gt_loss=None, gt_acc1=None, gt_acc5=None, 
                          hapi_loss=None, hapi_acc1=None, hapi_acc5=None)
    if resume and lr_scheduler:
        for _ in range(resume):
            lr_scheduler.step()
    iterator = range(resume, epochs)
    if verbose and output_freq == 'epoch':
        header: str = '{blue_light}{0}: {reset}'.format(print_prefix, **ansi)
        header = header.ljust(max(len(header), 30) + get_ansi_len(header))
        iterator = logger.log_every(range(resume, epochs),
                                    header=print_prefix,
                                    tqdm_header='Epoch',
                                    indent=indent)
    for _epoch in iterator:
        _epoch += 1
        logger.reset()
        if callable(epoch_fn):
            activate_params(module, [])
            epoch_fn(optimizer=optimizer, lr_scheduler=lr_scheduler,
                     _epoch=_epoch, epochs=epochs, start_epoch=start_epoch)
        loader_epoch = loader_train
        if verbose and output_freq == 'iter':
            header: str = '{blue_light}{0}: {1}{reset}'.format(
                'Epoch', output_iter(_epoch, epochs), **ansi)
            header = header.ljust(max(len('Epoch'), 30) + get_ansi_len(header))
            loader_epoch = logger.log_every(loader_train, header=header,
                                            tqdm_header='Batch',
                                            indent=indent)
        if change_train_eval:
            module.train()
        activate_params(module, params)



#----------------------------------------------------------------

        # _input, _label = get_data_fn(data, mode='valid', **kwargs)
        # with torch.no_grad():
        #     tea_output = forward_tea_fn(_input, amp=amp, parallel=True)
        # results.append([_input,  tea_output, _label])

        # _input, _label = get_data_fn(data, mode='valid')
        # if pre_conditioner is not None and not amp:
        #     pre_conditioner.track.enable()
        # _output = forward_fn(_input, amp=amp, parallel=True)
#----------------------------------------------------------------
#-------------------------new---------------------------------------

        if _epoch < 10000:
            mode = 'train_STU' #kl loss / return raw data
            print(_epoch,mode)
        elif _epoch >= 10000:
            mode = 'train_ADV_STU'  #kl loss / return adv data
            print(_epoch,mode)


#---------------------------new-------------------------------------
        # if _epoch%5 == 0:
        #     mode = 'train_GEN'
        #     print(_epoch,mode)
        # else:
        #     mode = 'train_STU'
        #     print(_epoch,mode)
        
        for i, data in enumerate(loader_epoch):
            _iter = _epoch * len_loader_train + i
            # data_time.update(time.perf_counter() - end)
            # optimizer.zero_grad()
            if mixmatch:
                mixed_input, mixed_target, batch_size = get_data_fn(data, mode=mode)
                # interleave labeled and unlabed samples between batches to get correct batchnorm calculation 
                mixed_input = list(torch.split(mixed_input, batch_size))
                mixed_input = interleave_fn(mixed_input, batch_size)

                logits = [forward_fn(mixed_input[0])]
                for input in mixed_input[1:]:
                    logits.append(forward_fn(input))

                # put interleaved samples back
                logits = interleave_fn(logits, batch_size)
                logits_x = logits[0]
                logits_u = torch.cat(logits[1:], dim=0)

                loss = loss_fn(outputs_x = logits_x, targets_x = mixed_target[:batch_size], outputs_u = logits_u, targets_u = mixed_target[batch_size:], iter = _iter)


            else:
                _input, _label, _soft_label, hapi_label  = get_data_fn(data, mode=mode)
                
                if pre_conditioner is not None and not amp:
                    pre_conditioner.track.enable()
                    #TODO: maybe can remove
                _output = forward_fn(_input, amp=amp, parallel=True)
                loss = loss_fn(_input=_input, _soft_label=_soft_label, _output=_output, amp=amp)
                # print("train loss： ",loss)
                # soft_target = tea_forward_fn(_input, amp=amp, parallel=True)
                # _output =  forward_fn(_input, amp=amp, parallel=True)
                # loss = soft_loss_fn(_input, soft_target,_output)
            if backward_and_step:
                optimizer.zero_grad()
                if amp:
                    scaler.scale(loss).backward()
                    if callable(after_loss_fn) or grad_clip is not None:
                        scaler.unscale_(optimizer)
                    if callable(after_loss_fn) and mode == 'train_ADV_STU':
                        after_loss_fn(_input=_input, _label=_label,
                                      _soft_label=_soft_label, _output=_output,
                                      loss=loss, optimizer=optimizer,
                                      loss_fn=loss_fn,
                                      amp=amp, scaler=scaler,
                                      _iter=_iter, total_iter=total_iter)
                    if grad_clip is not None:
                        nn.utils.clip_grad_norm_(params, grad_clip)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    #backward the weights 
                    loss.backward()
                    if callable(after_loss_fn) and mode == 'train_ADV_STU':#miss
                        # print("after_loss_fn+train_ADV_STU")
                        after_loss_fn(_input=_input, _label=_label,
                                      _soft_label=_soft_label, _output=_output,
                                      loss=loss, optimizer=optimizer,
                                      loss_fn=loss_fn,
                                      amp=amp, scaler=scaler,
                                      _iter=_iter, total_iter=total_iter,
                                      tea_forward_fn=tea_forward_fn)
                        # start_epoch=start_epoch, _epoch=_epoch, epochs=epochs)
                    if pre_conditioner is not None:
                        pre_conditioner.track.disable()
                        pre_conditioner.step()
                    if grad_clip is not None:
                        nn.utils.clip_grad_norm_(params, grad_clip)
                    optimizer.step()

            if model_ema and i % model_ema_steps == 0:
                model_ema.update_parameters(module)
                if _epoch <= lr_warmup_epochs:
                    # Reset ema buffer to keep copying weights
                    # during warmup period
                    model_ema.n_averaged.fill_(0)

            if lr_scheduler and lr_scheduler_freq == 'iter':
                lr_scheduler.step()
                
               
            if mixmatch:
                 logger.update(n=batch_size, loss=float(loss))
            else:    
                hapi_acc1, hapi_acc5 = accuracy_fn(
                    _output, hapi_label, num_classes=num_classes, topk=(1, 5))
                gt_acc1, gt_acc5 = accuracy_fn(
                    _output, _label, num_classes=num_classes, topk=(1, 5))
                batch_size = int(_label.size(0)) 
                logger.update(n=batch_size, gt_acc1=gt_acc1, gt_acc5=gt_acc5, 
                            hapi_loss=float(loss), hapi_acc1=hapi_acc1, hapi_acc5=hapi_acc5)
            empty_cache()
        optimizer.zero_grad()
        if lr_scheduler and lr_scheduler_freq == 'epoch':
            lr_scheduler.step()
        if change_train_eval:
            module.eval()
        activate_params(module, [])
        if mixmatch:
            loss=(logger.meters['loss'].global_avg)
            if writer is not None:
                from torch.utils.tensorboard import SummaryWriter
                assert isinstance(writer, SummaryWriter)
                writer.add_scalars(main_tag='loss/' + main_tag,
                            tag_scalar_dict={tag: loss}, global_step=_epoch + start_epoch)        
        else:
            gt_acc1, hapi_loss, hapi_acc1 = (
                    logger.meters['gt_acc1'].global_avg,
                    logger.meters['hapi_loss'].global_avg,
                    logger.meters['hapi_acc1'].global_avg)
            if writer is not None:
                from torch.utils.tensorboard import SummaryWriter
                assert isinstance(writer, SummaryWriter)
                writer.add_scalars(main_tag='gt_acc1/' + main_tag,
                            tag_scalar_dict={tag: gt_acc1}, global_step=_epoch + start_epoch)        
                writer.add_scalars(main_tag='hapi_loss/' + main_tag,
                            tag_scalar_dict={tag: hapi_loss}, global_step=_epoch + start_epoch)
                writer.add_scalars(main_tag='hapi_acc1/' + main_tag,
                        tag_scalar_dict={tag: hapi_acc1}, global_step=_epoch + start_epoch)
            
        if validate_interval != 0 and (_epoch % validate_interval == 0 or _epoch == epochs):
            validate_result = validate_fn(module=module,
                                          num_classes=num_classes,
                                          loader=loader_valid,
                                          get_data_fn=get_data_fn,
                                          forward_fn=forward_fn,
                                          loss_fn=loss_fn,
                                          writer=writer, tag=tag,
                                          _epoch=_epoch + start_epoch,
                                          verbose=verbose, indent=indent,
                                          tea_forward_fn=tea_forward_fn,
                                          **kwargs)
            cur_acc = validate_result[0]
            if cur_acc >= best_acc:
                best_validate_result = validate_result
                if verbose:
                    prints('{purple}best result update!{reset}'.format(
                        **ansi), indent=indent)
                    prints(f'Current Acc: {cur_acc:.3f}    '
                           f'Previous Best Acc: {best_acc:.3f}',
                           indent=indent)
                best_acc = cur_acc
                if save:
                    save_fn(file_path=file_path, folder_path=folder_path,
                            suffix=suffix, verbose=verbose)
            if verbose:
                prints('-' * 50, indent=indent)
    module.zero_grad()
    return best_validate_result


def dis_validate(module: nn.Module, num_classes: int,
             loader: torch.utils.data.DataLoader,
             print_prefix: str = 'Validate', indent: int = 0,
             verbose: bool = True,
             get_data_fn: Callable[
                 ..., tuple[torch.Tensor, torch.Tensor]] = None,
             forward_fn: Callable[..., torch.Tensor] = None,
             loss_fn: Callable[..., torch.Tensor] = None,
             writer=None, main_tag: str = 'valid',
             tag: str = '', _epoch: int = None,
             accuracy_fn: Callable[..., list[float]] = None,
                        stu_arch_list=None,
              tea_forward_fn: Callable[..., torch.Tensor] = None,
             **kwargs) -> tuple[float, float]:
    r"""Evaluate the model.

    Returns:
        (float, float): Accuracy and loss.
    """
    module.eval()
    get_data_fn = get_data_fn or (lambda x: x)
    forward_fn = forward_fn or module.__call__
    loss_fn = loss_fn or nn.CrossEntropyLoss()
    accuracy_fn = accuracy_fn or accuracy
    logger = MetricLogger()
    logger.create_meters( gt_loss=None, gt_acc1=None, gt_acc5=None, 
                          hapi_loss=None, hapi_acc1=None, hapi_acc5=None)
    loader_epoch = loader  
    if verbose:
        header: str = '{yellow}{0}{reset}'.format(print_prefix, **ansi)
        header = header.ljust(max(len(print_prefix), 30) + get_ansi_len(header))
        loader_epoch = logger.log_every(loader, header=header,
                                        tqdm_header='Batch',
                                        indent=indent)
    for data in loader_epoch:
        _input, _label, _soft_label, hapi_label = get_data_fn(data, mode='valid', **kwargs)
        with torch.no_grad():
            _output = forward_fn(_input)
            gt_loss = float(loss_fn(_input=_input, _label=_label, _output=_output, **kwargs))
            hapi_loss = float(loss_fn(_input=_input, _soft_label=_soft_label, _output=_output, temp=1.0, **kwargs))
            hapi_acc1, hapi_acc5 = accuracy_fn(
                _output, hapi_label, num_classes=num_classes, topk=(1, 5))
            gt_acc1, gt_acc5 = accuracy_fn(
                _output, _label, num_classes=num_classes, topk=(1, 5))
            batch_size = int(_label.size(0))
            logger.update(n=batch_size, gt_loss=float(gt_loss), gt_acc1=gt_acc1, gt_acc5=gt_acc5, 
                          hapi_loss=float(hapi_loss), hapi_acc1=hapi_acc1, hapi_acc5=hapi_acc5)
            
    gt_loss, gt_acc1, hapi_loss, hapi_acc1 = (logger.meters['gt_loss'].global_avg,
                 logger.meters['gt_acc1'].global_avg,
                 logger.meters['hapi_loss'].global_avg,
                 logger.meters['hapi_acc1'].global_avg)


    if writer is not None and _epoch is not None and main_tag:
        from torch.utils.tensorboard import SummaryWriter
        assert isinstance(writer, SummaryWriter)
        writer.add_scalars(main_tag='gt_loss/' + main_tag,
                    tag_scalar_dict={tag: gt_loss}, global_step=_epoch)
        writer.add_scalars(main_tag='gt_acc1/' + main_tag,
                    tag_scalar_dict={tag: gt_acc1}, global_step=_epoch)        
        writer.add_scalars(main_tag='hapi_loss/' + main_tag,
                    tag_scalar_dict={tag: hapi_loss}, global_step=_epoch)
        writer.add_scalars(main_tag='hapi_acc1/' + main_tag,
                    tag_scalar_dict={tag: hapi_acc1}, global_step=_epoch)

    return hapi_acc1, hapi_loss


@torch.no_grad()
def compare(module1: nn.Module, module2: nn.Module,
            loader: torch.utils.data.DataLoader,
            print_prefix='Validate', indent=0, verbose=True,
            get_data_fn: Callable[...,
                                  tuple[torch.Tensor, torch.Tensor]] = None,
            criterion: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = nn.CrossEntropyLoss(),
            **kwargs) -> float:
    module1.eval()
    module2.eval()
    get_data_fn = get_data_fn if get_data_fn is not None else lambda x: x

    logger = MetricLogger()
    logger.create_meters(loss=None)
    loader_epoch = loader
    if verbose:
        header: str = '{yellow}{0}{reset}'.format(print_prefix, **ansi)
        header = header.ljust(
            max(len(print_prefix), 30) + get_ansi_len(header))
        if env['tqdm']:
            loader_epoch = tqdm(loader_epoch, leave=False)
        loader_epoch = logger.log_every(
            loader_epoch, header=header, indent=indent)
    for data in loader_epoch:
        _input, _label = get_data_fn(data, **kwargs)
        _output1: torch.Tensor = module1(_input)
        _output2: torch.Tensor = module2(_input)
        loss = criterion(_output1, _output2.softmax(1)).item()
        batch_size = int(_label.size(0))
        logger.update(n=batch_size, loss=loss)
    return logger.meters['loss'].global_avg
