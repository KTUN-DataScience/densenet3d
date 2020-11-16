import torch
from torch.autograd import Variable
import torch.nn.functional as F
import json
import time
import os
import sys
from config import Config
from utils.train_utils import *
from utils.spatial_transforms import *
from utils.temporal_transforms import *
from utils.target_transforms import ClassLabel, VideoID
from utils.target_transforms import Compose as TargetCompose
from utils.get_data import get_test_set, get_validation_set



def train_epoch(epoch, data_loader, model, criterion, optimizer,
                epoch_logger, batch_logger):
    print('train at epoch {}'.format(epoch))

    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    end_time = time.time()
    for i, (inputs, targets) in enumerate(data_loader):
        data_time.update(time.time() - end_time)

        if Config.cuda:
            targets = targets.cuda()
        inputs = Variable(inputs)
        targets = Variable(targets)
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        losses.update(loss.data, inputs.size(0))
        prec1, prec5 = calculate_accuracy(outputs.data, targets.data, topk=(1,5))
        top1.update(prec1, inputs.size(0))
        top5.update(prec5, inputs.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end_time)
        end_time = time.time()

        batch_logger.log({
            'epoch': epoch,
            'batch': i + 1,
            'iter': (epoch - 1) * len(data_loader) + (i + 1),
            'loss': losses.val.item(),
            'prec1': top1.val.item(),
            'prec5': top5.val.item(),
            'lr': optimizer.param_groups[0]['lr']
        })
        if i % 10 == 0:
            print('Epoch: [{0}][{1}/{2}]\t lr: {lr:.5f}\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.5f} ({top1.avg:.5f})\t'
                  'Prec@5 {top5.val:.5f} ({top5.avg:.5f})'.format(
                      epoch,
                      i,
                      len(data_loader),
                      batch_time=batch_time,
                      data_time=data_time,
                      loss=losses,
                      top1=top1,
                      top5=top5,
                      lr=optimizer.param_groups[0]['lr']))

    epoch_logger.log({
        'epoch': epoch,
        'loss': losses.avg.item(),
        'prec1': top1.avg.item(),
        'prec5': top5.avg.item(),
        'lr': optimizer.param_groups[0]['lr']
    })

    #if epoch % opt.checkpoint == 0:
    #    save_file_path = os.path.join(opt.result_path,
    #                                  'save_{}.pth'.format(epoch))
    #    states = {
    #        'epoch': epoch + 1,
    #        'arch': opt.arch,
    #        'state_dict': model.state_dict(),
    #        'optimizer': optimizer.state_dict(),
    #    }
    #    torch.save(states, save_file_path)


def val_epoch(epoch, data_loader, model, criterion, logger):
    print('validation at epoch {}'.format(epoch))

    model.eval()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    end_time = time.time()
    for i, (inputs, targets) in enumerate(data_loader):
        data_time.update(time.time() - end_time)

        if Config.cuda:
            targets = targets.cuda()
        with torch.no_grad():
            inputs = Variable(inputs)
            targets = Variable(targets)
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        prec1, prec5 = calculate_accuracy(outputs.data, targets.data, topk=(1,5))
        top1.update(prec1, inputs.size(0))
        top5.update(prec5, inputs.size(0))

        losses.update(loss.data, inputs.size(0))

        batch_time.update(time.time() - end_time)
        end_time = time.time()

        print('Epoch: [{0}][{1}/{2}]\t'
              'Time {batch_time.val:.5f} ({batch_time.avg:.5f})\t'
              'Data {data_time.val:.5f} ({data_time.avg:.5f})\t'
              'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
              'Prec@1 {top1.val:.5f} ({top1.avg:.5f})\t'
              'Prec@5 {top5.val:.5f} ({top5.avg:.5f})'.format(
                  epoch,
                  i + 1,
                  len(data_loader),
                  batch_time=batch_time,
                  data_time=data_time,
                  loss=losses,
                  top1=top1,
                  top5=top5))

    logger.log({'epoch': epoch,
                'loss': losses.avg.item(),
                'prec1': top1.avg.item(),
                'prec5': top5.avg.item()})

    return losses.avg.item(), top1.avg.item()


def calculate_video_results(output_buffer, video_id, test_results, class_names):
    video_outputs = torch.stack(output_buffer)
    average_scores = torch.mean(video_outputs, dim=0)
    sorted_scores, locs = torch.topk(average_scores, k=10)

    video_results = []
    for i in range(sorted_scores.size(0)):
        video_results.append({
            'label': class_names[int(locs[i])],
            'score': float(sorted_scores[i])
        })

    test_results['results'][video_id] = video_results


def test(data_loader, model, class_names):

    model.eval()

    batch_time = AverageMeter()
    data_time = AverageMeter()

    end_time = time.time()
    output_buffer = []
    previous_video_id = ''
    test_results = {'results': {}}
    for i, (inputs, targets) in enumerate(data_loader):
        data_time.update(time.time() - end_time)

        with torch.no_grad():
            inputs = Variable(inputs)
        outputs = model(inputs)
        if Config.softmax_in_test:
            outputs = F.softmax(outputs, dim=1)

        for j in range(outputs.size(0)):
            if not (i == 0 and j == 0) and targets[j] != previous_video_id:
                calculate_video_results(output_buffer, previous_video_id,
                                        test_results, class_names)
                output_buffer = []
            output_buffer.append(outputs[j].data.cpu())
            previous_video_id = targets[j]

        if (i % 100) == 0:
            with open(
                    os.path.join(Config.result_path, '{}.json'.format(
                        Config.test_subset)), 'w') as f:
                json.dump(test_results, f)

        batch_time.update(time.time() - end_time)
        end_time = time.time()

        print('[{}/{}]\t'
              'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
              'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'.format(
                  i + 1,
                  len(data_loader),
                  batch_time=batch_time,
                  data_time=data_time))
    with open(
            os.path.join(Config.result_path, '{}.json'.format(Config.test_subset)),
            'w') as f:
        json.dump(test_results, f)

def evaluate_model(model):
    torch.manual_seed(Config.seed)
    model = init_model(model)
    mean = get_mean(Config.norm_value, dataset = Config.mean_dataset)
    std = get_std(Config.norm_value)
    norm_method = set_norm_method(mean, std)

    spatial_transform = Compose([
        Scale(int(Config.sample_size / Config.scale_in_test)),
        CornerCrop(Config.sample_size, Config.crop_position_in_test),
        ToTensor(Config.norm_value), norm_method
    ])

    temporal_transform = TemporalRandomCrop(Config.sample_duration, Config.downsample)
    
    target_transform = ClassLabel()

    test_data = get_test_set(spatial_transform, temporal_transform, target_transform)
    test_loader = torch.utils.data.DataLoader(
            test_data,
            batch_size=1,
            shuffle=False,
            num_workers=Config.n_threads,
            pin_memory=True)

    model.eval()

    recorder = []

    batch_time = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    start = time.time()

    for i, (inputs, targets) in enumerate(test_loader):
        if Config.cuda:
            targets = targets.cuda(async=True)
        #inputs = Variable(torch.squeeze(inputs), volatile=True)
        inputs = Variable(inputs, volatile=True)
        targets = Variable(targets, volatile=True)
        outputs = model(inputs)

        recorder.append(outputs.data.cpu().numpy().copy())
        #outputs = torch.unsqueeze(torch.mean(outputs, 0), 0)
        prec1, prec5 = calculate_accuracy(outputs, targets, topk=(1, 5))

        top1.update(prec1, inputs.size(0))
        top5.update(prec5, inputs.size(0))

        batch_time.update(time.time() - start)
        end_time = time.time()

        print('[{0}/{1}]\t'
            'Time {batch_time.val:.5f} ({batch_time.avg:.5f})\t'
            'prec@1 {top1.avg:.5f} prec@5 {top5.avg:.5f}'.format(
                i + 1,
                len(test_loader),
                batch_time=batch_time,
                top1=top1,
                top5=top5))

    video_pred = [np.argmax(np.mean(x, axis=0)) for x in recorder]
    print(video_pred)

    with open('dataset/annotation/categories.txt') as f:
        lines = f.readlines()
        categories = [item.rstrip() for item in lines]

    name_list = [x.strip().split()[0] for x in open('dataset/annotation/vallist.txt')]

    order_dict = {e:i for i, e in enumerate(sorted(name_list))}
    reorder_output = [None] * len(recorder)
    reorder_pred = [None] * len(recorder)
    output_csv = []

    # for i in range(len(recorder)):
    #     idx = order_dict[name_list[i]]
    #     reorder_output[idx] = recorder[i]
    #     reorder_pred[idx] = video_pred[i]
    #     output_csv.append('%s;%s'%(name_list[i],
    #                             categories[video_pred[i]]))

    #     with open(Config.result_path +'/'+Config.dataset + '_predictions.csv','w') as f:
    #         f.write('\n'.join(output_csv))
 
    print('-----Evaluation is finished------')
    print('Overall Prec@1 {:.05f}% Prec@5 {:.05f}%'.format(top1.avg, top5.avg))
    

