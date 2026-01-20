import logging
import os
import time
import torch
from utils.meter import AverageMeter
from utils.metrics import R1_mAP_eval,R1_mAP_eval_LTCC
from torch.cuda import amp
import torch.distributed as dist
from torch.utils.tensorboard import SummaryWriter
import datetime
import numpy as np


def do_train(cfg,
             model,
             center_criterion,
             train_loader,
             optimizer,
             optimizer_center,
             scheduler,
             loss_fn,
             local_rank,
             dataset,
             val_loader = None,
             val_loader_same = None):
    log_period = cfg.SOLVER.LOG_PERIOD
    eval_period = cfg.SOLVER.EVAL_PERIOD

    device = "cuda"
    epochs = cfg.SOLVER.MAX_EPOCHS

    logger = logging.getLogger("EVA-attribure.train")
    logger.info('start training')
    _LOCAL_PROCESS_GROUP = None

    if device:
        model.to(local_rank)
        if torch.cuda.device_count() > 1 and cfg.MODEL.DIST_TRAIN:
            print('Using {} GPUs for training'.format(torch.cuda.device_count()))
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank],find_unused_parameters=True)

    train_writer = SummaryWriter(os.path.join(cfg.OUTPUT_DIR, 'train'))
    rank_writer = SummaryWriter(os.path.join(cfg.OUTPUT_DIR, 'rank'))
    mAP_writer = SummaryWriter(os.path.join(cfg.OUTPUT_DIR, 'mAP'))
    loss_meter = AverageMeter()
    acc_meter = AverageMeter()
    reranking_flag = bool(getattr(cfg.TEST, "RERANKING", False))
    if cfg.DATA.DATASET == 'ltcc':
        evaluator_diff = R1_mAP_eval_LTCC(dataset.num_query_imgs, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM, reranking=cfg.TEST.RERANKING)  # ltcc
        evaluator_general = R1_mAP_eval(dataset.num_query_imgs, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM, reranking=cfg.TEST.RERANKING)

    elif cfg.DATA.DATASET == 'prcc':
        evaluator_diff = R1_mAP_eval(dataset.num_query_imgs_diff, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM, reranking=cfg.TEST.RERANKING)  # prcc
        evaluator_same = R1_mAP_eval(dataset.num_query_imgs_same, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM, reranking=cfg.TEST.RERANKING)
    else:
        evaluator = R1_mAP_eval(dataset.num_query_imgs, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM, reranking=cfg.TEST.RERANKING)


    scaler = amp.GradScaler()
    best_rank1 = -np.inf
    best_epoch = 0
    start_train_time = time.time()
    # train
    for epoch in range(cfg.TRAIN.START_EPOCH, epochs + 1):
        if hasattr(train_loader.sampler, "set_epoch"):
            train_loader.sampler.set_epoch(epoch)
        start_time = time.time()
        loss_meter.reset()
        acc_meter.reset()

        if cfg.DATA.DATASET == 'ltcc':
            evaluator_diff.reset()
            evaluator_general.reset()

        elif cfg.DATA.DATASET == 'prcc':
            evaluator_diff.reset()
            evaluator_same.reset()
        else:
            evaluator.reset()

        scheduler.step(epoch)
        model.train()
        for idx, data in enumerate(train_loader):
            # print(f"=== Iteration {idx} ===")
            # print(f"Type of data: {type(data)}")
            
            # # 如果是元组或列表，查看长度
            # if isinstance(data, (tuple, list)):
            #     print(f"Length of data: {len(data)}")
            #     for i, item in enumerate(data):
            #         print(f"  data[{i}] type: {type(item)}, shape: {getattr(item, 'shape', 'N/A')}")

            if cfg.MODEL.ADD_META:
                samples, targets, camids, _,clothes, meta = data
                meta = [m.float() for m in meta]
                meta = torch.stack(meta, dim=0)
                meta = meta.cuda(non_blocking=True)
                if cfg.MODEL.MASK_META:
                    if cfg.DATA.RANDOM_NOISE:
                        exclude_columns = [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
                                           35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52,
                                           53, 54, 55, 56,
                                           84, 90, 92, 93, 94, 95, 97, 100, 102, 103, 104]
                        random_matrix = torch.ones_like(meta, dtype=torch.int64)
                        random_matrix = random_matrix.to(device).to(meta.dtype)
                        all_columns = set(range(meta.shape[1]))
                        include_columns = list(all_columns - set(exclude_columns))
                        random_columns = torch.tensor(include_columns, dtype=torch.long).to(device)
                        random_columns = random_columns[
                            torch.randperm(len(random_columns))[:int(0.5 * len(random_columns))]]
                        random_matrix[:, random_columns] = torch.randint(2, size=(meta.shape[0], len(random_columns)),
                                                                         dtype=random_matrix.dtype, device=device)
                        random_matrix[:, exclude_columns] = 0
                        meta = meta * random_matrix
                    else:
                        meta[:, 5:21] = 0
                        meta[:, 35:57] = 0
                        meta[:, 84] = 0
                        meta[:, 90] = 0
                        meta[:, 92:96] = 0
                        meta[:, 97] = 0
                        meta[:, 100] = 0
                        meta[:, 102:105] = 0
            else:
                # ImageDataset(aux_info=False) returns 5 items:
                #   (img, pid, camid, clothes_id, cloth_id_batch)
                # Some experimental branches may append extra fields (e.g., text).
                if len(data) == 5:
                    samples, targets, camids, _, clothes = data
                    meta, text = None, None
                elif len(data) == 6:
                    samples, targets, camids, _, clothes, text = data
                    meta = None
                elif len(data) == 7:
                    samples, targets, camids, _, clothes, meta, text = data
                else:
                    raise ValueError(f"Unexpected batch size: len(data)={len(data)}")


            samples = samples.cuda(non_blocking=True)
            targets = targets.cuda(non_blocking=True)
            optimizer.zero_grad()
            optimizer_center.zero_grad()
            with amp.autocast(enabled=True):
                if cfg.MODEL.ADD_TEXT:
                    # text 如果 dataloader 没给，就传 None，让 model 内部用 cocoop 生成
                    score, feat = model(samples, pids=targets, text=text)
                elif cfg.MODEL.ADD_META:
                    score, feat = model(samples, meta)
                else:
                    score, feat = model(samples)

            loss = loss_fn(score, feat, targets, camids)
            train_writer.add_scalar('loss', loss.item(), epoch)
            scaler.scale(loss).backward()

            scaler.step(optimizer)
            scaler.update()
            if 'center' in cfg.MODEL.METRIC_LOSS_TYPE:
                for param in center_criterion.parameters():
                    param.grad.data *= (1. / cfg.SOLVER.CENTER_LOSS_WEIGHT)
                scaler.step(optimizer_center)
                scaler.update()
            if isinstance(score, list):
                acc = (score[0].max(1)[1] == targets).float().mean()
            else:
                acc = (score.max(1)[1] == targets).float().mean()

            loss_meter.update(loss.item(), samples.shape[0])
            acc_meter.update(acc, 1)

            torch.cuda.synchronize()
            if (idx + 1) % log_period == 0:
                logger.info("Epoch[{}] Iteration[{}/{}] Loss: {:.3f}, Acc: {:.3f}, Base Lr: {:.2e}"
                            .format(epoch, (idx + 1), len(train_loader),
                                    loss_meter.avg, acc_meter.avg, scheduler._get_lr(epoch)[0]))

        end_time = time.time()
        time_per_batch = (end_time - start_time) / (idx + 1)
        if cfg.MODEL.DIST_TRAIN:
            pass
        else:
            logger.info("Epoch {} done. Time per batch: {:.3f}[s] Speed: {:.1f}[samples/s]"
                    .format(epoch, time_per_batch, train_loader.batch_size / time_per_batch))

        if epoch % eval_period == 0:
            model.eval()
            if cfg.DATA.DATASET == 'prcc':
                logger.info("Clothes changing setting")
                rank1= test(cfg, model, evaluator_diff, val_loader, logger, device,epoch, rank_writer, mAP_writer)
                logger.info("Standard setting")
                test(cfg, model, evaluator_same, val_loader_same, logger, device, epoch,  rank_writer, mAP_writer,test=True)
            elif cfg.DATA.DATASET == 'ltcc':
                logger.info("Clothes changing setting")
                rank1 = test(cfg, model, evaluator_diff, val_loader, logger, device,epoch,rank_writer, mAP_writer, cc=True)
                logger.info("Standard setting")
                test(cfg, model, evaluator_general, val_loader, logger, device, epoch, rank_writer, mAP_writer,test=True)
            else:
                rank1= test(cfg, model, evaluator, val_loader, logger, device,epoch,rank_writer, mAP_writer)
            # is_best = rank1 > best_rank1
            # if is_best:
            #     best_rank1 = rank1
            #     best_epoch = epoch
            #     logger.info("==> Best Rank-1 {:.1%}, achieved at epoch {}".format(best_rank1, best_epoch))
            #     if cfg.MODEL.DIST_TRAIN:
            #         if dist.get_rank() == 0:
            #             torch.save(model.state_dict(),
            #                        os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + '_best.pth'))
            save_path = os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + '_best.pth')
            is_best = rank1 > best_rank1
            if is_best:
                best_rank1 = rank1
                best_epoch = epoch
                logger.info("==> Best Rank-1 {:.1%}, achieved at epoch {}".format(best_rank1, best_epoch))

                if cfg.MODEL.DIST_TRAIN:
                    if dist.get_rank() == 0:
                        torch.save(model.state_dict(), save_path)
                else:
                    torch.save(model.state_dict(), save_path)

    if cfg.MODEL.DIST_TRAIN:
        if dist.get_rank() == 0:
            logger.info("==> Best Rank-1 {:.1%}, achieved at epoch {}".format(best_rank1, best_epoch))

    total_time = time.time() - start_train_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info('Training time {}'.format(total_time_str))


def do_inference(cfg,
                 model,
                 dataset,
                 val_loader = None,
                 val_loader_same=None):
    logger = logging.getLogger("EVA-attribure.test")
    logger.info("Enter inferencing")

    logger.info("transreid inferencing")
    device = "cuda"
    # reranking = bool(getattr(cfg.TEST, "RERANKING", False))
    if cfg.DATA.DATASET == 'ltcc':
        evaluator_diff = R1_mAP_eval_LTCC(dataset.num_query_imgs, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM, reranking=cfg.TEST.RERANKING)  # ltcc
        evaluator_general = R1_mAP_eval(dataset.num_query_imgs, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM, reranking=cfg.TEST.RERANKING)

    elif cfg.DATA.DATASET == 'prcc':
        evaluator_diff = R1_mAP_eval(dataset.num_query_imgs_diff, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM, reranking=cfg.TEST.RERANKING)  # prcc
        evaluator_same = R1_mAP_eval(dataset.num_query_imgs_same, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM, reranking=cfg.TEST.RERANKING)
    else:
        evaluator = R1_mAP_eval(dataset.num_query_imgs, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM, reranking=cfg.TEST.RERANKING)
    if cfg.DATA.DATASET == 'ltcc':
        evaluator_diff.reset()
        evaluator_general.reset()

    elif cfg.DATA.DATASET == 'prcc':
        evaluator_diff.reset()
        evaluator_same.reset()
    else:
        evaluator.reset()
    model.to(device)
    model.eval()
    if cfg.DATA.DATASET == 'prcc':
        logger.info("Clothes changing setting")
        test(cfg, model, evaluator_diff, val_loader, logger, device, test=True)
        logger.info("Standard setting")
        test(cfg, model, evaluator_same, val_loader_same, logger, device, test=True)
    elif cfg.DATA.DATASET == 'ltcc':
        logger.info("Clothes changing setting")
        test(cfg, model, evaluator_diff, val_loader, logger, device, test=True,cc=True)
        logger.info("Standard setting")
        test(cfg, model, evaluator, val_loader, logger, device, test=True)
    else:
        test(cfg, model, evaluator, val_loader, logger, device, test=True)

# def test(cfg, model, evaluator, val_loader, logger, device, epoch=None, rank_writer=None, mAP_writer=None,test=False,cc=False):
#     for n_iter, batch in enumerate(val_loader):
#         # val_loader may return:
#         # 5 items: (imgs, pids, camids, clothes_id, clothes_ids)
#         # 6 items: (..., meta)
#         # 7 items: (..., meta, text)  or (..., text) depending on your branch
#         if len(batch) == 5:
#             imgs, pids, camids, clothes_id, clothes_ids = batch
#             meta = None
#             text = None
#         elif len(batch) == 6:
#             imgs, pids, camids, clothes_id, clothes_ids, meta = batch
#             text = None
#         elif len(batch) == 7:
#             imgs, pids, camids, clothes_id, clothes_ids, meta, text = batch
#         else:
#             raise ValueError(f"Unexpected val batch size: len(batch)={len(batch)}")

#         imgs = imgs.to(device, non_blocking=True)

#         # meta only exists when dataset provides it
#         if meta is not None:
#             meta = meta.to(device, non_blocking=True).to(torch.float32)

#         with torch.no_grad():
#             imgs = imgs.to(device)
#             if meta is None:
#                 meta = torch.zeros((imgs.size(0), 105), device=device, dtype=torch.float32)
#             if cfg.MODEL.MASK_META:
#                 meta[:, 5:21] = 0
#                 meta[:, 35:57] = 0
#                 meta[:, 84] = 0
#                 meta[:, 90] = 0
#                 meta[:, 92:96] = 0
#                 meta[:, 97] = 0
#                 meta[:, 100] = 0
#                 meta[:, 102:105] = 0
#             if cfg.TEST.TYPE == 'image_only':
#                 meta = torch.zeros_like(meta)
#             feat = model(imgs, meta)
#             if cc:
#                 evaluator.update((feat, pids, camids, clothes_id))
#             else:
#                 evaluator.update((feat, pids, camids))
#     cmc, mAP, _, _, _, _, _ = evaluator.compute()
#     for r in [1, 5, 10]:
#         logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
#     if test :
#         torch.cuda.empty_cache()
#         return
#     logger.info("Validation Results - Epoch: {}".format(epoch))
#     rank1 = cmc[0]
#     rank_writer.add_scalar('rank1', rank1, epoch)
#     mAP_writer.add_scalar('mAP', mAP, epoch)
#     torch.cuda.empty_cache()
#     return rank1
def test(cfg, model, evaluator, val_loader, logger, device,
         epoch=None, rank_writer=None, mAP_writer=None, test=False, cc=False):

    for n_iter, batch in enumerate(val_loader):

        # -------- 1) batch 解包：兼容 5/6/7 元素 --------
        if len(batch) == 5:
            imgs, pids, camids, clothes_id, clothes_ids = batch
            meta, text = None, None
        elif len(batch) == 6:
            imgs, pids, camids, clothes_id, clothes_ids, meta = batch
            text = None
        elif len(batch) == 7:
            imgs, pids, camids, clothes_id, clothes_ids, meta, text = batch
        else:
            raise ValueError(f"Unexpected val batch size: len(batch)={len(batch)}")

        with torch.no_grad():
            imgs = imgs.to(device, non_blocking=True)

            # -------- 2) meta 仅在存在时处理/搬运 --------
            if meta is not None:
                meta = meta.to(device, non_blocking=True).to(torch.float32)

                if cfg.MODEL.MASK_META:
                    meta[:, 5:21] = 0
                    meta[:, 35:57] = 0
                    meta[:, 84] = 0
                    meta[:, 90] = 0
                    meta[:, 92:96] = 0
                    meta[:, 97] = 0
                    meta[:, 100] = 0
                    meta[:, 102:105] = 0

                if cfg.TEST.TYPE == 'image_only':
                    meta = torch.zeros_like(meta)

            # # -------- 3) 关键：纯EVA不传meta；需要meta才传 --------
            # if getattr(cfg.MODEL, "ADD_TEXT", False):
            #     # # 如果你开启了 ADD_META 但 val_loader 没给 meta，就补一个全0
            #     # if text is None:
            #     #     text_dim = getattr(cfg.MODEL, "TEXT_DIMS", 512)
            #     #     text = torch.zeros((imgs.size(0), text_dim), device=device, dtype=torch.float32)

            #     out = model(imgs, text=text if text is not None else None, pids=pids)   # meta模型/你自定义wrapper才走这里
            # else:
            #     out = model(imgs)         # 纯 timm EVA 一定走这里
            # -------- 3) A: eval/test 强制 image_only 时，不进入 add_text 注入路径 --------
            # 只要 TEST.TYPE=image_only，就一律只跑图像分支（不传 text / pids）
            force_image_only = str(getattr(cfg.TEST, "TYPE", "image_only")).lower() == "image_only"

            if force_image_only:
                out = model(imgs)  # 强制图像路径
            else:
                # 非 image_only 时，按需走文本/融合路径
                if getattr(cfg.MODEL, "ADD_TEXT", False):
                    out = model(imgs, text=text if text is not None else None, pids=pids)
                else:
                    out = model(imgs)

            # -------- 4) 兼容不同 model 输出格式：feat 或 (score, feat) --------
            if isinstance(out, (tuple, list)):
                feat = out[-1]  # 常见是 (score, feat) / (feat, ...)
            else:
                feat = out

            if cc:
                evaluator.update((feat, pids, camids, clothes_id))
            else:
                evaluator.update((feat, pids, camids))
    # 在 evaluator.compute() 之前插入
    try:
        q_pids = np.asarray(evaluator.q_pids)
        g_pids = np.asarray(evaluator.g_pids)
        inter = set(q_pids.tolist()) & set(g_pids.tolist())
        logger.info(f"[SANITY] #q={len(q_pids)} #g={len(g_pids)} "
                    f"#q_pids={len(set(q_pids.tolist()))} #g_pids={len(set(g_pids.tolist()))} "
                    f"#intersection={len(inter)}")
    except Exception as e:
        logger.warning(f"[SANITY] cannot access evaluator pids: {e}")

    cmc, mAP, _, _, _, _, _ = evaluator.compute()
    for r in [1, 5, 10]:
        logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
        logger.info("mAP: {:.1%}".format(mAP))


    if test:
        torch.cuda.empty_cache()
        return

    logger.info("Validation Results - Epoch: {}".format(epoch))
    rank1 = cmc[0]
    rank_writer.add_scalar('rank1', rank1, epoch)
    mAP_writer.add_scalar('mAP', mAP, epoch)
    torch.cuda.empty_cache()
    return rank1