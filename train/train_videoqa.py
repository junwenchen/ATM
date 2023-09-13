import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import collections
from util import compute_aggreeings, AverageMeter, get_mask, mask_tokens
import os.path as osp
import json

import pdb

def eval(model, data_loader, a2v, criterion, csl_criterion, shuffle_criterion, args, test=False):
    model.eval()
    count = 0
    metrics, counts = collections.defaultdict(int), collections.defaultdict(int)
    metrics_loss = collections.defaultdict(float)

    with torch.no_grad():
        if not args.mc:
            model.module._compute_answer_embedding(a2v)
        results = {}
        for i, batch in enumerate(data_loader):
            # answer_id, answer, video_o, video_f, video_m, question, question_id = (
            #     batch["answer_id"],
            #     batch["answer"],
            #     batch["video_o"].cuda(),
            #     batch["video_f"].cuda(),
            #     batch["video_m"].cuda(),
            #     batch["question"].cuda(),
            #     batch['question_id']
            # )
            answer_id, answer, video_o, video_f, video_m, question, question_id, question_verb, answer_verb = (
                batch["answer_id"],
                batch["answer"],
                batch["video_o"].cuda(),
                batch["video_f"].cuda(),
                batch["video_m"].cuda(),
                batch["question"].cuda(),
                batch['question_id'],
                batch['question_verb'],
                batch['answer_verb'],
            )
            question_verb_mask = (question_verb > 0).float()
            answer_verb_mask = (answer_verb > 0).float()

            video_len = batch["video_len"]
            seq_len = batch["seq_len"]
            question_mask = (question > 0).float()
            answer_mask = (answer > 0).float()
            video_mask = get_mask(video_len, video_f.size(1)).cuda()
            count += answer_id.size(0)
            video = (video_o, video_f, video_m)
            # video = (video_o, video_f)
            if not args.mc:
                predicts = model(
                    video,
                    question,
                    text_mask=question_mask,
                    video_mask=video_mask,
                    seq_len = seq_len,
                )
                topk = torch.topk(predicts, dim=1, k=10).indices.cpu()
                if args.dataset != "ivqa":
                    answer_id_expanded = answer_id.view(-1, 1).expand_as(topk)
                else:
                    answer_id = (answer_id / 2).clamp(max=1)
                    answer_id_expanded = answer_id
                metrics = compute_aggreeings(
                    topk,
                    answer_id_expanded,
                    [1, 10],
                    ["acc", "acc10"],
                    metrics,
                    ivqa=(args.dataset == "ivqa"),
                )
                for bs, qid in enumerate(question_id):
                    results[qid] = {'prediction': int(topk.numpy()[bs,0]), 'answer':int(answer_id.numpy()[bs])}
            else:
                # fusion_proj, _, answer_proj = model(
                #     video,
                #     question,
                #     text_mask=answer_mask,
                #     video_mask=video_mask,
                #     answer=answer.cuda(),
                #     seq_len = seq_len,
                # )
                fusion_proj, _, answer_proj, qv_proj, av_proj, = model(
                    video,
                    question,
                    text_mask=answer_mask,
                    video_mask=video_mask,
                    answer=answer.cuda(),
                    q_mask=question_verb_mask,
                    q_verb=question_verb.cuda(),
                    a_mask=answer_verb_mask,
                    a_verb=answer_verb.cuda(),
                    seq_len = seq_len,
                    # temporal_encoding = temporal_encoding
                )
                # answer_gt_proj = torch.gather(answer_proj, 1, answer_id.to(answer_proj.device).unsqueeze(1).unsqueeze(2).repeat(1, 1, \
                # answer_proj.shape[2])).squeeze(1)

                # Sf = fusion_proj @ answer_gt_proj.permute(1, 0)
                # if args.csl_type == "L1":
                #     Sf_diag = torch.zeros(answer_gt_proj.shape[0], requires_grad=True).to(answer_proj.device)
                #     Sf_diag = Sf_diag.clone()
                #     for j in range(answer_gt_proj.shape[0]):
                #         Sf_diag[j] = Sf[j, j]
                #     margin_loss = F.relu(Sf-Sf_diag.unsqueeze(1) + args.Delta).mean() + F.relu(Sf-Sf_diag.unsqueeze(0) + args.Delta).mean()
                #     csl_loss = csl_criterion(margin_loss, torch.zeros_like(margin_loss))
                # elif args.csl_type == "CE":
                #     Sf_id = torch.arange(0, len(Sf)).to(Sf.device)
                #     csl_loss = criterion(Sf, Sf_id.cuda())
                if "qv" in args.query_list:
                    answer_gt_proj = qv_proj.squeeze(1)
                    Sf = fusion_proj @ answer_gt_proj.permute(1, 0)
                    if args.csl_type == "L1":
                        Sf_diag = torch.zeros(answer_gt_proj.shape[0], requires_grad=True).to(answer_proj.device)
                        Sf_diag = Sf_diag.clone()
                        for j in range(answer_gt_proj.shape[0]):
                            Sf_diag[j] = Sf[j, j]
                        margin_loss = F.relu(Sf-Sf_diag.unsqueeze(1) + args.Delta).mean() + F.relu(Sf-Sf_diag.unsqueeze(0) + args.Delta).mean()
                        qv_csl_loss = csl_criterion(margin_loss, torch.zeros_like(margin_loss))
                    elif args.csl_type == "CE":
                        Sf_id = torch.arange(0, len(Sf)).to(Sf.device)
                        qv_csl_loss = criterion(Sf, Sf_id.cuda())

                if "av" in args.query_list:
                    answer_gt_proj = av_proj.squeeze(1)
                    Sf = fusion_proj @ answer_gt_proj.permute(1, 0)
                    if args.csl_type == "L1":
                        Sf_diag = torch.zeros(answer_gt_proj.shape[0], requires_grad=True).to(answer_proj.device)
                        Sf_diag = Sf_diag.clone()
                        for j in range(answer_gt_proj.shape[0]):
                            Sf_diag[j] = Sf[j, j]
                        margin_loss = F.relu(Sf-Sf_diag.unsqueeze(1) + args.Delta).mean() + F.relu(Sf-Sf_diag.unsqueeze(0) + args.Delta).mean()
                        av_csl_loss = csl_criterion(margin_loss, torch.zeros_like(margin_loss))
                    elif args.csl_type == "CE":
                        Sf_id = torch.arange(0, len(Sf)).to(Sf.device)
                        av_csl_loss = criterion(Sf, Sf_id.cuda())

                if "qav" in args.query_list:
                    answer_gt_proj = torch.gather(answer_proj, 1, answer_id.to(answer_proj.device).unsqueeze(1).unsqueeze(2).repeat(1, 1, \
                    answer_proj.shape[2])).squeeze(1)
                    # answer_gt_proj = av_proj.squeeze(1)
                    Sf = fusion_proj @ answer_gt_proj.permute(1, 0)
                    if args.csl_type == "L1":
                        Sf_diag = torch.zeros(answer_gt_proj.shape[0], requires_grad=True).to(answer_proj.device)
                        Sf_diag = Sf_diag.clone()
                        for j in range(answer_gt_proj.shape[0]):
                            Sf_diag[j] = Sf[j, j]
                        margin_loss = F.relu(Sf-Sf_diag.unsqueeze(1) + args.Delta).mean() + F.relu(Sf-Sf_diag.unsqueeze(0) + args.Delta).mean()
                        qav_csl_loss = csl_criterion(margin_loss, torch.zeros_like(margin_loss))
                    elif args.csl_type == "CE":
                        Sf_id = torch.arange(0, len(Sf)).to(Sf.device)
                        qav_csl_loss = criterion(Sf, Sf_id.cuda())

                fusion_proj = fusion_proj.unsqueeze(2)
                predicts = torch.bmm(answer_proj, fusion_proj).squeeze()
                predicted = torch.max(predicts, dim=1).indices.cpu()
                metrics["acc"] += (predicted == answer_id).sum().item()
                # if "csl" in args.loss_list:
                #     metrics_loss["loss"] += csl_loss.cpu().item()
                if "csl" in args.loss_list:
                    if "qav" in args.query_list:
                        metrics_loss["loss"] += qav_csl_loss.cpu().item()
                    if "av" in args.query_list:
                        metrics_loss["loss"] += av_csl_loss.cpu().item()
                    if "qv" in args.query_list:
                        metrics_loss["loss"] += qv_csl_loss.cpu().item()
                for bs, qid in enumerate(question_id):
                    results[qid] = {'prediction': int(predicted.numpy()[bs]), 'answer':int(answer_id.numpy()[bs])}
    

    step = "val" if not test else "test"
    
    for k in metrics:
        # print(metrics[k], count)
        v = metrics[k] / count
        logging.info(f"{step} {k}: {v:.2%}")
        break
    
    if "csl" in args.loss_list:
        return metrics["acc"] / count, metrics_loss["loss"] / count, results

    return metrics["acc"] / count, results


def train(model, train_loader, a2v, optimizer, criterion, csl_criterion, shuffle_criterion, scheduler, epoch, args, tokenizer):
    model.train()
    running_vqa_loss, running_csl_loss, running_shuffle_loss, running_acc, running_mlm_loss = (
        AverageMeter(),
        AverageMeter(),
        AverageMeter(),
        AverageMeter(),
        AverageMeter(),
    )
    for i, batch in enumerate(train_loader):
        # answer_id, answer, qtype, video_o, video_f, video_m, question = (
        #     batch["answer_id"],
        #     batch["answer"],
        #     batch["type"],
        #     batch["video_o"].cuda(),
        #     batch["video_f"].cuda(),
        #     batch["video_m"].cuda(),
        #     batch["question"].cuda(),
        # )
        answer_id, answer, qtype, video_o, video_f, video_m, question, question_verb, answer_verb = (
            batch["answer_id"],
            batch["answer"],
            batch["type"],
            batch["video_o"].cuda(),
            batch["video_f"].cuda(),
            batch["video_m"].cuda(),
            batch["question"].cuda(),
            batch['question_verb'],
            batch['answer_verb'],
        )

        video_len = batch["video_len"]
        question_mask = (question > 0).float()
        answer_mask = (answer > 0).float()

        question_verb_mask = (question_verb > 0).float()
        answer_verb_mask = (answer_verb > 0).float()
        video_mask = (
            get_mask(video_len, video_f.size(1)).cuda() if args.max_feats > 0 else None
        )
        # print(video_mask.shape)
        video = (video_o, video_f, video_m)
        # video = (video_o, video_f)
        N = answer_id.size(0)
        seq_len = batch["seq_len"]
        if not args.mc:
            model.module._compute_answer_embedding(a2v)
            predicts = model(
                video,
                question,
                text_mask=question_mask,
                video_mask=video_mask,
                seq_len = seq_len,
                temporal_encoding=temporal_encoding
            )
            # print(predicts.shape, answer_id)
        else:
            # fusion_proj, shuffle_fusion_proj, answer_proj = model(
            fusion_proj, shuffle_fusion_proj, answer_proj, qv_proj, av_proj = model(
                video,
                question,
                text_mask=answer_mask,
                video_mask=video_mask,
                answer=answer.cuda(),
                q_mask=question_verb_mask,
                q_verb=question_verb.cuda(),
                a_mask=answer_verb_mask,
                a_verb=answer_verb.cuda(),
                seq_len = seq_len,
                # temporal_encoding = temporal_encoding
            )

            if "qv" in args.query_list:
                answer_gt_proj = qv_proj.squeeze(1)
                Sf = fusion_proj @ answer_gt_proj.permute(1, 0)
                if args.csl_type == "L1":
                    Sf_diag = torch.zeros(answer_gt_proj.shape[0], requires_grad=True).to(answer_proj.device)
                    Sf_diag = Sf_diag.clone()
                    for j in range(answer_gt_proj.shape[0]):
                        Sf_diag[j] = Sf[j, j]
                    margin_loss = F.relu(Sf-Sf_diag.unsqueeze(1) + args.Delta).mean() + F.relu(Sf-Sf_diag.unsqueeze(0) + args.Delta).mean()
                    qv_csl_loss = csl_criterion(margin_loss, torch.zeros_like(margin_loss))
                elif args.csl_type == "CE":
                    Sf_id = torch.arange(0, len(Sf)).to(Sf.device)
                    qv_csl_loss = criterion(Sf, Sf_id.cuda())
            if "av" in args.query_list:
                answer_gt_proj = av_proj.squeeze(1)
                Sf = fusion_proj @ answer_gt_proj.permute(1, 0)
                if args.csl_type == "L1":
                    Sf_diag = torch.zeros(answer_gt_proj.shape[0], requires_grad=True).to(answer_proj.device)
                    Sf_diag = Sf_diag.clone()
                    for j in range(answer_gt_proj.shape[0]):
                        Sf_diag[j] = Sf[j, j]
                    margin_loss = F.relu(Sf-Sf_diag.unsqueeze(1) + args.Delta).mean() + F.relu(Sf-Sf_diag.unsqueeze(0) + args.Delta).mean()
                    av_csl_loss = csl_criterion(margin_loss, torch.zeros_like(margin_loss))
                elif args.csl_type == "CE":
                    Sf_id = torch.arange(0, len(Sf)).to(Sf.device)
                    av_csl_loss = criterion(Sf, Sf_id.cuda())
            if "qav" in args.query_list:
                answer_gt_proj = torch.gather(answer_proj, 1, answer_id.to(answer_proj.device).unsqueeze(1).unsqueeze(2).repeat(1, 1, \
                answer_proj.shape[2])).squeeze(1)
                # answer_gt_proj = av_proj.squeeze(1)
                Sf = fusion_proj @ answer_gt_proj.permute(1, 0)
                if args.csl_type == "L1":
                    Sf_diag = torch.zeros(answer_gt_proj.shape[0], requires_grad=True).to(answer_proj.device)
                    Sf_diag = Sf_diag.clone()
                    for j in range(answer_gt_proj.shape[0]):
                        Sf_diag[j] = Sf[j, j]
                    margin_loss = F.relu(Sf-Sf_diag.unsqueeze(1) + args.Delta).mean() + F.relu(Sf-Sf_diag.unsqueeze(0) + args.Delta).mean()
                    qav_csl_loss = csl_criterion(margin_loss, torch.zeros_like(margin_loss))
                elif args.csl_type == "CE":
                    Sf_id = torch.arange(0, len(Sf)).to(Sf.device)
                    qav_csl_loss = criterion(Sf, Sf_id.cuda())


            fusion_proj = fusion_proj.unsqueeze(2)
            predicts = torch.bmm(answer_proj, fusion_proj).squeeze()

            shuffle_predicts = torch.bmm(answer_proj, shuffle_fusion_proj.unsqueeze(2)).squeeze()


        if args.dataset == "ivqa":
            a = (answer_id / 2).clamp(max=1).cuda()
            vqa_loss = criterion(predicts, a)
            predicted = torch.max(predicts, dim=1).indices.cpu()
            predicted = F.one_hot(predicted, num_classes=len(a2v))
            running_acc.update((predicted * a.cpu()).sum().item() / N, N)
        else:
            vqa_loss = criterion(predicts, answer_id.cuda())
            #adding shuffle
            shuffle_loss = shuffle_criterion(shuffle_predicts, qtype.to(shuffle_predicts.device))
            # shuffle_loss = shuffle_criterion(shuffle_predicts)

            predicted = torch.max(predicts, dim=1).indices.cpu() 
            running_acc.update((predicted == answer_id).sum().item() / N, N)

        loss = 0
        csl_loss = 0
        if "vqa" in args.loss_list:
            loss += vqa_loss 
        if "csl" in args.loss_list:
            if "qv" in args.query_list:
                loss += qv_csl_loss
                csl_loss += qv_csl_loss
            if "av" in args.query_list:
                loss += av_csl_loss
                csl_loss += av_csl_loss
            if "qav" in args.query_list:
                loss += qav_csl_loss
                csl_loss += qav_csl_loss
        if "shuffle" in args.loss_list:
            loss += shuffle_loss * args.lambda_sf

        optimizer.zero_grad()
        loss.backward()
        if args.clip:
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.clip)
        optimizer.step()
        scheduler.step()

        running_vqa_loss.update(vqa_loss.detach().cpu().item(), N)
        if "csl" in args.loss_list:
            running_csl_loss.update(csl_loss.detach().cpu().item(), N)
        if "shuffle" in args.loss_list:
            running_shuffle_loss.update(shuffle_loss.detach().cpu().item(), N)
        if args.mlm_prob:
            running_mlm_loss.update(mlm_loss.detach().cpu().item(), N)
        if (i + 1) % (len(train_loader) // args.freq_display) == 0:
            if args.mlm_prob:
                logging.info(
                    f"Epoch {epoch + 1}/{args.epochs}, Lr:{optimizer.param_groups[0]['lr']}, Progress: {float(i + 1) / len(train_loader):.4f}, VQA loss: "
                    f"{running_vqa_loss.avg:.4f}, CSL loss: {running_csl_loss.avg:.2%}, MLM loss: {running_mlm_loss.avg:.4f}"
                )
            else:
                if "csl" in args.loss_list:
                    logging.info(
                        f"Epoch {epoch + 1}/{args.epochs}, Lr:{optimizer.param_groups[0]['lr']}, Progress: {float(i + 1) / len(train_loader):.4f}, VQA loss: "
                        f"{running_vqa_loss.avg:.4f}, CSL loss: {running_csl_loss.avg:.2%}, Train acc: {running_acc.avg:.2%}"
                    )
                elif "shuffle" in args.loss_list:
                    logging.info(
                        f"Epoch {epoch + 1}/{args.epochs}, Lr:{optimizer.param_groups[0]['lr']}, Progress: {float(i + 1) / len(train_loader):.4f}, VQA loss: "
                        f"{running_vqa_loss.avg:.4f}, shuffle loss: {running_shuffle_loss.avg:.2%}, Train acc: {running_acc.avg:.2%}"
                    )         
                else:
                    logging.info(
                        f"Epoch {epoch + 1}/{args.epochs}, Lr:{optimizer.param_groups[0]['lr']}, Progress: {float(i + 1) / len(train_loader):.4f}, VQA loss: "
                        f"{running_vqa_loss.avg:.4f}, Train acc: {running_acc.avg:.2%}"
                    )
            running_acc.reset()
            running_vqa_loss.reset()
            if "csl" in args.loss_list:
                running_csl_loss.reset()
            if "shuffle" in args.loss_list:
                running_shuffle_loss.reset()
            running_mlm_loss.reset()
