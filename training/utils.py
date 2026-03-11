import torch
from torch.functional import F
import config
import torch.nn as nn
# from mlx.misc import unsqueeze
from tools import patcher

criterion = nn.BCEWithLogitsLoss()


@torch.no_grad()
def eval_cre(model, dataloader, run):
    sum_recon_loss, sum_class_loss, sum_acc, num_inputs = 0, 0, 0, 0
    for batch_id, (img, target) in iter(enumerate(dataloader)):
        img, target = img.to(config.DEVICE), target.to(config.DEVICE)
        recon, logit = model(img)
        class_loss = F.cross_entropy(input=logit, target=target)
        recon_loss = F.mse_loss(input=dataloader.dataset.transform.transforms[-1](recon), target=img)
        acc = simple_accuracy(logits=logit, target=target)
        num_inputs += img.shape[0]
        sum_class_loss += class_loss * img.shape[0]
        sum_recon_loss += recon_loss * img.shape[0]
        sum_acc += acc * img.shape[0]
    if run is not None:
        run["series/val/class_loss"].append(sum_class_loss / num_inputs)
        run["series/val/acc"].append(sum_acc / num_inputs)
        run["series/val/recon_loss"].append(sum_recon_loss / num_inputs)
    return sum_class_loss / num_inputs


@torch.no_grad()
def eval_recon_cre(model, dataloader, run):
    sum_recon_loss, sum_class_loss, sum_acc, num_inputs = 0, 0, 0, 0
    for batch_id, (img, target) in iter(enumerate(dataloader)):
        img, target = img.to(config.DEVICE), target.to(config.DEVICE)
        recon = model(img)
        recon_loss = F.mse_loss(input=dataloader.dataset.transform.transforms[-1](recon), target=img)
        num_inputs += img.shape[0]
        sum_recon_loss += recon_loss * img.shape[0]
    if run is not None:
        run["series/val/recon_loss"].append(sum_recon_loss / num_inputs)
    return sum_class_loss / num_inputs


@torch.no_grad()
def eval_class_mamba(model, env, seq_len, target_size, n_eval_envs=1, run=None):
    sum_recon_loss, sum_class_loss, sum_acc, n_evals = 0, 0, 0, 0
    model.eval()
    for _ in range(n_eval_envs):
        patches, _, move_x, move_y, class_labels = env.reset(model.foveation_properties)
        seq, pos_x, pos_y = env.get_rnd_seq(model.foveation_properties, seq_len)
        _, logits, recon = model.forward_seq(seq, pos_x, pos_y, None, None, forward_full_seq=False)
        class_loss = F.cross_entropy(input=logits[:, -1], target=class_labels)
        acc = simple_accuracy(logits=logits[:, -1], target=class_labels)
        recon_loss = F.mse_loss(input=env.normalize(recon[:, -1]), target=env.get_recons_target(size=target_size))
        sum_class_loss += class_loss
        sum_recon_loss += recon_loss
        sum_acc += acc
        n_evals += 1
    if run:
        run["series/val/class_loss"].append(sum_class_loss / n_evals)
        run["series/val/acc"].append(sum_acc / n_evals)
        run["series/val/recon_loss"].append(sum_recon_loss / n_evals)
    return sum_class_loss.item() / n_evals


@torch.no_grad()
def eval_class_vit(model, dataloader, seq_lens, run=None):
    sum_acc, sum_ce, n_inputs = torch.zeros(len(seq_lens)).to(config.DEVICE), torch.zeros(len(seq_lens)).to(
        config.DEVICE), 0
    model.eval()
    for batch_id, (img, target) in enumerate(dataloader):
        img, target = img.to(config.DEVICE, non_blocking=True), target.to(config.DEVICE, non_blocking=True)
        for i in range(len(seq_lens)):
            seq, pos_x, pos_y = patcher.get_seq(fov_props=model.foveation_properties, img=img, seq_len=seq_lens[i],
                                                seq_type='rnd')
            logit = model.forward_seq(seq, pos_x, pos_y)
            sum_acc[i] += simple_accuracy(logits=logit, target=target) * img.shape[0]
            sum_ce[i] += F.cross_entropy(input=logit, target=target) * img.shape[0]
        n_inputs += img.shape[0]
    sum_ce /= n_inputs
    sum_acc /= n_inputs
    if run:
        run["series/val/loss"].append(sum_ce[-1])
        for i in range(len(seq_lens)):
            run["series/val/ce_loss_" + str(seq_lens[i])].append(sum_ce[i])
            run["series/val/acc_" + str(seq_lens[i] + 1)].append(sum_acc[i])
    return sum_ce[-1]


@torch.no_grad()
def eval_class_vit_rel_only(model, dataloader, seq_type='relevant_only', run=None):
    sum_acc, sum_ce, n_inputs = 0, 0, 0
    model.eval()
    for batch_id, (img, target) in enumerate(dataloader):
        img, target = img.to(config.DEVICE, non_blocking=True), target.to(config.DEVICE, non_blocking=True)
        seq, pos_x, pos_y = patcher.get_seq(fov_props=model.foveation_properties, img=img, seq_type=seq_type)
        logit = model.forward_seq(seq, pos_x, pos_y)
        sum_acc += simple_accuracy(logits=logit, target=target) * img.shape[0]
        sum_ce += F.cross_entropy(input=logit, target=target) * img.shape[0]
        n_inputs += img.shape[0]
    sum_ce /= n_inputs
    sum_acc /= n_inputs
    if run:
        run["series/val/loss"].append(sum_ce)
        run["series/val/acc"].append(sum_acc)
    return sum_ce


@torch.no_grad()
def eval_vit(model, dataloader, seq_lens, run=None, best_loss_len=None, sub_img_size=None):
    sum_recon_errors, n_inputs = torch.zeros(len(seq_lens)).to(config.DEVICE), 0
    model.eval()
    for batch_id, (img, target) in enumerate(dataloader):
        img, target = img.to(config.DEVICE, non_blocking=True), target.to(config.DEVICE, non_blocking=True)
        for i in range(len(seq_lens)):
            seq, pos_x, pos_y = patcher.get_seq(fov_props=model.foveation_properties, img=img, seq_len=seq_lens[i],
                                                seq_type='rnd', sub_img_size=sub_img_size)
            recon = model.forward_seq(seq, pos_x, pos_y)
            sum_recon_errors[i] += F.mse_loss(input=dataloader.dataset.transform.transforms[-1](recon), target=img) * \
                                   img.shape[0]
        n_inputs += img.shape[0]
        if n_inputs > 5000:
            break  # to speed up things
    sum_recon_errors /= n_inputs
    best_loss_idx = seq_lens.index(best_loss_len) if best_loss_len is not None else - 1
    if run:
        run["series/val/loss"].append(sum_recon_errors[best_loss_idx].item())
        for i in range(len(seq_lens)):
            run["series/val/recon_loss_" + str(seq_lens[i])].append(sum_recon_errors[i].item())
    return sum_recon_errors[best_loss_idx].item()


@torch.no_grad()
def eval_seq_img_mamba(model, dataloader, seq_len, query_len, run=None, num_eval_inputs=5000):
    sum_recon_errors, n_inputs = 0, 0
    model.eval()
    for batch_id, (img, target) in enumerate(dataloader):
        img, target = img.to(config.DEVICE, non_blocking=True), target.to(config.DEVICE, non_blocking=True)
        seq, pos_x, pos_y = patcher.get_seq(fov_props=model.foveation_properties, img=img, seq_len=seq_len, seq_type='rnd')
        query_seq, query_pos_x, query_pos_y = patcher.get_seq(fov_props=model.foveation_properties, img=img,
                                                              seq_len=query_len, seq_type='rnd')
        recon = model.forward_seq(seq=seq, pos_x=pos_x, pos_y=pos_y, query_pos_x=query_pos_x, query_pos_y=query_pos_y)
        loss = F.mse_loss(input=dataloader.dataset.transform.transforms[-1](recon), target=query_seq[0])
        sum_recon_errors += loss.item() * img.shape[0]
        n_inputs += img.shape[0]
        if n_inputs > num_eval_inputs:  # speed up evaluating
            break
    sum_recon_errors /= n_inputs
    if run:
        run["series/val/loss"].append(sum_recon_errors)
    return sum_recon_errors


@torch.no_grad()
def eval_seq_img_mamba_state_passing(model, dataloader, seq_len, query_len, run=None, num_eval_inputs=5000):
    sum_recon_errors, n_inputs = 0, 0
    model.eval()
    for batch_id, (img, target) in enumerate(dataloader):
        img, target = img.to(config.DEVICE, non_blocking=True), target.to(config.DEVICE, non_blocking=True)
        seq, pos_x, pos_y = patcher.get_seq(fov_props=model.foveation_properties, img=img, seq_len=seq_len,
                                            seq_type='rnd')
        query_seq, query_pos_x, query_pos_y = patcher.get_seq(fov_props=model.foveation_properties, img=img,
                                                              seq_len=query_len, seq_type='rnd')
        recon, _, _ = model.forward_seq(seq=seq, pos_x=pos_x, pos_y=pos_y, query_pos_x=query_pos_x, query_pos_y=query_pos_y)
        loss = F.mse_loss(input=dataloader.dataset.transform.transforms[-1](recon), target=query_seq[0])
        sum_recon_errors += loss.item() * img.shape[0]
        n_inputs += img.shape[0]
        if n_inputs > num_eval_inputs:  # speed up evaluating
            break
    sum_recon_errors /= n_inputs
    if run:
        run["series/val/loss"].append(sum_recon_errors)
    return sum_recon_errors


@torch.no_grad()
def eval_tokens_to_rem_mamba(model, dataloader, seq_len, query_len, tokens_to_remember, run=None, num_eval_inputs=5000):
    sum_recon_errors, n_inputs = 0, 0
    model.eval()
    for batch_id, (img, target) in enumerate(dataloader):
        img, target = img.to(config.DEVICE, non_blocking=True), target.to(config.DEVICE, non_blocking=True)
        seq, pos_x, pos_y = patcher.get_seq(fov_props=model.foveation_properties, img=img, seq_len=seq_len,
                                            seq_type='rnd')
        perm = torch.randperm(min(tokens_to_remember, pos_y.shape[1]))

        query_seq, query_pos_x, query_pos_y = [seq[0][:, -tokens_to_remember:][:, perm]], pos_x[:, -tokens_to_remember:][:, perm], pos_y[ :, -tokens_to_remember:][:, perm]
        # query_seq, query_pos_x, query_pos_y = patcher.get_seq(fov_props=model.foveation_properties, img=img,
        #                                                       seq_len=query_len, seq_type='rnd')
        recon = model.forward_seq(seq=seq, pos_x=pos_x, pos_y=pos_y, query_pos_x=query_pos_x, query_pos_y=query_pos_y)
        loss = F.mse_loss(input=dataloader.dataset.transform.transforms[-1](recon), target=query_seq[0])
        sum_recon_errors += loss.item() * img.shape[0]
        n_inputs += img.shape[0]
        if n_inputs > num_eval_inputs:  # speed up evaluating
            break
    sum_recon_errors /= n_inputs
    if run:
        run["series/val/loss"].append(sum_recon_errors)
    return sum_recon_errors


@torch.no_grad()
def eval_seq_img_mamba_reg(model, dataloader, seq_len, query_len, run=None, num_eval_inputs=5000):
    sum_recon_errors, n_inputs = 0, 0
    model.eval()
    for batch_id, (img, target) in enumerate(dataloader):
        img, target = img.to(config.DEVICE, non_blocking=True), target.to(config.DEVICE, non_blocking=True)
        seq, pos_x, pos_y = patcher.get_seq(fov_props=model.foveation_properties, img=img, seq_len=seq_len,
                                            seq_type='rnd')
        query_seq, query_pos_x, query_pos_y = patcher.get_seq(fov_props=model.foveation_properties, img=img,
                                                              seq_len=query_len, seq_type='rnd')
        recon, _ = model.forward_seq(seq=seq, pos_x=pos_x, pos_y=pos_y, query_pos_x=query_pos_x,
                                     query_pos_y=query_pos_y)
        loss = F.mse_loss(input=dataloader.dataset.transform.transforms[-1](recon), target=query_seq[0])
        sum_recon_errors += loss.item() * img.shape[0]
        n_inputs += img.shape[0]
        if n_inputs > num_eval_inputs:  # speed up evaluating
            break
    sum_recon_errors /= n_inputs
    if run:
        run["series/val/loss"].append(sum_recon_errors)
    return sum_recon_errors


@torch.no_grad()
def eval_seq_img_mamba_reg_2(model, dataloader, seq_len, query_len, run=None, num_eval_inputs=5000):
    sum_recon_errors, n_inputs = 0, 0
    model.eval()
    for batch_id, (img, target) in enumerate(dataloader):
        img, target = img.to(config.DEVICE, non_blocking=True), target.to(config.DEVICE, non_blocking=True)
        seq, pos_x, pos_y = patcher.get_seq(fov_props=model.foveation_properties, img=img, seq_len=seq_len,
                                            seq_type='rnd')
        query_seq, query_pos_x, query_pos_y = patcher.get_seq(fov_props=model.foveation_properties, img=img,
                                                              seq_len=query_len, seq_type='rnd')
        recon, _, _ = model.forward_seq(seq=seq, pos_x=pos_x, pos_y=pos_y, query_pos_x=query_pos_x,
                                        query_pos_y=query_pos_y)
        loss = F.mse_loss(input=dataloader.dataset.transform.transforms[-1](recon), target=query_seq[0])
        sum_recon_errors += loss.item() * img.shape[0]
        n_inputs += img.shape[0]
        if n_inputs > num_eval_inputs:  # speed up evaluating
            break
    sum_recon_errors /= n_inputs
    if run:
        run["series/val/loss"].append(sum_recon_errors)
    return sum_recon_errors


@torch.no_grad()
def eval_seq_img_vit(model, dataloader, seq_len, query_len, run=None, num_eval_inputs=5000):
    sum_recon_errors, n_inputs = 0, 0
    model.eval()
    for batch_id, (img, target) in enumerate(dataloader):
        img, target = img.to(config.DEVICE, non_blocking=True), target.to(config.DEVICE, non_blocking=True)
        seq, pos_x, pos_y = patcher.get_seq(fov_props=model.foveation_properties, img=img, seq_len=seq_len,
                                            seq_type='rnd')
        query_seq, query_pos_x, query_pos_y = patcher.get_seq(fov_props=model.foveation_properties, img=img,
                                                              seq_len=query_len, seq_type='rnd')
        recon = model.forward_seq(seq, pos_x, pos_y, query_pos_x, query_pos_y)
        loss = F.mse_loss(input=dataloader.dataset.transform.transforms[-1](recon), target=query_seq[0])
        sum_recon_errors += loss.item() * img.shape[0]
        n_inputs += img.shape[0]
        if n_inputs > num_eval_inputs:  # speed up evaluating
            break
    sum_recon_errors /= n_inputs
    if run:
        run["series/val/loss"].append(sum_recon_errors)
    return sum_recon_errors


@torch.no_grad()
def eval_simple_recons_mamba(model, dataloader, seq_lens, recon_mode='full_img', best_loss_len=None, run=None,
                             sub_img_size=None, num_eval_inputs=5000):
    sum_recon_errors, n_inputs = 0, 0
    model.eval()
    for batch_id, (img, target) in enumerate(dataloader):
        img, target = img.to(config.DEVICE, non_blocking=True), target.to(config.DEVICE, non_blocking=True)
        seq, pos_x, pos_y = patcher.get_seq(fov_props=model.foveation_properties, img=img, seq_len=seq_lens[-1] + 1,
                                            seq_type='rnd', sub_img_size=sub_img_size)
        output_pos_x, output_pos_y = None, None
        if recon_mode == 'pixel_wise':
            pixels, output_pos_x, output_pos_y = get_patched_pixel_wise_img(img)
            img = pixels
        _, _, recon = model.forward_seq(seq, pos_x, pos_y, output_pos_x, output_pos_y, forward_full_seq=False,
                                        forward_indices=seq_lens)
        target = img.unsqueeze(1).expand_as(recon)
        recon_errors = F.mse_loss(input=dataloader.dataset.transform.transforms[-1](recon), target=target,
                                  reduction='none')
        mean_recon_errors = torch.mean(recon_errors, dim=(0, 2, 3, 4))
        sum_recon_errors += mean_recon_errors * img.shape[0]
        n_inputs += img.shape[0]
        if n_inputs > num_eval_inputs:  # speed up evaluating
            break
    sum_recon_errors /= n_inputs
    best_loss_idx = seq_lens.index(best_loss_len - 1) if best_loss_len is not None else - 1
    if run:
        run["series/val/loss"].append(sum_recon_errors[best_loss_idx].item())
        for i in range(len(seq_lens)):
            run["series/val/recon_loss_" + str(seq_lens[i] + 1)].append(sum_recon_errors[i].item())
    return sum_recon_errors[best_loss_idx].item()


@torch.no_grad()
def eval_ssm_state_mamba(model, dataloader, seq_lens, best_loss_len=None, run=None, sub_img_size=None,
                         num_eval_inputs=5000, *args, **kwargs):
    sum_recon_errors, n_inputs = torch.zeros(len(seq_lens)).to(config.DEVICE), 0
    model.eval()
    for batch_id, (img, target) in enumerate(dataloader):
        img, target = img.to(config.DEVICE, non_blocking=True), target.to(config.DEVICE, non_blocking=True)
        for i in range(len(seq_lens)):
            seq, pos_x, pos_y = patcher.get_seq(fov_props=model.foveation_properties, img=img, seq_len=seq_lens[i],
                                                seq_type='rnd', sub_img_size=sub_img_size)
            _, _, recon = model.forward_seq(seq, pos_x, pos_y)
            sum_recon_errors[i] += F.mse_loss(input=dataloader.dataset.transform.transforms[-1](recon), target=img) * \
                                   img.shape[0]
        n_inputs += img.shape[0]
        if n_inputs > num_eval_inputs:  # speed up evaluating
            break
    sum_recon_errors /= n_inputs
    best_loss_idx = seq_lens.index(best_loss_len) if best_loss_len is not None else - 1
    if run:
        run["series/val/loss"].append(sum_recon_errors[best_loss_idx].item())
        for i in range(len(seq_lens)):
            run["series/val/recon_loss_" + str(seq_lens[i])].append(sum_recon_errors[i].item())
    return sum_recon_errors[best_loss_idx].item()


@torch.no_grad()
def eval_simple_recons_mamba_2(model, dataloader, seq_lens, recon_mode='full_img', best_loss_len=None, run=None,
                               sub_img_size=None):
    sum_recon_errors, n_inputs = 0, 0
    model.eval()
    for batch_id, (img, target) in enumerate(dataloader):
        img, target = img.to(config.DEVICE, non_blocking=True), target.to(config.DEVICE, non_blocking=True)
        seq, pos_x, pos_y = patcher.get_seq(fov_props=model.foveation_properties, img=img, seq_len=seq_lens[-1] + 1,
                                            seq_type='rnd', sub_img_size=sub_img_size)
        output_pos_x, output_pos_y = None, None
        if recon_mode == 'pixel_wise':
            pixels, output_pos_x, output_pos_y = get_patched_pixel_wise_img(img)
            img = pixels
        _, _, recon = model.forward_seq(seq, pos_x, pos_y, output_pos_x, output_pos_y, forward_full_seq=False,
                                        forward_indices=seq_lens)
        target = img.unsqueeze(1).expand_as(recon)
        recon_errors = F.mse_loss(input=dataloader.dataset.transform.transforms[-1](recon), target=target,
                                  reduction='none')
        mean_recon_errors = torch.mean(recon_errors, dim=(0, 2, 3, 4))
        sum_recon_errors += mean_recon_errors * img.shape[0]
        n_inputs += img.shape[0]
    sum_recon_errors /= n_inputs
    best_loss_idx = seq_lens.index(best_loss_len - 1) if best_loss_len is not None else - 1
    if run:
        run["series/val/loss"].append(sum_recon_errors[best_loss_idx].item())
        for i in range(len(seq_lens)):
            run["series/val/recon_loss_" + str(seq_lens[i] + 1)].append(sum_recon_errors[i].item())
    return sum_recon_errors[best_loss_idx].item()


@torch.no_grad()
def eval_simple_class_mamba(model, dataloader, seq_lens, recon_mode='full_img', run=None):
    sum_ce_errors, sum_accs, n_inputs = 0, 0, 0
    model.eval()
    for batch_id, (img, target) in enumerate(dataloader):
        img, target = img.to(config.DEVICE, non_blocking=True), target.to(config.DEVICE, non_blocking=True)
        seq, pos_x, pos_y = patcher.get_seq(fov_props=model.foveation_properties, img=img, seq_len=seq_lens[-1] + 1,
                                            seq_type='rnd')
        output_pos_x, output_pos_y = None, None
        if recon_mode == 'pixel_wise':
            pixels, output_pos_x, output_pos_y = get_patched_pixel_wise_img(img)
            img = pixels
        _, logit, _ = model.forward_seq(seq, pos_x, pos_y, output_pos_x, output_pos_y, forward_full_seq=False,
                                        forward_indices=seq_lens)
        ces = F.cross_entropy(input=logit.permute(0, 2, 1), target=target.unsqueeze(1).expand(-1, logit.shape[1]),
                              reduction='none')
        mean_ces = torch.mean(ces, dim=(0,))
        sum_ce_errors += mean_ces * img.shape[0]
        accs = accuracy_over_seq(logits=logit, target=target)
        sum_accs += accs * img.shape[0]
        n_inputs += img.shape[0]
    sum_ce_errors /= n_inputs
    sum_accs /= n_inputs
    if run:
        run["series/val/loss"].append(sum_ce_errors[-1])
        for i in range(len(seq_lens)):
            run["series/val/ce_loss_" + str(seq_lens[i] + 1)].append(sum_ce_errors[i])
            run["series/val/acc_" + str(seq_lens[i] + 1)].append(sum_accs[i])
    return sum_ce_errors[-1]


@torch.no_grad()
def eval_simple_class_mamba_rel_only(model, dataloader, recon_mode='full_img', seq_type='relevant_only', run=None):
    sum_ce, sum_acc, n_inputs = 0, 0, 0
    model.eval()
    for batch_id, (img, target) in enumerate(dataloader):
        img, target = img.to(config.DEVICE, non_blocking=True), target.to(config.DEVICE, non_blocking=True)
        seq, pos_x, pos_y = patcher.get_seq(fov_props=model.foveation_properties, img=img, seq_type=seq_type)
        output_pos_x, output_pos_y = None, None
        _, logit, _ = model.forward_seq(seq, pos_x, pos_y, output_pos_x, output_pos_y, forward_full_seq=False)
        sum_acc += simple_accuracy(logits=logit[:, -1], target=target) * img.shape[0]
        sum_ce += F.cross_entropy(input=logit[:, -1], target=target) * img.shape[0]
        n_inputs += img.shape[0]
    sum_ce /= n_inputs
    sum_acc /= n_inputs
    if run:
        run["series/val/loss"].append(sum_ce)
        run["series/val/acc"].append(sum_acc)
    return sum_ce


@torch.no_grad()
def eval_class_vit_rel_only(model, dataloader, seq_type='relevant_only', run=None):
    sum_acc, sum_ce, n_inputs = 0, 0, 0
    model.eval()
    for batch_id, (img, target) in enumerate(dataloader):
        img, target = img.to(config.DEVICE, non_blocking=True), target.to(config.DEVICE, non_blocking=True)
        seq, pos_x, pos_y = patcher.get_seq(fov_props=model.foveation_properties, img=img, seq_type=seq_type)
        logit = model.forward_seq(seq, pos_x, pos_y)
        sum_acc += simple_accuracy(logits=logit, target=target) * img.shape[0]
        sum_ce += F.cross_entropy(input=logit, target=target) * img.shape[0]
        n_inputs += img.shape[0]
    sum_ce /= n_inputs
    sum_acc /= n_inputs
    if run:
        run["series/val/loss"].append(sum_ce)
        run["series/val/acc"].append(sum_acc)
    return sum_ce


@torch.no_grad()
def eval_recons_mamba(model, env, target_size, seq_len, recon_mode, n_eval_envs=1, run=None, num_rnd_positions=None,
                      sample_mode=None):
    sum_loss, n_evals = 0, 0
    model.eval()
    for _ in range(n_eval_envs):
        patches, _, move_x, move_y, class_labels = env.reset(model.foveation_properties)
        # target = env.get_patched_img(fov_props=model.foveation_properties)
        # target = env.get_recons_target(size=target_size)
        if recon_mode == 'pixel_wise':
            patches, output_pos_x, output_pos_y = env.get_patched_pixel_wise_img()
        elif recon_mode == 'patch_wise':
            patches, output_pos_x, output_pos_y = env.get_patched_img(model.foveation_properties,
                                                                      num_rnd_positions=num_rnd_positions,
                                                                      sample_mode=sample_mode)
        elif recon_mode == 'full_img':
            patches = env.get_recons_target(size=target_size)
            output_pos_x, output_pos_y = None, None
        else:
            raise NameError(f'Unknown recon_mode {recon_mode}')
        seq, pos_x, pos_y = env.get_rnd_seq(model.foveation_properties, seq_len)
        _, _, recon = model.forward_seq(seq, pos_x, pos_y, output_pos_x, output_pos_y, forward_full_seq=False)
        loss = F.mse_loss(input=env.normalize(recon[:, -1]), target=patches)
        sum_loss += loss
        n_evals += 1
    if run:
        run["series/val/loss"].append(sum_loss / n_evals)
    return sum_loss.item() / n_evals


@torch.no_grad()
def eval_foveated_mamba_4(model, n_iterations, env, beta, gamma, n_eval_envs=1, run=None):
    sum_rl_loss, total_sum_entropy, sum_returns, sum_loss, n_evals = 0, 0, 0, 0, 0
    model.eval()
    for _ in range(n_eval_envs):
        patches, _, move_x, move_y, class_labels = env.reset(model.foveation_properties)
        cache = model.initialize(patches[0])
        l_log_probs, rewards, sum_entropy, = [], [], 0
        for _ in range(n_iterations):
            cache, (action_logit, value, class_logit) = model.step(patches, cache, move_x, move_y)
            actions, shift_x, shift_y, log_prob, entropy = model.get_action(action_logit, max_action=True)
            sum_entropy += entropy / n_iterations
            l_log_probs.append(log_prob)
            patches, reward, move_x, move_y, class_labels = env.step(shift_x, shift_y, model.foveation_properties)
            rewards.append(reward)

        returns = compute_simple_return(rewards=rewards, gamma=gamma, normalize=False)
        log_probs = torch.stack(l_log_probs, dim=1)
        rl_loss = (returns.detach() * -log_probs).mean()
        loss = rl_loss - beta * sum_entropy.mean()
        total_sum_entropy += sum_entropy.mean()
        sum_rl_loss += rl_loss
        sum_loss += loss
        sum_returns += returns.mean()
        n_evals += 1
    if run:
        run["series/val/loss"].append(-(sum_loss / n_evals))
        run["series/val/rl_loss"].append(sum_rl_loss / n_evals)
        run["series/val/returns"].append(sum_returns / n_evals)
        run["series/val/entropy"].append(total_sum_entropy / n_evals)

    return -(sum_loss / n_evals).item()


@torch.no_grad()
def eval_foveated_mamba_3(model, n_iterations, env, beta, gamma, n_eval_envs=1, run=None):
    sum_rl_loss, total_sum_entropy, sum_returns, sum_loss, n_evals = 0, 0, 0, 0, 0
    model.eval()
    for _ in range(n_eval_envs):
        patches, _, move_x, move_y, class_labels = env.reset(model.foveation_properties)
        cache = model.initialize(patches[0])
        l_log_probs, rewards, sum_entropy, = [], [], 0
        for _ in range(n_iterations):
            _, action_logits, cache = model.step(patches, cache, move_x, move_y)
            actions, shift_x, shift_y, log_prob, entropy = model.get_action(action_logits, max_action=True)
            sum_entropy += entropy / n_iterations
            l_log_probs.append(log_prob)
            patches, reward, move_x, move_y, class_labels = env.step(shift_x, shift_y, model.foveation_properties)
            rewards.append(reward)

        returns = compute_simple_return(rewards=rewards, gamma=gamma, normalize=False)
        log_probs = torch.stack(l_log_probs, dim=1)
        rl_loss = (returns.detach() * -log_probs).mean()
        loss = rl_loss - beta * sum_entropy.mean()
        total_sum_entropy += sum_entropy.mean()
        sum_rl_loss += rl_loss
        sum_loss += loss
        sum_returns += returns.mean()
        n_evals += 1
    if run:
        run["series/val/loss"].append(-(sum_loss / n_evals))
        run["series/val/rl_loss"].append(sum_rl_loss / n_evals)
        run["series/val/returns"].append(sum_returns / n_evals)
        run["series/val/entropy"].append(total_sum_entropy / n_evals)

    return -(sum_loss / n_evals).item()


@torch.no_grad()
def eval_foveated_mamba_2(model, n_iterations, env, beta, gamma, n_eval_envs=1, run=None):
    sum_rl_loss, total_sum_entropy, sum_returns, sum_loss, n_evals = 0, 0, 0, 0, 0
    model.eval()
    for _ in range(n_eval_envs):
        patches, _, move_x, move_y = env.reset(model.foveation_properties)
        cache = model.initialize(patches[0])
        l_log_probs, rewards, sum_entropy = [], [], 0
        for _ in range(n_iterations):
            _, action_logits, cache = model.step(patches, cache, move_x, move_y)
            actions, shift_x, shift_y, log_prob, entropy = model.get_action(action_logits, max_action=True)
            sum_entropy += entropy / n_iterations
            l_log_probs.append(log_prob)
            patches, reward, move_x, move_y = env.step(shift_x, shift_y, model.foveation_properties)
            rewards.append(reward)
        returns = compute_simple_return(rewards=rewards, gamma=gamma, normalize=False)

        log_probs = torch.stack(l_log_probs, dim=1)
        rl_loss = (returns.detach() * -log_probs).mean()
        loss = rl_loss - beta * sum_entropy.mean()
        total_sum_entropy += sum_entropy.mean()
        sum_rl_loss += rl_loss
        sum_loss += loss
        sum_returns += returns.mean()
        n_evals += 1
    if run:
        run["series/val/loss"].append(-(sum_loss / n_evals))
        run["series/val/rl_loss"].append(sum_rl_loss / n_evals)
        run["series/val/returns"].append(sum_returns / n_evals)
        run["series/val/entropy"].append(total_sum_entropy / n_evals)
    return -(sum_loss / n_evals).item()


def eval_img_mamba(model, dataloader, seq_type, seq_len_range=None, run=None, continuous=False):
    model.eval()
    sum_loss, sum_acc, num_inputs = 0, 0, 0
    with torch.no_grad():
        for batch_id, (tensor, target) in enumerate(dataloader):
            tensor, target = tensor.to(config.DEVICE, non_blocking=True), target.to(config.DEVICE, non_blocking=True)
            if seq_type == 'random':
                class_logit, pos_logit = model.forward_as_rnd_seq(tensor, seq_len_range=seq_len_range)
            elif seq_type == 'scan_path':
                class_logit, pos_logit = model.forward_as_scan_path_seq(tensor)
            else:
                raise NameError(f'Unknown sequence type {seq_type}')
            if continuous:
                loss = F.cross_entropy(input=class_logit.permute(0, 2, 1),
                                       target=target.unsqueeze(1).repeat(1, class_logit.shape[1]))
            else:
                loss = F.cross_entropy(input=class_logit[:, -1, :], target=target)
            acc = accuracy(logits=class_logit, target=target)
            sum_loss += loss * len(tensor)
            num_inputs += len(tensor)
            sum_acc += acc * len(tensor)
    if run:
        run["series/val/loss"].append(sum_loss / num_inputs)
        run["series/val/accuracy"].append(sum_acc / num_inputs)
    return (sum_loss / num_inputs).item()


def eval_mamba_agent(model, dataloader, n_iterations, run=None, continuous=True):
    model.eval()
    sum_loss, sum_acc, num_inputs = 0, 0, 0
    with torch.no_grad():
        for batch_id, (tensor, target) in enumerate(dataloader):
            tensor, target = tensor.to(config.DEVICE, non_blocking=True), target.to(config.DEVICE, non_blocking=True)
            class_logit, fix_logit = model.forward_iterative(tensor, n_iterations)
            if continuous:
                loss = F.cross_entropy(input=class_logit.permute(0, 2, 1),
                                       target=target.unsqueeze(1).repeat(1, class_logit.shape[1]))
            else:
                loss = F.cross_entropy(input=class_logit[:, -1, :], target=target)
            acc = accuracy(logits=class_logit, target=target)
            sum_loss += loss * len(tensor)
            num_inputs += len(tensor)
            sum_acc += acc * len(tensor)
    if run:
        run["series/val/loss"].append(sum_loss / num_inputs)
        run["series/val/accuracy"].append(sum_acc / num_inputs)
    return (sum_loss / num_inputs).item()


def simple_accuracy(logits, target):
    pred = logits.argmax(dim=1, keepdim=True)
    e = pred.eq(target.view_as(pred)).sum() / target.shape[0]
    return e


def accuracy(logits, target):
    logits = logits[:, -1, :]
    pred = logits.argmax(dim=1, keepdim=True)
    e = pred.eq(target.view_as(pred)).sum() / target.shape[0]
    return e


def accuracy_over_seq(logits, target):
    pred = logits.argmax(dim=2, keepdim=True)
    target = target.unsqueeze(1).repeat(1, pred.shape[1]).view_as(pred)
    e = torch.sum(pred.eq(target).squeeze(2), dim=0) / target.shape[0]
    return e


def collate_as_sequence(batch, to_seq, *to_seq_args, **to_seq_kwargs):
    img, label = zip(*batch)
    tensor = torch.stack(img, dim=0)
    label = torch.tensor(label)
    foveated_tensor = to_seq(batch=tensor, *to_seq_args, **to_seq_kwargs)
    return foveated_tensor, label


def compute_simple_return(rewards, gamma=0.99, normalize=True):
    returns = []
    R = 0
    for reward in reversed(rewards):
        R = reward + gamma * R
        returns.insert(0, R)
    returns = torch.stack(returns, dim=1)
    if normalize:
        returns = (returns - returns.mean()) / (returns.std() + 1e-9)  # normalize
    return returns


def compute_returns(l_class_logits, target, criterion, gamma=1.0):
    returns = []
    R = 0
    for class_logit in reversed(l_class_logits):
        R = F.cross_entropy(input=class_logit, target=target, reduction='none') + gamma * R  # just take negative CE
        returns.insert(0, R)
    returns = -torch.stack(returns, dim=1)
    # returns = (returns - returns.mean()) / (returns.std() + 1e-9)  # normalize
    return returns


def get_patched_pixel_wise_img(img):
    x_pos = torch.arange(start=0, end=img.shape[-1]).to(config.DEVICE)
    y_pos = torch.arange(start=0, end=img.shape[-2]).to(config.DEVICE)
    cartesian = torch.cartesian_prod(x_pos, y_pos)
    x_pos, y_pos = cartesian[:, 1], cartesian[:, 0]
    x_pos, y_pos = abs_pos_to_rel_pos(img, x_pos, y_pos)
    patches = img.flatten(start_dim=-2).permute(0, 2, 1).unsqueeze(-1).unsqueeze(-1)
    # patches = patches.reshape(*patches.shape, 1, 1, 1)
    # for debugging
    # for k in range(900):
    #     ax_pos = x_pos[k]
    #     ay_pos = y_pos[k]
    #     a1 = patches[0][0][k]
    #     a2 = self.img[0][0][ay_pos + 15 + 2][ax_pos + 15 + 2]
    #     print(a1 + a2)
    return patches, x_pos, y_pos


def abs_pos_to_rel_pos(img, abs_x, abs_y):
    origin_x, origin_y = img.shape[-1] // 2, img.shape[-2] // 2
    rel_x, rel_y = abs_x - origin_x, abs_y - origin_y
    return rel_x, rel_y


if __name__ == '__main__':
    print("stupid test")
