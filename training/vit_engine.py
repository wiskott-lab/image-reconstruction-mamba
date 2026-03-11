import tools.patcher as patcher
import config
from training.utils import eval_seq_img_vit
import tools.neptune_utils as nu
import random
from torch.functional import F
from tools.checkpoint_utils import generate_checkpoint
from tools.model_utils import generate_model_state


def train(model, epochs, dataloader_train, dataloader_val, optimizer, loss_scaler, val_step, train_step, best_loss,
          lr_scheduler, run, seq_len_range, seq_len_eval, query_len_range, eval_query_len, clip_grad=1.0,  intermediate_save=None):
    model_state = generate_model_state(model)
    if best_loss is None:
        best_loss = eval_seq_img_vit(model=model, dataloader=dataloader_val, run=run, seq_len=seq_len_eval,
                                       query_len=eval_query_len)
        checkpoint = generate_checkpoint(model, optimizer, best_loss, val_step, train_step, lr_scheduler, loss_scaler, )
        nu.upload_model_state(model_state, run)
        nu.upload_checkpoint(checkpoint, run, intermediate_save=intermediate_save)


    for epoch in range(epochs):
        # lr_scheduler.step(val_step)
        model.train()
        for batch_id, (img, target) in enumerate(dataloader_train):
            img, target = img.to(config.DEVICE, non_blocking=True), target.to(config.DEVICE, non_blocking=True)

            seq_len = random.randint(seq_len_range[0], seq_len_range[1])
            query_len = random.randint(query_len_range[0], query_len_range[1])
            seq, pos_x, pos_y = patcher.get_seq(fov_props=model.foveation_properties, img=img, seq_len=seq_len,
                                                seq_type='rnd')

            query_seq, query_pos_x, query_pos_y = patcher.get_seq(fov_props=model.foveation_properties, img=img,
                                                                  seq_len=query_len,
                                                                  seq_type='rnd')
            recon = model.forward_seq(seq, pos_x, pos_y, query_pos_x, query_pos_y)
            loss = F.mse_loss(input=dataloader_val.dataset.transform.transforms[-1](recon), target=query_seq[0])
            optimizer.zero_grad()
            loss_scaler(loss, optimizer, clip_grad=clip_grad, parameters=model.parameters())
            run["series/train/loss"].append(loss.detach().item())

            train_step += 1
        val_loss =  eval_seq_img_vit(model=model, dataloader=dataloader_val, run=run, seq_len=seq_len_eval,
                                       query_len=eval_query_len)
        val_step += 1
        if val_loss < best_loss:
            print(f'updating model after val step {val_step}')
            model_state = generate_model_state(model)
            nu.upload_model_state(model_state, run)
            best_loss = val_loss
        checkpoint = generate_checkpoint(model, optimizer, best_loss, val_step, train_step, lr_scheduler, loss_scaler)
        nu.upload_checkpoint(checkpoint, run)
    # nu.sync(run=run)
    run.stop()

