"""
Pre-train encoder and code generator, using source data
"""

import torch
from ml_toolkit.data_process import make_variable
from ml_toolkit.pytorch_utils.loss import get_pairwise_sim_loss

def pretrain(enc, h, src_loader,params):
    "enc is the encoder, h is code generator"
    opt_enc = torch.optim.Adam(enc.parameters(), lr=params.learning_rate)
    opt_h = torch.optim.Adam(h.parameters(), lr=params.learning_rate)

    loss_records = {"hash":[]}

    for i in range(params.iterations):

        loader = enumerate(src_loader)
        acc_loss = {key:0 for key in loss_records}

        for step, (images_src, labels_src) in loader:

            print("epoch {}/ batch {}".format(i,step))

            images_src = make_variable(images_src)
            g_src = enc(images_src)
            hash_src = h(g_src)
            hash_loss = get_pairwise_sim_loss(feats=hash_src,labels=labels_src,num_classes=params.num_classes)

            acc_loss["hash"] += hash_loss.cpu().data.numpy()[0]

            hash_loss.backward()
            opt_enc.step(); opt_h.step()
            opt_enc.zero_grad(); opt_h.zero_grad()

        # record average loss
        for key in loss_records.keys():
            loss_records[key].append(acc_loss[key] / (step + 1))

    models = {
        "enc": enc,
        "h": h
    }
    return {
        "models": models,
        "loss_records": loss_records
    }
