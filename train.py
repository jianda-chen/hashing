"""
Few-shot adversarial domain adaptation
"""
import torch
import itertools
from ml_toolkit.data_process import make_variable
from ml_toolkit.pytorch_utils.loss import get_pairwise_sim_loss

NUM_CLASSES = 10
# group labels
SAME_CLS_SRC = 0 # G1
SAME_CLS_BOTH = 1 # G2
DIFF_CLS_SRC = 2  # G3
DIFF_CLS_BOTH = 3  # G4

def compute_hash_loss(hash,labels):
    return get_pairwise_sim_loss(feats=hash,labels=labels,num_classes=NUM_CLASSES)

def make_pairs(src_feats,tgt_feats,src_labels,tgt_labels):
    pair_groups = {i:[] for i in range(4)}
    # collect pairs
    ## diff domain
    for i, src_label in enumerate(src_labels):
        for j, tgt_label in enumerate(tgt_labels):
            if (i == j):
                pair_groups[SAME_CLS_BOTH].append(torch.cat([src_feats[i],tgt_feats[j]]))
            else: # diff domain, diff class
                pair_groups[DIFF_CLS_BOTH].append(torch.cat([src_feats[i], tgt_feats[j]]))

    ## source domain only
    for i in range(len(src_labels)):
        l1 = src_labels[i]
        for j in range(i,len(src_labels)):
            l2 = src_labels[j]
            if (i!=j and l1 == l2):
                pair_groups[SAME_CLS_SRC].append(torch.cat([src_feats[i],src_feats[j]]))
            elif(i!=j and l1 != l2):
                pair_groups[DIFF_CLS_SRC].append(torch.cat([src_feats[i], src_feats[j]]))

    # the 4 groups might have different # of items, we pick the minimum size among the 4 (say M)
    # and we only keep M items for each group
    pairs = None
    pair_labels = []
    min_size = min([len(value) for value in pair_groups.values()])
    for key, value in pair_groups.items():
        value = torch.stack(value[:min_size])
        pairs = value if pairs is None else torch.cat([pairs,value])
        pair_labels += [key for _ in range(min_size)]

    return pairs, pair_labels

def compute_dcd_confusion_loss(outputs,labels):
    "this loss makes discriminator unable to tell between G1 and G2, and between G3 and G4"
    G2_outputs = [] # correspond to label 1
    G4_outputs = [] # correspond to label 3

    # collect DCD outputs of G2, G4 pairs
    for i, label in enumerate(labels):
        if (label == 1):
            G2_outputs.append(outputs[i])
        elif(label == 3):
            G4_outputs.append(outputs[i])

    G2_outputs = torch.stack(G2_outputs)
    G4_outputs = torch.stack(G4_outputs)

    # calculate cross-entropy loss
    criterion = torch.nn.CrossEntropyLoss()
    G1_labels = torch.LongTensor([0 for _ in range(len(G2_outputs))])
    G3_labels = torch.LongTensor([2 for _ in range(len(G2_outputs))])
    return 0.5 * criterion(G2_outputs,make_variable(G1_labels,requires_grad=False)) + \
            0.5 * criterion(G4_outputs,make_variable(G3_labels,requires_grad=False))

def compute_dcd_classification_loss(outputs,labels):
    criterion = torch.nn.CrossEntropyLoss()
    labels = torch.LongTensor(labels)
    return criterion(outputs,make_variable(labels,requires_grad=False))

def train(enc, dcd, h, src_loader,tgt_loader, params):
    "enc is the encoder, DCD is discriminator"
    opt_enc = torch.optim.Adam(enc.parameters(), lr=params.learning_rate)
    opt_dcd = torch.optim.Adam(dcd.parameters(), lr=params.learning_rate)
    opt_h = torch.optim.Adam(h.parameters(), lr=params.learning_rate)

    loss_records = {"dcd_confusion":[],"hash":[],"dcd_clf":[]}

    for i in range(params.iterations):

        loader = enumerate(zip(src_loader, tgt_loader))
        acc_loss = {key:0 for key in loss_records}

        for step, ((images_src, labels_src), (images_tgt, labels_tgt)) in loader:

            print("epoch {}/ batch {}".format(i,step))

            images_src = make_variable(images_src)
            images_tgt = make_variable(images_tgt)

            g_src = enc(images_src)
            g_tgt = enc(images_tgt)

            # prepare G1, G2, G3, G4 pairs, and pass to discriminator
            pairs, pair_labels = make_pairs(src_feats=g_src,tgt_feats=g_tgt,src_labels=labels_src,tgt_labels=labels_tgt)
            dcd_out = dcd(pairs)

            ######################
            # 1. update enc and h#
            ######################
            hash_src = h(g_src)
            hash_tgt = h(g_tgt)
            hash_loss = compute_hash_loss(hash=torch.cat([hash_src,hash_tgt]),labels=torch.cat([labels_src,labels_tgt]))
            dcd_confusion_loss = compute_dcd_confusion_loss(outputs=dcd_out,labels=pair_labels)
            combined_loss = params.gamma * dcd_confusion_loss + hash_loss

            combined_loss.backward(retain_variables=True)
            opt_enc.step()
            opt_h.step()
            opt_enc.zero_grad(); opt_h.zero_grad(); opt_dcd.zero_grad()

            # record loss
            acc_loss["dcd_confusion"] += dcd_confusion_loss.cpu().data.numpy()[0]
            acc_loss["hash"] += hash_loss.cpu().data.numpy()[0]

            ##################
            # 2. update DCD  #
            ##################
            dcd_clf_loss = compute_dcd_classification_loss(outputs=dcd_out,labels=pair_labels)
            dcd_clf_loss.backward()
            opt_dcd.step()
            opt_dcd.zero_grad(); opt_enc.zero_grad()

            # record loss
            acc_loss["dcd_clf"] += dcd_clf_loss.cpu().data.numpy()[0]

        # record average loss
        for key in loss_records.keys():
            loss_records[key].append(acc_loss[key] / (step + 1))

    models = {
        "enc": enc,
        "h": h,
        "dcd": dcd
    }
    return {
        "models": models,
        "loss_records": loss_records
    }