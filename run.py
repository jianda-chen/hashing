import itertools
import os, sys
from ml_toolkit.pytorch_utils.misc import autocuda, get_data_loader, hash_func_wrapper
from ml_toolkit.pytorch_utils.test_utils import run_test, load_models, save_test_results
from ml_toolkit.pytorch_utils.train_utils import save_models, save_loss_records

sys.path.append(os.path.dirname(__file__))

from train import train
from pretrain import pretrain


def run_pretrain(model_def, params):
    "train encoder and hash generator on source data"
    save_model_to = "saved_models/pretrain"

    enc = autocuda(model_def.LeNetEncoder())
    h = autocuda(model_def.LeNetCodeGen(code_len=params.hash_size))

    src_loader = get_data_loader(data_path=params.train_data_path["source"], dataset_mean=params.dataset_mean,
                                 dataset_std=params.dataset_std,
                                 batch_size=params.batch_size, shuffle_batch=True, image_scale=params.image_scale)

    train_results = pretrain(params=params, enc=enc, h=h, src_loader=src_loader)
    save_models(models=train_results["models"], save_model_to=save_model_to)
    for key, value in train_results["loss_records"].items():
        save_loss_records(loss_records=value, loss_name=key, save_to=save_model_to)


def main(model_def, params, pretrained_model_path=None):
    "training encoder, code generator using adversarial domain adaptation"
    save_model_to = "saved_models"

    # load models
    if (pretrained_model_path):
        pretrained_model = load_models(path=pretrained_model_path,model_names=["enc","h"],test_mode=False)
        enc = autocuda(pretrained_model["enc"])
        h = autocuda(pretrained_model["h"])
    else:
        enc = autocuda(model_def.LeNetEncoder())
        h = autocuda(model_def.LeNetCodeGen(code_len=params.hash_size))

    dcd = autocuda(model_def.Discriminator(input_dims=params.dcd_input_dims, hidden_dims=params.dcd_hidden_dims, output_dims=params.dcd_output_dims))

    # run training
    src_loader = get_data_loader(data_path=params.train_data_path["source"], dataset_mean=params.dataset_mean,
                    dataset_std=params.dataset_std,
                    batch_size=params.batch_size, shuffle_batch=True, image_scale=params.image_scale)
    tgt_loader = get_data_loader(data_path=params.train_data_path["target"], dataset_mean=params.dataset_mean,
                                 dataset_std=params.dataset_std,
                                 batch_size=params.batch_size, shuffle_batch=True, image_scale=params.image_scale)

    train_results = train(params=params,enc=enc,h=h,dcd=dcd,src_loader=src_loader,tgt_loader=tgt_loader)

    # save models and records
    save_models(models=train_results["models"],save_model_to=save_model_to)
    for key, value in train_results["loss_records"].items():
        save_loss_records(loss_records=value,loss_name=key,save_to=save_model_to)


def run_testing(params, model_path, model_def):
    # load data
    query_loader = get_data_loader(data_path=params.test_data_path["query"],dataset_mean=params.dataset_mean,dataset_std=params.dataset_std,
                                   batch_size=params.batch_size,shuffle_batch=False,image_scale=params.image_scale)
    db_loader = get_data_loader(data_path=params.test_data_path["db"], dataset_mean=params.dataset_mean,
                                   dataset_std=params.dataset_std, batch_size=params.batch_size, shuffle_batch=False,image_scale=params.image_scale)
    # load model
    models = load_models(path=model_path,model_names=["enc","h"])
    hash_model = model_def.get_hash_function(enc=autocuda(models["enc"]),h=autocuda(models["h"]))
    hash_model = hash_func_wrapper(hash_model)

    # run test
    test_results = run_test(query_loader=query_loader,db_loader=db_loader,query_hash_model=hash_model,db_hash_model=hash_model,radius=params.precision_radius)
    save_test_results(test_results=test_results,save_to=os.path.join(model_path,"test_results"))

########################################################################################
##############    The following are experiment suites to run  ##########################
########################################################################################

def suite_1():
    "pretraining on source"
    import model as model_def
    import params
    params.iterations = 300
    run_pretrain(model_def=model_def,params=params)

def suite_2():
    "adversarial domain adaptation training "
    import model as model_def
    import params
    params.iterations = 500
    main(model_def=model_def,params=params,pretrained_model_path="saved_models/pretrain")


def mnist_test_on_training_data():
    import model as model_def
    import params
    # params.test_data_path = {
    #     "query": params.train_data_path["target"],
    #     "db": params.train_data_path["source"]
    # }
    run_testing(model_def=model_def, params=params, model_path="saved_models/")


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = '2'
    suite_1()
    # suite_2()
    mnist_test_on_training_data()
