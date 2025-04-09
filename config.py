from pathlib import Path

def get_config():
    return{
        "batch_size":2,
        "num_epochs" : 2,
        "lr":10**-4,
        "seq_len" : 350,
        "d_model":64,
        "lang_src":"en",
        "lang_tgt" :"it",
        "model_folder":"weights",
        "model_basename":"tmodel_",
        "preload":None,
        "tokenizer_file":"tokenizer_{0}.json",
        "experiment_name":"rune/t_model"
        
    }

def get_weights_file_path(config, epoch:str):
    model_folder = config["model_folder"]
    model_basename = config["model_basename"]
    model_filename = f"{model_basename}{epoch}.pt"
    return str(Path(".")/ model_folder/model_filename)