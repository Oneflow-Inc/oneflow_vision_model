#import torch
#import torch.nn as nn
import oneflow as flow
import oneflow.typing as tp

#def get_state_dict_on_cpu(obj):
    #cpu_device = torch.device('cpu')
    #state_dict = obj.state_dict()
    #for key in state_dict.keys():
        #state_dict[key] = state_dict[key].to(cpu_device)

    #return state_dict

@flow.global_function(type="predict")
def job() -> tp.Numpy:
    n_iter = flow.get_variable("n_iter",
        shape=(1,),
        initializer=flow.random_normal_initializer()
        )
    return n_iter

def save_ckpt(ckpt_name):
    # ckpt_dict = {'n_iter': [n_iter]}  #可能存在问题
    # flow.load_variables(ckpt_dict)
    #for prefix, model in models:
        #ckpt_dict[prefix] = get_state_dict_on_cpu(model)

    #for prefix, optimizer in optimizers:
        #ckpt_dict[prefix] = optimizer.state_dict()
    #torch.save(ckpt_dict, ckpt_name)

    flow.checkpoint.save(ckpt_name)



def load_ckpt(ckpt_name):
    #ckpt_dict = torch.load(ckpt_name)
    flow.load_variables(flow.checkpoint.get(ckpt_name))

    #for prefix, model in models:
        #assert isinstance(model, nn.Module)
        #model.load_state_dict(ckpt_dict[prefix], strict=False)
    #if optimizers is not None:
        #for prefix, optimizer in optimizers:
            #optimizer.load_state_dict(ckpt_dict[prefix])

