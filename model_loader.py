import torch
import os
import copy


def load_model(E_Q, E_K, FC_Q, FC_K, memory, queue, dir, pre_dir, optimizer=None, D=None):
    # 加载上次训练的数据
    if os.path.exists(dir):
        checkpoint = torch.load(dir)
        E_Q.load_state_dict(checkpoint['model_state_dict_E'])
        E_K.load_state_dict(checkpoint['model_state_dict_E_K'])
        FC_Q.load_state_dict(checkpoint['model_state_dict_FC_Q'])
        FC_K.load_state_dict(checkpoint['model_state_dict_FC_K'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        # D.load_state_dict(checkpoint['model_state_dict_D'])
        memory.global_memory_item = checkpoint['memory']
        memory.memory_item = checkpoint['memory2']
        queue.queue = checkpoint['queue']
        print("Successful loading model!")
    # 加载预训练数据
    elif os.path.exists(pre_dir):
        checkpoint = torch.load(pre_dir)
        E_Q.load_state_dict(checkpoint['model_state_dict_E'])
        E_K.load_state_dict(checkpoint['model_state_dict_E_K'])
        FC_Q.load_state_dict(checkpoint['model_state_dict_FC_Q'])
        FC_K.load_state_dict(checkpoint['model_state_dict_FC_K'])
        # D.load_state_dict(checkpoint['model_state_dict_D'])
        memory.global_memory_item = checkpoint['memory']
        memory.memory_item = checkpoint['memory2']
        queue.queue = checkpoint['queue']
        print("Successful loading pretrain model!")
    # 没有可加载的数据，从头开始训练
    else:
        print("no pretrain model exist! Start training directly ")

def save_model(E_Q, E_K, FC_Q, FC_K, memory, queue, save_dir, optimizer=None, D=None):
    checkpoint = {
        'model_state_dict_E': E_Q.state_dict(),
        'model_state_dict_E_K': E_K.state_dict(),
        'model_state_dict_FC_Q': FC_Q.state_dict(),
        'model_state_dict_FC_K': FC_K.state_dict(),
        'optimizer': optimizer.state_dict(),
        # 'model_state_dict_D': D.state_dict(),
        'memory': memory.global_memory_item,
        'memory2': memory.memory_item,
        'queue': queue.queue,
    }
    torch.save(checkpoint, save_dir)
