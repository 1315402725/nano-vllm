import os
from glob import glob
import torch
from torch import nn
from safetensors import safe_open


def default_weight_loader(param: nn.Parameter, loaded_weight: torch.Tensor):
    param.data.copy_(loaded_weight)


# 将模型目录中的训练权重加载到已创建的模型参数中；model 是模型对象，path 是本地模型目录。
def load_model(model: nn.Module, path: str):
    # 获取模型定义的合并映射，例如 q_proj → (qkv_proj, "q")；没有该属性时使用空字典。
    packed_modules_mapping = getattr(model, "packed_modules_mapping", {})
    # 找出目录中的所有 .safetensors 权重文件并逐个处理；一个模型的权重可能分散在多个文件中。
    for file in glob(os.path.join(path, "*.safetensors")):
        # 按 PyTorch 格式打开文件，读取出的张量放在 CPU；with 会在退出代码块时关闭文件上下文。
        with safe_open(file, "pt", "cpu") as f:
            # 遍历当前文件保存的参数名称，例如 model.layers.0.self_attn.q_proj.weight。
            for weight_name in f.keys():
                # 逐个检查需要合并的原始模块名，例如 q_proj、k_proj、v_proj、gate_proj、up_proj。
                for k in packed_modules_mapping:
                    # 判断当前权重名称是否包含这个模块名；这里使用字符串子串匹配。
                    if k in weight_name:
                        # v 是模型中合并后的模块名，shard_id 标记写入哪一段，例如 (qkv_proj, "q")。
                        # shard_id 不是 GPU 编号；GPU 分片由具体加载器根据当前 rank 处理。
                        v, shard_id = packed_modules_mapping[k]
                        # 把文件中的模块名替换为模型中的模块名，例如 self_attn.q_proj.weight → self_attn.qkv_proj.weight。
                        param_name = weight_name.replace(k, v)
                        # 按完整的参数路径找到模型里已经分配好存储的 nn.Parameter 对象。
                        param = model.get_parameter(param_name)
                        # 取出绑定在参数上的专用加载方法；该分支没有默认值，要求参数提供 weight_loader。
                        weight_loader = getattr(param, "weight_loader")
                        # 从文件读取 CPU 权重，交给专用加载器；加载器选出本卡分片并复制到合并参数的指定区域。
                        weight_loader(param, f.get_tensor(weight_name), shard_id)
                        # 当前权重已成功匹配并加载，退出内层映射循环，避免同一权重再次加载。
                        break
                # 这是 for...else：只有内层循环没有执行 break 时才进入，表示没有命中任何合并映射。
                else:
                    # 不需要改名或合并时，直接用文件中的参数名称找到模型参数。
                    param = model.get_parameter(weight_name)
                    # 优先使用参数自带的加载器（例如词表分片加载器）；没有则使用默认的整体 copy_ 加载方法。
                    weight_loader = getattr(param, "weight_loader", default_weight_loader)
                    # 读取 CPU 权重并复制到目标参数；若目标参数在 GPU 上，加载器中的 copy_ 会完成设备间复制。
                    weight_loader(param, f.get_tensor(weight_name))
