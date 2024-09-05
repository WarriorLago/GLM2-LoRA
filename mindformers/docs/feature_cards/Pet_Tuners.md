# 低参微调

MindPet（Pet：Parameter-Efficient Tuning）是属于Mindspore领域的微调算法套件。随着计算算力不断增加，大模型无限的潜力也被挖掘出来。但随之在应用和训练上带来了巨大的花销，导致商业落地困难。因此，出现一种新的参数高效（parameter-efficient）算法，与标准的全参数微调相比，这些算法仅需要微调小部分参数，可以大大降低计算和存储成本，同时可媲美全参微调的性能。

目前低参微调针对MindFormers仓库已有的大模型进行统一架构设计，对于LLM类语言模型，我们可以统一调度修改，做到只需要调用接口或者是自定义相关配置文件，即可完成对LLM类模型的低参微调算法的适配。

## [微调支持列表](../model_support_list.md#微调支持列表)

## Lora使用示例

1. 确定需要替换的模块，lora模块一般替换transformers模块的query，key，value等线性层，替换时需要找到（query, key, value）等模块的变量量，在统一框架中采用的是正则匹配规则对需要替换的模块进行lora微调算法的替换。

```python
# 以GPT为例，在GPT的attention定义中，我们可以查看到qkv的定义如下：
class MultiHeadAttention(Cell):
    ...
    # Query
    self.dense1 = Linear(hidden_size,
                          hidden_size,
                          compute_dtype=compute_dtype,
                          param_init_type=param_init_type)
    # Key
    self.dense2 = Linear(hidden_size,
                          hidden_size,
                          compute_dtype=compute_dtype,
                          param_init_type=param_init_type)
    # Value
    self.dense3 = Linear(hidden_size,
                          hidden_size,
                          compute_dtype=compute_dtype,
                          param_init_type=param_init_type)
    ...
```

找到如上定义后，在步骤2中则可以定义lora的正则匹配规则为：`r'.*dense1|.*dense2|.*dense3'`

2. 定义lora的配置参数修改已有的配置文件，如根据`configs/gpt2/run_gpt2.yaml`，在`model_config`中增加lora相关的配置，如下所示：

```yaml
model:
  model_config:
    type: GPT2Config
    ...
    pet_config: # configurition of lora
      pet_type: lora
      lora_rank: 8
      lora_alpha: 16
      lora_dropout: 0.1
      target_modules: ".*dense1|.*dense2|.*dense3"
  arch:
    type: GPT2LMHeadModel
```

修改完毕后，可以参考训练流程使用该配置文件进行模型训练。

3. 使用MindFormer的Trainer进行模型训练：

```python
import mindspore as ms
from mindformers.trainer.trainer import Trainer

ms.set_context(mode=0) # 设定为图模式加速

gpt2_trainer = Trainer(
    task='text_generation',
    model='gpt2',
    pet_method='lora',
    train_dataset="/data/wikitext-2/train",
)

gpt2_trainer.finetune()
```

至此，完成了一个微调算法适配过程，最后执行上述步骤3中的代码即可拉起微调算法的训练流程。