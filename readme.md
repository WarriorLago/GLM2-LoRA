### 1. 项目背景和目标

这个项目主要是针对医疗数据处理和生成的自动化需求，使用ChatGLM2模型来处理复杂的医疗对话和数据，提升医疗数据分析的效率和准确性。项目的最终目标是开发一个能够自动提取医疗数据并生成合理对话输出的系统，帮助医疗机构更好地管理和处理患者纠纷数据。

### 2. 数据集收集与处理

项目的数据集来源于多个渠道，包括教师提供的Excel文件和通过大模型生成的模拟数据。具体步骤：

- **数据生成**：通过大模型生成模拟医疗场景的对话和登记信息，涵盖不同的病例和医疗纠纷情况。
- **数据提取和预处理**：使用自定义的脚本（如 `excel转json.py`(excel转json)）将Excel数据转换为JSON格式，提取关键信息如姓名、性别、诊断证明等。这一步确保了数据的结构化处理和模型的输入输出一致性。

### 3. 模型训练

我使用了预训练的大语言模型（如ChatGLM2）进行微调。在训练过程中，设计了专门的 `MedicalDataLoader` 加载器来处理这些数据。配置和训练参数包括：

- **模型配置**：ChatGLM2的配置，设定了层数、隐藏单元、注意力头数等参数。
- **训练参数**：如学习率、批次大小、训练轮数等，保证模型能够稳定收敛。

训练的过程通过定制脚本来实现，主要使用MindSpore框架(run_chat_cli)，并且采用了LoRA微调技术来降低训练成本。

### 4. 问题与挑战

项目中遇到了一些挑战：

- **数据不足问题**：通过大模型生成了额外的数据，增加了训练数据的多样性(大作业数据集)。
- **模型调参问题**：通过实验不断调整数据量和学习率，最终提升了模型性能。

### 5. 实践总结

项目的完成不仅提高了我在数据处理、模型训练等方面的技术能力，还提升了团队协作和解决问题的能力。







### 1. **学习率、批次大小、训练轮数等**：

这些参数控制模型的训练过程，主要用来优化模型的收敛性和训练效果：

- **学习率**：决定模型在每次迭代中参数更新的步伐大小。
- **批次大小（batch size）**：每次训练迭代中处理的样本数。
- **训练轮数（epochs）**：训练过程中完整遍历数据集的次数。

这些参数通常放在 **LoRA的配置文件（如 YAML 文件）** 中，用来定义模型训练的过程。

### 2. **层数、隐藏单元数、注意力头数等**：

这些是**模型的结构参数**，它们定义了模型的复杂性和容量：

- **层数**：模型的深度，表示网络中包含的层的数量。
- **隐藏单元数（hidden size）**：每一层中的神经元数量，决定了模型的容量。
- **注意力头数**：决定自注意力机制中并行计算注意力的头数，影响模型的表示能力。

这些参数通常也放在 **LoRA的YAML配置文件** 中，但它们定义的是模型的架构而不是训练过程。

### 3.**数据加载器（MedicalDataLoader）中的参数**：

`MedicalDataLoader` 负责加载和预处理数据，常见配置参数包括：

- **数据路径**：指定训练数据的位置。
- **shuffle**：是否在每次训练时随机打乱数据。
- **批次大小（batch size）**：用于控制每次加载数据时的数量。
- **num_workers**：定义数据加载时使用的并行线程数。









MindSpore 是一个通用的AI框架，主要用于训练和部署深度学习模型。 Langchain 是一个专注于大语言模型应用的框架，用于构建和管理基于LLM的复杂任务。 我还是不懂 这不都是框架吗

你说得对，**MindSpore** 和 **Langchain** 都是框架，但它们的用途和功能是不同的，尽管都是为AI和机器学习提供支持的工具。为了帮助你更好地理解，我将通过它们的核心功能和使用场景做详细的对比。

### 1. **MindSpore：通用AI框架**

MindSpore 是一个**通用的深度学习框架**，类似于 TensorFlow 和 PyTorch，用于搭建和训练**各种神经网络模型**。它的主要目标是提供一个完整的AI开发平台，涵盖从数据加载、模型构建到训练与部署的全流程。

- **功能**：
  - **训练模型**：你可以用它来训练各种AI模型（如图像分类、自然语言处理、语音识别等任务的模型）。
  - **优化硬件性能**：MindSpore 是针对**华为昇腾**芯片优化的，能够高效利用硬件资源进行模型训练和推理。
  - **微调与推理**：它支持微调（例如在特定任务上微调预训练模型），并且可以部署模型进行推理。
- **应用场景**：
  - 在昇腾芯片上进行大规模深度学习训练。
  - 构建AI系统，比如图像分类、机器翻译、自动驾驶等任务。

MindSpore更偏向于**构建AI模型的底层基础**，它是一个开发者用来**实现模型训练**、**优化模型**、**在硬件上部署**的工具。

### 2. **Langchain：大语言模型（LLM）应用框架**

Langchain 是为使用和管理**大语言模型（LLM）**设计的，旨在帮助开发者快速构建基于LLM的应用，特别是在复杂的自然语言处理任务上，如对话生成、信息抽取等。

- **功能**：
  - **管理LLM**：Langchain 帮助开发者管理和使用多个大语言模型，并将这些模型集成到应用中。
  - **构建复杂对话**：它专注于构建多轮对话系统或其他语言处理应用，例如问答系统、信息摘要等。
  - **连接数据源**：Langchain 可以连接多个数据源，帮助开发者在大语言模型中结合外部信息进行推理。
- **应用场景**：
  - 开发智能对话系统。
  - 构建基于LLM的自然语言处理任务，如聊天机器人、自动生成文档、知识问答等。

Langchain更适合**开发基于语言模型的应用**，它不是用来训练模型，而是用来**管理和应用预训练的大模型**（例如 GPT 系列），从而构建智能对话、信息处理系统等。

### 总结：

- **MindSpore**：适合**模型的构建、训练、优化和部署**，它是一个底层框架，主要用于通用AI任务。
- **Langchain**：适合**构建复杂的基于大语言模型的应用**，它是一个上层工具，用于处理语言模型任务，而不是直接训练模型。

**MindSpore** 是面向构建深度学习模型的平台，**Langchain** 是面向使用预训练语言模型来开发语言应用的框架。