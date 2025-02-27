# SPO 是什么

![SPO-method](https://s2.loli.net/2025/02/21/D7MpUjFhLZTxoGd.png)

SPO（Self-Supervised Prompt Optimization）是一种基于大语言模型自监督能力的提示优化框架。与传统依赖人工标注或基准答案的方法不同，SPO通过对比不同提示生成的输出质量，自主完成优化迭代。

该框架创新性地将优化过程分解为执行-评估-优化三阶段循环，利用LLM自身对任务需求的理解能力，通过成对比较输出结果获得优化信号。**自监督提示优化（SPO）**相较于传统方法，实现了优秀的性能，使得成本效率高出17.8-90.9倍。🚀

# 如何运行 SPO

## 通过镜像一键部署和运行

在这里特别感谢 `UCloud` 优云智算提供的 GPU 算力支持！让项目得到了快速的部署和调试运行。

### UCloud 介绍

![UCloud](https://s2.loli.net/2025/02/13/dDV4fosLACQgpmJ.png)

> 优云智算是 UCloud 优刻得的GPU算力租赁平台，专注于为用户提供灵活的算力资源。支持按天、按小时短期租赁及包月长期租赁，满足各类需求。
> 
> 结合丰富的公共镜像社区，优云智算提供多种预配置的容器镜像，如LLamaFactory、SD-webUI 和 LLM 等，实现一键部署，5分钟就能快速上手 AI，助力用户快速启动和扩展项目。

### 1. 使用该镜像创建实例

**镜像发布页（神秘通道）**：<https://www.compshare.cn/images-detail?ImageID=compshareImage-18u5hmtunbzm&referral_code=4sOb83sEXe4BLkKYqw9G4P&ytag=GPU_hych_Lcsdn_csdn_display>

>【算力福利速递】神秘通道秒领40枚算力金币解锁20小时顶配4090显卡试驾体验！学生党/职场人亮出大佬身份，立享永久VIP+额外金币补给包，快乐白嫖手慢无~

首先，在`镜像发布页`可以查看到我制作完成并分享到平台的实例镜像，通过右侧的`使用该镜像创建实例`可以快速创建一个实例。

![UCloud_use_mirror](https://s2.loli.net/2025/02/27/C2n4pcISifrqXsb.png)

### 2. 部署GPU实例

可按需选择配置后再`立即部署`。

![UCloud_mirror_ini](https://s2.loli.net/2025/02/27/bmWo37YZvtkgqIL.png)

### 3. 启动实例

稍等片刻后，实例就会自动创建并启动，通过查看`实例列表`可查看实例的运行状态，并支持随时关闭或启用。

![UCloud_contorl](https://s2.loli.net/2025/02/13/Jw9BvKVS5POXW2k.png)

实例同时提供了一个 `JupyterLab` 应用作为交互式开发环境，它提供了更现代化和灵活的用户界面，方便我们继续后续的步骤。

![UCloud_JupyterLab](https://s2.loli.net/2025/02/13/utpxBdQqGCMOZSA.png)

### 4. 运行 SPO WebUI 服务

启动实例后，你可以通过 `JupyterLab` 应用的终端输入以下命令来快速启动服务：
```bash
python -m streamlit run app.py
```

WebUI 服务默认通过 `8501` 端口进行访问，镜像已经配置了端口转发，你可以直接通过公网访问。

## 本地部署 — 环境准备

### 1. 拉取项目：
```bash
git clone https://github.com/Airmomo/SPO.git
```

### 2. 进入项目目录：
```bash
cd SPO/
```

### 3. 创建并激活 Python 虚拟环境：
```bash
# Windows/macOS/Linux 通用命令
python -m venv myenv

# 如果遇到 python 命令无效，尝试用具体版本号：
python3 -m venv myenv
```

### 4. 激活虚拟环境：
- Windows 系统：
```bash
# 常规命令提示符（CMD）
myenv\Scripts\activate.bat

# PowerShell
.\myenv\Scripts\Activate.ps1
```

- macOS/Linux 系统：
```bash
source myenv/bin/activate
```

### 5. 安装依赖：
```bash
pip install -e .
cd ../
```

## 本地运行 — 快速开始！

项目提供了更加友好的交互体验，可以使用 Streamlit Web 界面来配置LLM和运行优化器。

首先，安装 Streamlit：

```bash
pip install "streamlit~=1.42.0"
```

> 安装`Streamlit`后可能会提示存在依赖版本冲突，不会影响正常运行，可以忽略！

然后运行 Web 界面：

```bash
python -m streamlit run app.py
```

默认运行在8501端口，启动后会自动打开浏览器并访问`http://localhost:8501/`.

![SPO-LLM-ini](https://s2.loli.net/2025/02/21/QX5gSr9umoOwHd3.png)

## 命令行运行

### 1. 配置 API 密钥和参数

在运行 PromptOptimizer 之前，需要配置语言模型 (LLM) 的参数。这些参数可以在 `config/config2.yaml` 文件中设置，你可以参考 `examples/spo/config2.example.yaml` 文件的格式进行配置。

### 2. 定义迭代模板

创建一个迭代模板文件 `metagpt/ext/spo/settings/task_name.yaml`，模板内容如下：

```yaml
prompt: |
  Please solve the following problem.

requirements: |
  ...

count: None

qa:
  - question: |
      ...
    answer: |
      ...

  - question: |
      ...
    answer: |
      ...
```

#### 模板字段说明：
- **prompt**：迭代的初始提示。
- **requirements**：期望的效果或结果（例如，生成更多思考或使用更幽默的语言）。
- **count**：生成提示的目标字数（例如，50）。设置为 `None` 表示不限制字数。
- **qa**：用于迭代的问答对，通常包含 3 个左右的问答对。
  - **question**：数据集中用于迭代的问题。
  - **answer**：对应的答案，可以包含期望的思考模式或响应，也可以留空。

参考示例：`metagpt/ext/spo/settings/Navigate.yaml`

---

### 3. 实现 PromptOptimizer

PromptOptimizer 提供了三种运行方式，分别是 Python 脚本、命令行接口和 Streamlit Web 界面。

#### 通过 Python 脚本运行

以下是通过 Python 脚本运行 PromptOptimizer 的示例代码：

```python
from metagpt.ext.spo.components.optimizer import PromptOptimizer
from metagpt.ext.spo.utils.llm_client import SPO_LLM

if __name__ == "__main__":
    # 初始化 LLM 设置
    SPO_LLM.initialize(
        optimize_kwargs={"model": "claude-3-5-sonnet-20240620", "temperature": 0.7},
        evaluate_kwargs={"model": "gpt-4o-mini", "temperature": 0.3},
        execute_kwargs={"model": "gpt-4o-mini", "temperature": 0}
    )

    # 创建并运行优化器
    optimizer = PromptOptimizer(
        optimized_path="workspace",  # 输出目录
        initial_round=1,  # 起始轮次
        max_rounds=10,  # 最大优化轮次
        template="Poem.yaml",  # 模板文件
        name="Poem",  # 项目名称
    )

    optimizer.optimize()
```

#### 通过命令行接口运行

运行以下命令以通过命令行接口启动优化器：

```bash
python -m examples.spo.optimize
```

可用的命令行选项如下：

```
--opt-model            用于优化的模型（默认：claude-3-5-sonnet-20240620）
--opt-temp            优化的温度参数（默认：0.7）
--eval-model          用于评估的模型（默认：gpt-4o-mini）
--eval-temp          评估的温度参数（默认：0.3）
--exec-model          用于执行的模型（默认：gpt-4o-mini）
--exec-temp          执行的温度参数（默认：0）
--workspace          输出目录路径（默认：workspace）
--initial-round      初始轮次编号（默认：1）
--max-rounds        最大轮次数量（默认：10）
--template          模板文件名称（默认：Poem.yaml）
--name              项目名称（默认：Poem）
```

查看帮助信息：

```bash
python -m examples.spo.optimize --help
```

---

### 4. 查看结果

优化完成后，结果将存储在 `workspace` 目录中，结构如下：

```
workspace
  └── Project_name
      └── prompts
          ├── results.json 
          ├── round_1
          │   ├── answers.txt
          │   └── prompt.txt
          ├── round_2
          │   ├── answers.txt
          │   └── prompt.txt
          ├── round_3
          │   ├── answers.txt
          │   └── prompt.txt
          ├── ...
          └── round_n
              ├── answers.txt
              └── prompt.txt
```

文件说明：
- **results.json**：存储每轮迭代是否成功判断及其他相关信息。
- **prompt.txt**：对应轮次的优化提示。
- **answers.txt**：使用该提示生成的输出结果。

# Citation

If you use SPO in your research, please cite our paper:

```
@misc{xiang2025spo,
      title={Self-Supervised Prompt Optimization}, 
      author={Jinyu Xiang and Jiayi Zhang and Zhaoyang Yu and Fengwei Teng and Jinhao Tu and Xinbing Liang and Sirui Hong and Chenglin Wu and Yuyu Luo},
      year={2025},
      eprint={2502.06855},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2502.06855}, 
}
```