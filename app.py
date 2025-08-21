import asyncio
from pathlib import Path
from typing import Dict, List
import streamlit as st
import yaml
from loguru import logger as _logger
import shutil
import uuid
import os
import time

from metagpt.const import METAGPT_ROOT
from metagpt.ext.spo.components.optimizer import PromptOptimizer
from metagpt.ext.spo.utils.llm_client import SPO_LLM, RequestType

from dotenv import load_dotenv
load_dotenv()

# 读取环境变量


API_KEY_ENV = os.getenv("API_KEY", "")
BASE_URL_ENV = os.getenv("BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
MODEL_ENV = os.getenv("MODEL", "qwen-max-latest")

def get_user_workspace():
    if "user_id" not in st.session_state:
        st.session_state.user_id = str(uuid.uuid4())

    workspace_dir = Path("workspace") / st.session_state.user_id
    workspace_dir.mkdir(parents=True, exist_ok=True)
    return workspace_dir


def cleanup_workspace(workspace_dir: Path) -> None:
    try:
        if workspace_dir.exists():
            shutil.rmtree(workspace_dir)
            _logger.info(f"Cleaned up workspace directory: {workspace_dir}")
    except Exception as e:
        _logger.error(f"Error cleaning up workspace: {e}")


def get_all_templates() -> List[str]:
    """
    Get list of all available templates (both default and user-specific)
    :return: List of template names
    """
    settings_path = Path("metagpt/ext/spo/settings")

    # Get default templates
    templates = [f.stem for f in settings_path.glob("*.yaml")]

    # Get user-specific templates if user_id exists
    if "user_id" in st.session_state:
        user_path = settings_path / st.session_state.user_id
        if user_path.exists():
            user_templates = [
                f"{st.session_state.user_id}/{f.stem}" for f in user_path.glob("*.yaml")]
            templates.extend(user_templates)

    return sorted(list(set(templates)))


def load_yaml_template(template_path: Path) -> Dict:
    if template_path.exists():
        with open(template_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    return {"prompt": "", "requirements": "", "count": None, "qa": [{"question": "", "answer": ""}]}


def save_yaml_template(template_path: Path, data: Dict) -> None:
    template_format = {
        "prompt": str(data.get("prompt", "")),
        "requirements": str(data.get("requirements", "")),
        "count": data.get("count"),
        "qa": [
            {"question": str(qa.get("question", "")).strip(
            ), "answer": str(qa.get("answer", "")).strip()}
            for qa in data.get("qa", [])
        ],
    }

    template_path.parent.mkdir(parents=True, exist_ok=True)

    with open(template_path, "w", encoding="utf-8") as f:
        yaml.dump(template_format, f, allow_unicode=True,
                  sort_keys=False, default_flow_style=False, indent=2)


def display_optimization_results(result_data):
    for result in result_data:
        round_num = result["round"]
        success = result["succeed"]
        prompt = result["prompt"]

        with st.expander(f"轮次 {round_num} {':white_check_mark:' if success else ':x:'}"):
            st.markdown("**提示词：**")
            st.code(prompt, language="text")
            st.markdown("<br>", unsafe_allow_html=True)

            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"**状态：** {'成功 ✅ ' if success else '失败 ❌ '}")
            with col2:
                st.markdown(f"**令牌数：** {result['tokens']}")

            st.markdown("**回答：**")
            for idx, answer in enumerate(result["answers"]):
                st.markdown(f"**问题 {idx + 1}：**")
                st.text(answer["question"])
                st.markdown("**答案：**")
                st.text(answer["answer"])
                st.markdown("---")

    # 总结
    success_count = sum(1 for r in result_data if r["succeed"])
    total_rounds = len(result_data)

    st.markdown("### 总结")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("总轮次", total_rounds)
    with col2:
        st.metric("成功轮次", success_count)


def main():
    if "optimization_results" not in st.session_state:
        st.session_state.optimization_results = []

    if "available_models" not in st.session_state:
        st.session_state.available_models = []
    if "base_url" not in st.session_state:
        st.session_state.base_url = "https://api.example.com"
    if "api_key" not in st.session_state:
        st.session_state.api_key = ""
    if "is_optimizing" not in st.session_state:
        st.session_state.is_optimizing = False

    workspace_dir = get_user_workspace()

    st.markdown(
        """
    <div style="background-color: #f0f2f6; padding: 20px; border-radius: 10px; margin-bottom: 25px">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px">
            <h1 style="margin: 0;">SPO | 自监督提示词优化 🤖</h1>
        </div>
        <div style="display: flex; gap: 20px; align-items: center">
            <a href="https://arxiv.org/pdf/2502.06855" target="_blank" style="text-decoration: none;">
                <img src="https://img.shields.io/badge/论文-PDF-red.svg" alt="论文">
            </a>
            <a href="https://github.com/Airmomo/SPO" target="_blank" style="text-decoration: none;">
                <img src="https://img.shields.io/badge/GitHub-仓库-blue.svg" alt="GitHub">
            </a>
            <span style="color: #666;">一个自监督提示词优化框架</span>
        </div>
    </div>
    """,
        unsafe_allow_html=True
    )

    # 创建导航栏
    tab_config, tab_template, tab_preview, tab_logs, tab_results, tab_test = st.tabs(
        ["LLM 配置", "模板配置", "当前模板预览", "优化日志", "优化结果", "测试优化后提示词"])

    # 配置选项卡
    with tab_config:
        st.header("LLM 配置")

        # LLM 设置
        st.subheader("添加 LLM 模型")

        base_url = st.text_input("BASE URL", value=BASE_URL_ENV)
        # 设置默认隐藏
        api_key = API_KEY_ENV 
        #st.text_input(
         #   "API KEY", type="password", value=API_KEY_ENV)
        model_name = st.text_input("模型名称", value=MODEL_ENV)

        col1, col2 = st.columns(2)
        with col1:
            if st.button("连通性测试并添加模型"):
                try:
                    if not model_name:
                        st.error("请输入模型名称")
                        return

                    # 进行LLM连通性测试
                    try:
                        from openai import OpenAI

                        # 初始化OpenAI客户端
                        client = OpenAI(
                            api_key=api_key,
                            base_url=base_url
                        )

                        # 测试连通性
                        response = client.chat.completions.create(
                            model=model_name,
                            messages=[{"role": "user", "content": "Hello"}],
                            temperature=0
                        )

                        if model_name not in st.session_state.available_models:
                            st.session_state.available_models.append(
                                model_name)
                        st.session_state.base_url = base_url
                        st.session_state.api_key = api_key
                        st.success("连通性测试成功，模型已添加！")
                    except Exception as e:
                        st.error(f"LLM连通性测试失败：{str(e)}")
                        return
                    finally:
                        if 'loop' in locals():
                            loop.close()

                except Exception as e:
                    st.error(f"添加模型时出错：{str(e)}")

        with col2:
            pass

        # 优化模型和优化器设置
        st.subheader("模型配置")

        # 配置保存和加载按钮
        col1, col2 = st.columns(2)
        with col1:
            # 创建配置数据
            def get_config_data():
                config_data = {}
                config_data["llm"] = {
                    "api_type": "openai",
                    "base_url": base_url,
                    "api_key": api_key
                }

                if "models" not in config_data:
                    config_data["models"] = {}

                for model in st.session_state.available_models:
                    config_data["models"][model] = {
                        "api_type": "openai",
                        "base_url": base_url,
                        "api_key": api_key,
                        "temperature": 0
                    }
                return config_data

            # 将配置转换为YAML字符串
            def get_yaml_string():
                try:
                    config_data = get_config_data()
                    return yaml.dump(config_data, allow_unicode=True, sort_keys=False, default_flow_style=False, indent=2)
                except Exception as e:
                    st.error(f"生成配置时出错：{str(e)}")
                    return ""

            # 使用download_button组件提供下载功能
            if not st.session_state.is_optimizing:
                yaml_str = get_yaml_string()
                st.download_button(
                    label="保存当前模型列表",
                    data=yaml_str,
                    file_name="config_models.yaml",
                    mime="text/yaml",
                    help="将当前模型列表下载到本地"
                )
            else:
                st.button("保存当前模型列表", disabled=True)

        with col2:
            # 每次上传新文件时生成一个新的key，防止显示之前上传的文件
            if "uploader_key" not in st.session_state:
                st.session_state.uploader_key = f"config_uploader_{int(time.time())}"
            uploaded_file = st.file_uploader(
                "上传模型列表文件", type=["yaml"], key=st.session_state.uploader_key)
            if uploaded_file is not None:  # 如果用户上传了文件
                try:
                    # 直接从上传的文件对象中读取内容
                    config_data = yaml.safe_load(uploaded_file)
                    if "llm" in config_data:
                        llm_config = config_data["llm"]
                        st.session_state.base_url = llm_config.get(
                            "base_url", "")
                        st.session_state.api_key = llm_config.get(
                            "api_key", "")
                        if "models" in config_data:
                            st.session_state.available_models = list(
                                config_data["models"].keys())
                            # 更新界面上的配置内容
                            st.success("配置已成功加载")
                except Exception as e:
                    st.error(f"加载配置时出错：{str(e)}")

        # 模型设置详细
        opt_model = st.selectbox(
            "优化模型", st.session_state.get("available_models", ["Null"]), index=0,
            disabled=st.session_state.is_optimizing,
            key=f"opt_model_{len(st.session_state.get('available_models', []))}"
        )
        opt_temp = st.slider("优化温度", 0.0, 1.0, value=0.7, step=0.1,
                             disabled=st.session_state.is_optimizing)

        eval_model = st.selectbox(
            "评估模型", st.session_state.get("available_models", ["Null"]), index=0,
            disabled=st.session_state.is_optimizing,
            key=f"eval_model_{len(st.session_state.get('available_models', []))}"
        )
        eval_temp = st.slider("评估温度", 0.0, 1.0, value=0.3, step=0.1,
                              disabled=st.session_state.is_optimizing)

        exec_model = st.selectbox(
            "执行模型", st.session_state.get("available_models", ["Null"]), index=0,
            disabled=st.session_state.is_optimizing,
            key=f"exec_model_{len(st.session_state.get('available_models', []))}"
        )
        exec_temp = st.slider("执行温度", 0.0, 1.0, value=0.0, step=0.1,
                              disabled=st.session_state.is_optimizing)

        # 优化器设置
        st.subheader("优化器设置")
        initial_round = st.number_input("初始轮次：决定了从第几轮优化结果继续优化", 1, 100, 1)
        max_rounds = st.number_input(
            "最大轮次", 1, 100, 10, disabled=st.session_state.is_optimizing)

    # 模板配置选项卡
    with tab_template:
        st.header("模板配置")

        # 模板选择/创建
        settings_path = Path("metagpt/ext/spo/settings")
        existing_templates = get_all_templates()
        template_options = existing_templates + ["创建新模板"]

        template_selection = st.selectbox(
            "选择模板", template_options, disabled=st.session_state.is_optimizing)
        is_new_template = template_selection == "创建新模板"

        if is_new_template:
            template_name = st.text_input(
                "新模板名称", disabled=st.session_state.is_optimizing)
        else:
            template_name = template_selection

        # 初始化template_path和加载模板数据
        template_path = settings_path / \
            f"{template_name}.yaml" if template_name else None
        template_data = load_yaml_template(template_path) if template_path and template_path.exists() else {
            "prompt": "", "requirements": "", "qa": []}

        if "current_template" not in st.session_state or st.session_state.current_template != template_name:
            st.session_state.current_template = template_name
            st.session_state.qas = template_data.get("qa", [])
            st.session_state.prompt = template_data.get("prompt", "")
            st.session_state.requirements = template_data.get(
                "requirements", "")
        elif is_new_template and not template_name:
            # 清空所有内容
            st.session_state.current_template = template_name
            st.session_state.qas = []
            st.session_state.prompt = ""
            st.session_state.requirements = ""

        # 使用session_state中的值填充输入框
        prompt = st.text_area(
            "提示词", value=st.session_state.get("prompt", ""), height=100, disabled=st.session_state.is_optimizing)
        requirements = st.text_area(
            "要求", value=st.session_state.get("requirements", ""), height=100, disabled=st.session_state.is_optimizing)

        # 问答部分
        st.subheader("问答示例")
        st.markdown(''':red[增加新的示例后需要 **保存模板** 才能应用！]''')

        if "qas" not in st.session_state:
            st.session_state.qas = []

        # 添加新问答按钮
        if st.button("添加新问答", disabled=st.session_state.is_optimizing):
            st.session_state.qas.append({"question": "", "answer": ""})

        # 编辑问答
        new_qas = []
        for i in range(len(st.session_state.qas)):
            st.markdown(f"**问答 #{i + 1}**")
            col1, col2, col3 = st.columns([45, 45, 10])

            with col1:
                question = st.text_area(
                    f"问题 {i + 1}", st.session_state.qas[i].get("question", ""), key=f"q_{i}", height=100
                )
            with col2:
                answer = st.text_area(
                    f"答案 {i + 1}", st.session_state.qas[i].get("answer", ""), key=f"a_{i}", height=100
                )
            with col3:
                if st.button("🗑️", key=f"delete_{i}", disabled=st.session_state.is_optimizing):
                    st.session_state.qas.pop(i)
                    st.rerun()

            new_qas.append({"question": question, "answer": answer})

        if template_name:
            template_path = settings_path / f"{template_name}.yaml"
            template_data = load_yaml_template(template_path)

            if not is_new_template:
                template_data = load_yaml_template(template_path)
                if "current_template" not in st.session_state or st.session_state.current_template != template_name:
                    st.session_state.current_template = template_name
                    st.session_state.qas = template_data.get("qa", [])
                    st.session_state.prompt = template_data.get("prompt", "")
                    st.session_state.requirements = template_data.get(
                        "requirements", "")

            if st.button("保存模板", disabled=st.session_state.is_optimizing):
                if not template_name:
                    st.error("必须填写模板名称！")
                else:
                    new_template_data = {
                        "prompt": prompt, "requirements": requirements, "count": None, "qa": new_qas}

                    save_yaml_template(template_path, new_template_data)

                    st.session_state.qas = new_qas
                    st.success(f"模板已保存到 {template_path}")

    # 当前模板预览选项卡
    with tab_preview:
        if "current_template" in st.session_state:
            st.header("当前模板预览")
            preview_data = {"qa": st.session_state.get("qas", []),
                            "requirements": st.session_state.get("requirements", ""),
                            "prompt": st.session_state.get("prompt", "")}
            st.code(yaml.dump(preview_data, allow_unicode=True), language="yaml")

    # 优化日志选项卡
    with tab_logs:
        st.header("优化日志")
        log_container = st.empty()

        class StreamlitSink:
            def write(self, message):
                current_logs = st.session_state.get("logs", [])
                current_logs.append(message.strip())
                st.session_state.logs = current_logs
                log_container.code(
                    "\n".join(current_logs), language="plaintext")

        streamlit_sink = StreamlitSink()
        _logger.remove()

        def prompt_optimizer_filter(record):
            return "optimizer" in record["name"].lower()

        _logger.add(
            streamlit_sink.write,
            format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} - {message}",
            filter=prompt_optimizer_filter,
        )
        _logger.add(METAGPT_ROOT /
                    "logs/{time:YYYYMMDD}.txt", level="DEBUG")

        # 开始优化按钮
        if st.button("开始优化"):
            try:
                st.session_state.is_optimizing = True
                # Initialize LLM
                SPO_LLM.initialize(
                    optimize_kwargs={"model": opt_model, "temperature": opt_temp, "base_url": base_url,
                                     "api_key": api_key},
                    evaluate_kwargs={"model": eval_model, "temperature": eval_temp, "base_url": base_url,
                                     "api_key": api_key},
                    execute_kwargs={"model": exec_model, "temperature": exec_temp, "base_url": base_url,
                                    "api_key": api_key},
                )

                # Create optimizer instance
                optimizer = PromptOptimizer(
                    optimized_path=str(workspace_dir),
                    initial_round=initial_round,
                    max_rounds=max_rounds,
                    template=f"{template_name}.yaml",
                    name=template_name,
                )

                # Run optimization with progress bar
                with st.spinner("正在优化提示词..."):
                    optimizer.optimize()

                st.success("优化完成！")
                prompt_path = optimizer.root_path / "prompts"
                result_data = optimizer.data_utils.load_results(
                    prompt_path)
                st.session_state.optimization_results = result_data
                st.session_state.is_optimizing = False

            except Exception as e:
                st.error(f"发生错误：{str(e)}")
                _logger.error(f"优化过程中出错：{str(e)}")
                st.session_state.is_optimizing = False

    # 优化结果选项卡
    with tab_results:
        st.header("优化结果")
        if st.session_state.optimization_results:
            display_optimization_results(
                st.session_state.optimization_results)

    # 测试优化后提示词选项卡
    with tab_test:
        st.header("测试优化后的提示词")
        col1, col2 = st.columns(2)

        with col1:
            test_prompt = st.text_area(
                "优化后的提示词", value="", height=200, key="test_prompt")

        with col2:
            test_question = st.text_area(
                "你的问题", value="", height=200, key="test_question")

        if st.button("测试提示词"):
            if test_prompt and test_question:
                try:
                    with st.spinner("正在生成回答..."):
                        SPO_LLM.initialize(
                            optimize_kwargs={"model": opt_model, "temperature": opt_temp, "base_url": base_url,
                                             "api_key": api_key},
                            evaluate_kwargs={"model": eval_model, "temperature": eval_temp, "base_url": base_url,
                                             "api_key": api_key},
                            execute_kwargs={"model": exec_model, "temperature": exec_temp, "base_url": base_url,
                                            "api_key": api_key},
                        )

                        llm = SPO_LLM.get_instance()
                        messages = [
                            {"role": "user", "content": f"{test_prompt}\n\n{test_question}"}]

                        async def get_response():
                            return await llm.responser(request_type=RequestType.EXECUTE, messages=messages)

                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)
                        try:
                            response = loop.run_until_complete(
                                get_response())
                        finally:
                            loop.close()

                        st.subheader("回答：")
                        st.markdown(response)

                except Exception as e:
                    st.error(f"生成回答时出错：{str(e)}")
                else:
                    st.warning("请输入提示词和问题。")


if __name__ == "__main__":
    main()
