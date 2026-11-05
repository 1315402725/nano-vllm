import os
from nanovllm import LLM, SamplingParams
from transformers import AutoTokenizer


def main():
    path = os.path.expanduser("~/huggingface/Qwen3-0.6B/")
    tokenizer = AutoTokenizer.from_pretrained(path)
    llm = LLM(path, enforce_eager=True, tensor_parallel_size=1)

    sampling_params = SamplingParams(temperature=0.6, max_tokens=256)
    prompts = [
        "introduce yourself",
        "list all prime numbers within 100",
    ]
    # 将每条普通提示词分别包装成模型要求的聊天文本；处理后仍是两个独立请求，不是同一段多轮对话。
    prompts = [
        # 使用模型分词器自带的聊天模板，添加角色、消息边界等标记；具体格式由模型的模板决定。
        tokenizer.apply_chat_template(
            # 一次调用接收一个消息列表；这里仅包含一条用户消息，role 表示说话者，content 是正文。
            [{"role": "user", "content": prompt}],
            # 返回格式化后的字符串，不在这里转成 token ID；后续 llm.generate() 内部会完成编码。
            tokenize=False,
            # 在末尾添加模板定义的助手回复开头，提示模型接下来继续生成助手的回答。
            add_generation_prompt=True,
        )
        # 列表推导式：遍历原 prompts，分别转换每条提示词，再将结果列表赋回 prompts。
        for prompt in prompts
    ]
    outputs = llm.generate(prompts, sampling_params)

    for prompt, output in zip(prompts, outputs):
        print("\n")
        print(f"Prompt: {prompt!r}")
        print(f"Completion: {output['text']!r}")


if __name__ == "__main__":
    main()
