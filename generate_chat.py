import os
import sys

import fire
import gradio as gr
import torch
import transformers
from peft import PeftModel
from transformers import GenerationConfig, LlamaForCausalLM, LlamaTokenizer

from utils.callbacks import Iteratorize, Stream
from utils.prompter import Prompter
import random
import time


if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

try:
    if torch.backends.mps.is_available():
        device = "mps"
except:  # noqa: E722
    pass


def get_model(device: str = '', load_8bit: bool = False, base_model: str = '', lora_weights: str = '', tokenizer=None):
    if device == "cuda":
        model = LlamaForCausalLM.from_pretrained(
            base_model,
            load_in_8bit=load_8bit,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        model = PeftModel.from_pretrained(
            model,
            lora_weights,
            torch_dtype=torch.float16,
        )
    elif device == "mps":
        model = LlamaForCausalLM.from_pretrained(
            base_model,
            device_map={"": device},
            torch_dtype=torch.float16,
        )
        model = PeftModel.from_pretrained(
            model,
            lora_weights,
            device_map={"": device},
            torch_dtype=torch.float16,
        )
    else:
        model = LlamaForCausalLM.from_pretrained(
            base_model, device_map={"": device}, low_cpu_mem_usage=True
        )
        model = PeftModel.from_pretrained(
            model,
            lora_weights,
            device_map={"": device},
        )

    # unwind broken decapoda-research config
    model.config.pad_token_id = tokenizer.pad_token_id = 0  # unk
    model.config.bos_token_id = 1
    model.config.eos_token_id = 2

    if not load_8bit:
        model.half()  # seems to fix bugs for some users.

    model.eval()
    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)

    return model


def main(
    load_8bit: bool = False,
    base_model: str = "decapoda-research/llama-7b-hf",
    lora_weights: str = "training_results/therapy_1",
    # The prompt template to use, will default to alpaca.
    prompt_template: str = "",
    # Allows to listen on all interfaces by providing '0.
    server_name: str = "0.0.0.0",
    share_gradio: bool = True,
    verbose: bool = True,
):
    base_model = base_model or os.environ.get("BASE_MODEL", "")
    assert (
        base_model
    ), "Please specify a --base_model, e.g. --base_model='huggyllama/llama-7b'"

    prompter = Prompter(prompt_template)
    tokenizer = LlamaTokenizer.from_pretrained(base_model)
    model = get_model(device, load_8bit, base_model, lora_weights, tokenizer)

    def evaluate(
        instruction,
        temperature=0.1,
        top_p=0.75,
        top_k=40,
        num_beams=4,
        max_new_tokens=128,
        message='',
        chat_history=[],
        **kwargs,
    ):
        my_input = ''
        for item in chat_history:
            my_input = my_input + item[0] + '\n' + item[1] + '\n'
        my_input = my_input + message + '\n'
        print("respond... ", message, chat_history, my_input)
        prompt = prompter.generate_prompt(instruction, my_input)
        if verbose:
            print("prompt = ", prompt, "\nEND_prompt\n")
        inputs = tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(device)
        generation_config = GenerationConfig(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            num_beams=num_beams,
            **kwargs,
        )

        generate_params = {
            "input_ids": input_ids,
            "generation_config": generation_config,
            "return_dict_in_generate": True,
            "output_scores": True,
            "max_new_tokens": max_new_tokens,
        }

        with torch.no_grad():
            generation_output = model.generate(
                input_ids=input_ids,
                generation_config=generation_config,
                return_dict_in_generate=True,
                output_scores=True,
                max_new_tokens=max_new_tokens,
            )
        s = generation_output.sequences[0]
        output = tokenizer.decode(s)
        if verbose:
            print("s = ", s, "\nEND_s")
            print("output = ", output, "\nEND_output\n")
        # yield prompter.get_response(output), output, prompt
        bot_message = prompter.get_response(output)
        chat_history.append((message, bot_message))
        time.sleep(1)
        return "Patient: ", chat_history

    with gr.Blocks() as demo:
        temperature = gr.components.Slider(
            minimum=0, maximum=1, value=0.1, label="Temperature"
        )
        top_p = gr.components.Slider(
            minimum=0, maximum=1, value=0.75, label="Top ppp"
        )
        top_k = gr.components.Slider(
            minimum=0, maximum=100, step=1, value=40, label="Top kkk"
        )
        beams = gr.components.Slider(
            minimum=1, maximum=4, step=1, value=4, label="Beams"
        )
        max_token = gr.components.Slider(
            minimum=1, maximum=2000, step=1, value=128, label="Max tokens"
        )
        instruction = gr.components.Textbox(
            lines=2,
            label="Instruction",
            value="Below is a dialogue between a patient and a therapist. Write one reply as if you were the therapist.",
        )
        chatbot = gr.Chatbot()
        msg = gr.Textbox(value='Patient: ')
        clear = gr.Button("Clear")

        msg.submit(evaluate, [
            instruction,
            temperature,
            top_p,
            top_k,
            beams,
            max_token,
            msg, chatbot, ], [msg, chatbot])
        clear.click(lambda: None, None, chatbot, queue=False)

    demo.launch(server_name=server_name, share=share_gradio)


if __name__ == "__main__":
    fire.Fire(main)
