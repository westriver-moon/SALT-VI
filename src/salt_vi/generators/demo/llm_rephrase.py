"""Generate one caption rephrase with a FastChat-compatible local model."""
import argparse
import os


def add_model_args(parser):
    parser.add_argument("--model-path", type=str, default="lmsys/vicuna-7b-v1.5")
    parser.add_argument("--revision", type=str, default="main")
    parser.add_argument("--device", choices=["cpu", "cuda", "mps", "xpu", "npu"], default="cuda")
    parser.add_argument("--gpus", type=str, default=None)
    parser.add_argument("--num-gpus", type=int, default=1)
    parser.add_argument("--max-gpu-memory", type=str)
    parser.add_argument("--dtype", choices=["float32", "float16", "bfloat16"], default=None)
    parser.add_argument("--load-8bit", action="store_true")
    parser.add_argument("--cpu-offloading", action="store_true")


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--repetition-penalty", type=float, default=1.0)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--input", required=True, help="Description to rephrase.")
    add_model_args(parser)
    return parser


def load_components(args):
    """Load optional FastChat dependencies only for an explicit CLI invocation."""
    if args.gpus is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    import torch
    from fastchat.model import get_conversation_template, load_model

    if "t5" in args.model_path and args.repetition_penalty == 1.0:
        args.repetition_penalty = 1.2
    model, tokenizer = load_model(
        args.model_path,
        num_gpus=args.num_gpus,
        device=args.device,
        max_gpu_memory=args.max_gpu_memory,
        load_8bit=args.load_8bit,
        cpu_offloading=args.cpu_offloading,
        revision=args.revision,
        debug=args.debug,
    )
    return torch, get_conversation_template, model, tokenizer


def get_augmented_description(description, *, args, torch_module, conversation_template, model, tokenizer):
    prompt = description + "rephrase the person's description above using similar words. Answer:"
    conversation = conversation_template(args.model_path)
    conversation.append_message(conversation.roles[0], prompt)
    conversation.append_message(conversation.roles[1], None)
    with torch_module.no_grad():
        inputs = tokenizer([conversation.get_prompt()], return_tensors="pt").to(args.device)
        output_ids = model.generate(
            **inputs,
            do_sample=args.temperature > 1e-5,
            temperature=args.temperature,
            repetition_penalty=args.repetition_penalty,
            max_new_tokens=args.max_new_tokens,
        )[0]
        if not model.config.is_encoder_decoder:
            output_ids = output_ids[len(inputs["input_ids"][0]):]
        return tokenizer.decode(output_ids, skip_special_tokens=True, spaces_between_special_tokens=False)


def main():
    args = build_parser().parse_args()
    torch_module, conversation_template, model, tokenizer = load_components(args)
    print(get_augmented_description(args.input, args=args, torch_module=torch_module, conversation_template=conversation_template, model=model, tokenizer=tokenizer))


if __name__ == "__main__":
    main()
