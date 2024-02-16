import os
import platform
import sys
import traceback

import cv2
from tqdm import trange

import llama
import torch
from PIL import Image

# Import utilities from LLaVA
sys.path.insert(0, os.path.abspath('../MLLMEval/'))
sys.path.insert(0, os.path.abspath('../MLLMEval/utility/'))
# sys.path.insert(0, os.path.abspath('.'))

from utility.utils import truncate_input, save, load_existing_results, load_data, print_colored
from prompts.prompts import get_prompt, get_prompt_for_caption
from arguments import parse_args

"""
pip install clip-by-openai --no-deps

Then download the 7B model from https://huggingface.co/nyanko7/LLaMA-7B/tree/main
"""

args = parse_args()

def get_device_for_platform():
    if platform.system() == "Windows":
        device = "cuda:0"
    elif platform.system() == "Darwin":
        device = "mps:0"
    elif platform.system() == "Linux":
        device = "cuda"
    else:
        raise ValueError("Unknown platform.")

    return device

if args.device is None:

    device = get_device_for_platform()

else:
    device = args.device

print("Using device:", device)

if "cuda:" in device:
    torch.cuda.set_device(int(device.split(':')[-1]))

df, image_directory_name = load_data(split=args.split, data_dir=args.data_dir, dataset_name=args.dataset_name)




# Path to the LLaMA-7B model.
# Note that this is not the huggingface version.
llama_dir = "../../models"

if not args.debug:
    # choose from BIAS-7B, LORA-BIAS-7B, CAPTION-7B.pth
    model, preprocess = llama.load("BIAS-7B", llama_dir, llama_type="7B", device=device)
    model.eval()

else:
    model = None
    preprocess = None

existing_results = load_existing_results(args.answers_file)

START = 0 if existing_results is None else len(existing_results)

answers_li = []

for idx in trange(START, len(df), args.batch_size):
    line = df.iloc[idx]

    explanation = caption = pred = None

    context = ""

    for question_type in args.question_types:

        try:

            if question_type == "question":
                question, image_file = get_prompt(args, line)

            elif question_type == "caption":
                question = get_prompt_for_caption()
                _, image_file = get_prompt(args, line)


            elif question_type == "explain":
                question = "Explain your reasons in details."

            else:
                raise ValueError

            if question_type == "explain":
                prompt = f"{context}{question}"

            else:

                prompt = llama.format_prompt(question)

            print_colored(prompt, "blue")

            img = Image.fromarray(cv2.imread(image_file))
            if img is None:
                raise ValueError("Image not found.")

            img = preprocess(img).unsqueeze(0).to(device)
            generated_text = model.generate(img, [prompt])[0]


            if question_type == "question":
                pred = generated_text

                generated_text = truncate_input(generated_text, 30)


            elif question_type == "caption":
                caption = generated_text
                generated_text = truncate_input(generated_text, 128)
                print_colored(caption, "yellow")

                if len(args.question_types) == 1:
                    break


            elif question_type == "explain":
                explanation = generated_text
                generated_text = truncate_input(generated_text, 128)

            context += f"{generated_text}"
            print_colored(generated_text, "yellow")





        except:
            traceback.print_exc()


    answer = {
        'idx': idx,
        "pred": pred,
        "caption": caption,
        "explanation": explanation,
    }

    answers_li += [answer]

    if (idx + 1) % args.save_every == 0:
        save(answers_li, args.answers_file, existing_results)

save(answers_li, args.answers_file, existing_results)
