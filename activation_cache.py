import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModel
import os
from tqdm import trange
import gc
from nnsight import LanguageModel


def get_token_batch(
    dataset_iterator, tokenizer, num_documents_per_batch: int, tokens_per_document: int
):
    tokens_BP = torch.zeros(
        (num_documents_per_batch, tokens_per_document), dtype=torch.long
    )

    cnt = 0
    while cnt < num_documents_per_batch:
        text = next(dataset_iterator)["text"]
        tokens = tokenizer.encode(text, return_tensors="pt")[0]
        if len(tokens) < tokens_per_document:
            continue

        tokens_BP[cnt] = tokens[:tokens_per_document]
        cnt += 1

    return tokens_BP


class ActivationHook:
    """Hook class to capture activations from a specific layer"""

    def __init__(self):
        self.activations = None

    def __call__(self, module, input, output):
        # Store the output activations (detach to avoid keeping gradients)
        if isinstance(output, tuple):
            output = output[0]
        self.activations = output.detach()


def get_batch_act_with_hooks(
    model,
    target_layer,
    batch_inputs_BP,
    device,
):
    # Set up the hook on the target layer
    hook = ActivationHook()
    target_module = model.layers[target_layer]
    hook_handle = target_module.register_forward_hook(hook)

    with torch.inference_mode(), torch.autocast(
        device_type=device, dtype=torch.bfloat16
    ):
        _ = model(batch_inputs_BP)

    return hook.activations

def get_batch_act_with_nnsight(
    model,
    target_layer,
    batch_inputs_BP,
    device,
):
    with (
        torch.inference_mode(),
        model.trace(batch_inputs_BP, scan=False, validate=False),
    ):
        act_batch = model.model.layers[target_layer].mlp.act_fn.output[0].save()
    return act_batch

def get_lm_activations(
    model,
    tokenizer,
    target_layer,
    dataset_iterator,
    num_total_tokens,
    num_tokens_per_batch,
    num_tokens_per_file,
    num_tokens_per_document,
    gemma_layer_width,
    save_dir,
    device,
    use_nnsight,
):
    """Optimized version using hooks to capture only target layer activations"""

    assert (
        num_total_tokens % num_tokens_per_file == 0
    ), "num_total_tokens must be divisible by num_tokens_per_file"
    assert (
        num_tokens_per_file % num_tokens_per_batch == 0
    ), "num_tokens_per_file must be divisible by num_tokens_per_batch"
    assert (
        num_tokens_per_batch % num_tokens_per_document == 0
    ), "num_tokens_per_batch must be divisible by num_tokens_per_document"

    num_files = num_total_tokens // num_tokens_per_file
    num_batches_per_file = num_tokens_per_file // num_tokens_per_batch
    num_documents_per_file = num_tokens_per_file // num_tokens_per_document
    num_documents_per_batch = num_tokens_per_batch // num_tokens_per_document

    for i in range(num_files):
        print(f"Processing save file {i} of {num_files}")
        save_file_name = f"{save_dir}/lm_activations_{i}_of_{num_files}_tokens{num_tokens_per_file}.pt"
        acts_lm_BPD = torch.zeros(
            num_documents_per_file,
            num_tokens_per_document,
            gemma_layer_width,
            dtype=torch.bfloat16,
        )

        for j in trange(num_batches_per_file, desc="Processing batches"):
            batch_start = j * num_documents_per_batch
            batch_end = batch_start + num_documents_per_batch

            batch_tokens = get_token_batch(
                dataset_iterator,
                tokenizer,
                num_documents_per_batch,
                num_tokens_per_document,
            )
            batch_tokens = batch_tokens.to(device)

            if use_nnsight:
                act_batch = get_batch_act_with_nnsight(
                    model,
                    target_layer,
                    batch_tokens,
                    device,
                )
            else:
                act_batch = get_batch_act_with_hooks(
                    model,
                    target_layer,
                    batch_tokens,
                    device,
                )
            act_batch = act_batch.cpu()
            acts_lm_BPD[batch_start:batch_end] = act_batch

            # Clean up
            del act_batch, batch_tokens
            torch.cuda.empty_cache()
            gc.collect()

        torch.save(acts_lm_BPD, save_file_name)

        del acts_lm_BPD
        torch.cuda.empty_cache()
        gc.collect()



def load_gemma(dtype, cache_dir, device, use_nnsight=False):
    """Load Gemma model with optional hook optimization"""
    if use_nnsight:
        print("Loading NNsight Gemma-2-2B...")
        model = LanguageModel(
            "google/gemma-2-2b",
            cache_dir=cache_dir,
            torch_dtype=dtype,
            device_map=device,
            dispatch=True
        )
        tokenizer = model.tokenizer
    else:
        print("Loading Huggingface Gemma-2-2B...")
        tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b", cache_dir=cache_dir)
        model = AutoModel.from_pretrained(
            "google/gemma-2-2b", cache_dir=cache_dir, torch_dtype=dtype
        )

    model.to(device)
    model.eval()

    return model, tokenizer


if __name__ == "__main__":

    # Setup
    DEVICE = "cuda"
    CACHE_DIR = "/home/can/models"

    DATASET_NAME = "monology/pile-uncopyrighted"
    TARGET_LAYER = 12
    DTYPE = torch.bfloat16
    GEMMA_LAYER_WIDTH = 9216

    USE_NNSIGHT = True

    DEBUG = True
    if DEBUG:
        SAVE_DIR = "precomputed_activations_debug"
        NUM_TOTAL_TOKENS = 2000
        SAVE_SIZE = 2000
        BATCH_SIZE = 100
        NUM_TOKENS_PER_DOCUMENT = 10
    else:
        SAVE_DIR = "precomputed_activations"

        NUM_TOTAL_TOKENS = 100_000_000
        SAVE_SIZE = 1_000_000
        NUM_TOKENS_PER_DOCUMENT = 100
        BATCH_SIZE = 100_000

    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)

    # Load dataset
    dataset_hf = load_dataset(DATASET_NAME, "default", streaming=True)
    dataset_iterator = iter(dataset_hf["train"])

    # Load model
    model, tokenizer = load_gemma(dtype=DTYPE, cache_dir=CACHE_DIR, device=DEVICE, use_nnsight=USE_NNSIGHT)

    # Get activations
    get_lm_activations(
        model=model,
        tokenizer=tokenizer,
        target_layer=TARGET_LAYER,
        dataset_iterator=dataset_iterator,
        num_total_tokens=NUM_TOTAL_TOKENS,
        num_tokens_per_batch=BATCH_SIZE,
        num_tokens_per_file=SAVE_SIZE,
        num_tokens_per_document=NUM_TOKENS_PER_DOCUMENT,
        gemma_layer_width=GEMMA_LAYER_WIDTH,
        save_dir=SAVE_DIR,
        device=DEVICE,
        use_nnsight=USE_NNSIGHT
    )
