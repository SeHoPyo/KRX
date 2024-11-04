from unsloth import FastLanguageModel
import torch
max_seq_length = 2048
dtype = None # None으로 지정할 경우 해당 컴퓨팅 유닛에 알맞은 dtype으로 저장됩니다. Tesla T4와 V100의 경우에는 Float16, Ampere+ 이상의 경우에는 Bfloat16으로 설정됩니다.
load_in_4bit = True # 메모리 사용량을 줄이기 위해서는 4bit 양자화를 사용하실 것을 권장합니다.

# 모델 및 토크나이저 선언
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/Meta-Llama-3.1-8B",
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
    # token = "hf_...", # gated model을 사용할 경우 허깅페이스 토큰을 입력해주시길 바라겠습니다.
)

# LoRA Adapter 선언
model = FastLanguageModel.get_peft_model(
    model,
    r = 16, # 0을 넘는 숫자를 선택하세요. 8, 16, 32, 64, 128이 추천됩니다.
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",], # target module도 적절하게 조정할 수 있습니다.
    lora_alpha = 16,
    lora_dropout = 0, # 어떤 값이든 사용될 수 있지만, 0으로 최적화되어 있습니다.
    bias = "none",    # 어떤 값이든 사용될 수 있지만, "none"으로 최적화되어 있습니다.
    use_gradient_checkpointing = "unsloth", # 매우 긴 context에 대해 True 또는 "unsloth"를 사용하십시오.
    random_state = 42,
    use_rslora = False,
    loftq_config = None
)

prompt_format = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Response:
{}"""

EOS_TOKEN = tokenizer.eos_token
def formatting_prompts_func(examples):
    instructions = examples["prompt"]
    outputs = examples["response"]
    texts = []
    for instruction, output in zip(instructions, outputs):
        text = prompt_format.format(instruction, output) + EOS_TOKEN # 마지막에 eos token을 추가해줌으로써 모델이 출력을 끝마칠 수 있게 만들어 줍니다.
        texts.append(text)
    return { "formatted_text" : texts, }
pass

from datasets import load_dataset
dataset = load_dataset("amphora/krx-sample-instructions", split = "train")
dataset = dataset.map(formatting_prompts_func, batched = True,)

from trl import SFTTrainer
from transformers import TrainingArguments
from unsloth import is_bfloat16_supported

trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    dataset_text_field = "formatted_text",
    max_seq_length = max_seq_length,
    dataset_num_proc = 2,
    packing = False, # True로 설정하면 짧은 텍스트 데이터에 대해서는 더 빠른 학습 속도로를 보여줍니다.
    args = TrainingArguments( # TrainingArguments는 자신의 학습 환경과 기호에 따라 적절하게 설정하면 됩니다.
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,
        warmup_steps = 5,
        # num_train_epochs = 1,
        max_steps = 60,
        learning_rate = 2e-4,
        fp16 = not is_bfloat16_supported(),
        bf16 = is_bfloat16_supported(),
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 42,
        output_dir = "outputs",
    ),
)


trainer_stats = trainer.train()