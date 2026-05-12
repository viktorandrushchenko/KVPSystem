import os
# СТАВИМ ПЕРВЫМИ СТРОКАМИ!!!
os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # Только GPU 1
os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["NCCL_IB_DISABLE"] = "1"
os.environ["UNSLOTH_WARN_UNINITIALIZED"] = '0'

from unsloth import FastVisionModel
import torch
from transformers import AutoModel

# Проверка - должна быть видна только GPU 1
print(f"Видимые GPU: {torch.cuda.device_count()}")
print(f"Используем: {torch.cuda.get_device_name(0)}")

# Загружаем модель (теперь она видит только GPU 1)
model, tokenizer = FastVisionModel.from_pretrained(
    "./deepseek_ocr",
    load_in_4bit=False,
    auto_model=AutoModel,
    trust_remote_code=True,
    unsloth_force_compile=True,
    use_gradient_checkpointing="unsloth",
    device_map={"": 0},  # Явно указываем использовать единственную видимую карту
)

print(f"Модель загружена на: {model.device}")
print(f"Память: {torch.cuda.memory_allocated(0)/1e9:.2f} GB")

model = FastVisionModel.get_peft_model(
    model,
    target_modules = [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],

    r = 16,           # The larger, the higher the accuracy, but might overfit
    lora_alpha = 16,  # Recommended alpha == r at least
    lora_dropout = 0,
    bias = "none",
    random_state = 3407,
    use_rslora = False,  # We support rank stabilized LoRA
    loftq_config = None, # And LoftQ
    # target_modules = "all-linear", # Optional now! Can specify a list if needed
)

from PIL import Image
import json
import os
from datasets import Dataset

def normalize_bbox(bbox, width, height):
    """Нормализует координаты bbox в диапазон 0-999"""
    if not bbox or len(bbox) != 4:
        return None
    
    x1, y1, x2, y2 = bbox
    norm_x1 = int(x1 * 999 / width)
    norm_y1 = int(y1 * 999 / height)
    norm_x2 = int(x2 * 999 / width)
    norm_y2 = int(y2 * 999 / height)
    
    return [norm_x1, norm_y1, norm_x2, norm_y2]

def convert_kvp10k_to_dataset(gts_folder, images_folder, max_images=None):
    """
    Преобразует KVP10k JSON в официальный формат DeepSeek OCR с тегами
    """
    converted_data = []
    
    json_files = sorted([f for f in os.listdir(gts_folder) if f.endswith('.json')])
    
    if max_images:
        json_files = json_files[:max_images]
    
    print(f"Найдено JSON файлов: {len(json_files)}")
    
    images_processed = 0
    total_pairs = 0
    bbox_count = 0
    
    for json_file in json_files:
        if max_images and images_processed >= max_images:
            break
        
        # Путь к JSON
        json_path = os.path.join(gts_folder, json_file)
        
        # Загружаем JSON
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Ищем изображение
        base_name = os.path.splitext(json_file)[0]
        image_path = None
        for ext in ['.png', '.jpg', '.jpeg']:
            potential = os.path.join(images_folder, base_name + ext)
            if os.path.exists(potential):
                image_path = potential
                break
        
        if not image_path:
            continue
        
        # Получаем размеры изображения (для нормализации bbox)
        with Image.open(image_path) as img:
            width, height = img.size
        
        images_processed += 1
        
        if images_processed % 100 == 0:
            print(f"Обработано изображений: {images_processed}")
        
        # Проходим по парам ключ-значение
        for kvp_item in data.get('kvps_list', []):
            
            # Получаем ключ
            key_text = None
            if 'key' in kvp_item and isinstance(kvp_item['key'], dict):
                key_text = kvp_item['key'].get('text', '').strip()
            
            # Получаем значение и bbox
            value_text = None
            value_bbox_raw = None
            if 'value' in kvp_item and isinstance(kvp_item['value'], dict):
                value_text = kvp_item['value'].get('text', '').strip()
                value_bbox_raw = kvp_item['value'].get('bbox', None)
            
            if not key_text or not value_text:
                continue
            
            # Нормализуем bbox
            if value_bbox_raw and len(value_bbox_raw) == 4:
                norm_bbox = normalize_bbox(value_bbox_raw, width, height)
                if norm_bbox:
                    bbox_count += 1
                    bbox_list = [[norm_bbox[0], norm_bbox[1], norm_bbox[2], norm_bbox[3]]]
                    bbox_str = json.dumps(bbox_list, separators=(',', ':'))
                    assistant_content = f"<|ref|>text<|/ref|><|det|>{bbox_str}<|/det|>{value_text}"
                else:
                    assistant_content = value_text
            else:
                assistant_content = value_text
            
            # Создаем диалог в официальном формате
            conversation = {
                "messages": [
                    {
                        "role": "<|User|>",
                        "content": f"<image>\n{key_text}",
                        "images": [image_path]  # ← ТОЛЬКО ПУТЬ, НЕ PIL!
                    },
                    {
                        "role": "<|Assistant|>",
                        "content": assistant_content
                    }
                ]
            }
            
            converted_data.append(conversation)
            total_pairs += 1
    
    print(f"\n{'='*50}")
    print(f"Обработано изображений: {images_processed}")
    print(f"Создано диалогов: {total_pairs}")
    print(f"С bbox: {bbox_count} ({bbox_count/total_pairs*100:.1f}%)")
    
    # Создаем датасет
    dataset = Dataset.from_list(converted_data)
    
    return dataset

# Использование
converted_dataset = convert_kvp10k_to_dataset(
    "KVP10k/train/gts/", 
    "KVP10k/train/images/",
    max_images=None  # без ограничений
)

# @title Create datacollator

import torch
import math
from dataclasses import dataclass
from typing import Dict, List, Any, Tuple
from PIL import Image, ImageOps
from torch.nn.utils.rnn import pad_sequence
import io

from deepseek_ocr.modeling_deepseekocr import (
    format_messages,
    text_encode,
    BasicImageTransform,
    dynamic_preprocess,
)

@dataclass
class DeepSeekOCRDataCollator:
    """
    Args:
        tokenizer: Tokenizer
        model: Model
        image_size: Size for image patches (default: 640)
        base_size: Size for global view (default: 1024)
        crop_mode: Whether to use dynamic cropping for large images
        train_on_responses_only: If True, only train on assistant responses (mask user prompts)
    """
    tokenizer: Any
    model: Any
    image_size: int = 640
    base_size: int = 1024
    crop_mode: bool = True
    image_token_id: int = 128815
    train_on_responses_only: bool = True

    def __init__(
        self,
        tokenizer,
        model,
        image_size: int = 640,
        base_size: int = 1024,
        crop_mode: bool = True,
        train_on_responses_only: bool = True,
    ):
        self.tokenizer = tokenizer
        self.model = model
        self.image_size = image_size
        self.base_size = base_size
        self.crop_mode = crop_mode
        self.image_token_id = 128815
        self.dtype = model.dtype  # Get dtype from model
        self.train_on_responses_only = train_on_responses_only

        self.image_transform = BasicImageTransform(
            mean = (0.5, 0.5, 0.5),
            std = (0.5, 0.5, 0.5),
            normalize = True
        )
        self.patch_size = 16
        self.downsample_ratio = 4

        # Get BOS token ID from tokenizer
        if hasattr(tokenizer, 'bos_token_id') and tokenizer.bos_token_id is not None:
            self.bos_id = tokenizer.bos_token_id
        else:
            self.bos_id = 0
            print(f"Warning: tokenizer has no bos_token_id, using default: {self.bos_id}")

    def deserialize_image(self, image_data) -> Image.Image:
        """Convert image data (path string, bytes dict or PIL Image) to PIL Image in RGB mode"""
        if isinstance(image_data, Image.Image):
            return image_data.convert("RGB")
        elif isinstance(image_data, str):  # ← ДОБАВЛЕНО: поддержка пути к файлу
            # Загружаем PIL из пути к файлу
            return Image.open(image_data).convert("RGB")
        elif isinstance(image_data, dict) and 'bytes' in image_data:
            image_bytes = image_data['bytes']
            image = Image.open(io.BytesIO(image_bytes))
            return image.convert("RGB")
        else:
            raise ValueError(f"Unsupported image format: {type(image_data)}")

    def calculate_image_token_count(self, image: Image.Image, crop_ratio: Tuple[int, int]) -> int:
        """Calculate the number of tokens this image will generate"""
        num_queries = math.ceil((self.image_size // self.patch_size) / self.downsample_ratio)
        num_queries_base = math.ceil((self.base_size // self.patch_size) / self.downsample_ratio)

        width_crop_num, height_crop_num = crop_ratio

        if self.crop_mode:
            img_tokens = num_queries_base * num_queries_base + 1
            if width_crop_num > 1 or height_crop_num > 1:
                img_tokens += (num_queries * width_crop_num + 1) * (num_queries * height_crop_num)
        else:
            img_tokens = num_queries * num_queries + 1

        return img_tokens

    def process_image(self, image: Image.Image) -> Tuple[List, List, List, List, Tuple[int, int]]:
        """
        Process a single image based on crop_mode and size thresholds

        Returns:
            Tuple of (images_list, images_crop_list, images_spatial_crop, tokenized_image, crop_ratio)
        """
        images_list = []
        images_crop_list = []
        images_spatial_crop = []

        if self.crop_mode:
            # Determine crop ratio based on image size
            if image.size[0] <= 640 and image.size[1] <= 640:
                crop_ratio = (1, 1)
                images_crop_raw = []
            else:
                images_crop_raw, crop_ratio = dynamic_preprocess(
                    image, min_num = 2, max_num = 9,
                    image_size = self.image_size, use_thumbnail = False
                )

            # Process global view with padding
            global_view = ImageOps.pad(
                image, (self.base_size, self.base_size),
                color = tuple(int(x * 255) for x in self.image_transform.mean)
            )
            images_list.append(self.image_transform(global_view).to(self.dtype))

            width_crop_num, height_crop_num = crop_ratio
            images_spatial_crop.append([width_crop_num, height_crop_num])

            # Process local views (crops) if applicable
            if width_crop_num > 1 or height_crop_num > 1:
                for crop_img in images_crop_raw:
                    images_crop_list.append(
                        self.image_transform(crop_img).to(self.dtype)
                    )

            # Calculate image tokens
            num_queries = math.ceil((self.image_size // self.patch_size) / self.downsample_ratio)
            num_queries_base = math.ceil((self.base_size // self.patch_size) / self.downsample_ratio)

            tokenized_image = ([self.image_token_id] * num_queries_base + [self.image_token_id]) * num_queries_base
            tokenized_image += [self.image_token_id]

            if width_crop_num > 1 or height_crop_num > 1:
                tokenized_image += ([self.image_token_id] * (num_queries * width_crop_num) + [self.image_token_id]) * (
                    num_queries * height_crop_num)

        else:  # crop_mode = False
            crop_ratio = (1, 1)
            images_spatial_crop.append([1, 1])

            # For smaller base sizes, resize; for larger, pad
            if self.base_size <= 640:
                resized_image = image.resize((self.base_size, self.base_size), Image.LANCZOS)
                images_list.append(self.image_transform(resized_image).to(self.dtype))
            else:
                global_view = ImageOps.pad(
                    image, (self.base_size, self.base_size),
                    color = tuple(int(x * 255) for x in self.image_transform.mean)
                )
                images_list.append(self.image_transform(global_view).to(self.dtype))

            num_queries = math.ceil((self.base_size // self.patch_size) / self.downsample_ratio)
            tokenized_image = ([self.image_token_id] * num_queries + [self.image_token_id]) * num_queries
            tokenized_image += [self.image_token_id]

        return images_list, images_crop_list, images_spatial_crop, tokenized_image, crop_ratio

    def process_single_sample(self, messages: List[Dict]) -> Dict[str, Any]:
            """
            Process a single conversation into model inputs.
            """

            # --- 1. Setup ---
            images = []
            for message in messages:
                if "images" in message and message["images"]:
                    for img_data in message["images"]:
                        if img_data is not None:
                            pil_image = self.deserialize_image(img_data)
                            images.append(pil_image)

            if not images:
                raise ValueError("No images found in sample. Please ensure all samples contain images.")

            tokenized_str = []
            images_seq_mask = []
            images_list, images_crop_list, images_spatial_crop = [], [], []

            prompt_token_count = -1 # Index to start training
            assistant_started = False
            image_idx = 0

            # Add BOS token at the very beginning
            tokenized_str.append(self.bos_id)
            images_seq_mask.append(False)

            for message in messages:
                role = message["role"]
                content = message["content"]

                # Check if this is the assistant's turn
                if role == "<|Assistant|>":
                    if not assistant_started:
                        # This is the split point. All tokens added *so far*
                        # are part of the prompt.
                        prompt_token_count = len(tokenized_str)
                        assistant_started = True

                    # Append the EOS token string to the *end* of assistant content
                    content = f"{content.strip()} {self.tokenizer.eos_token}"

                # Split this message's content by the image token
                text_splits = content.split('<image>')

                for i, text_sep in enumerate(text_splits):
                    # Tokenize the text part
                    tokenized_sep = text_encode(self.tokenizer, text_sep, bos = False, eos = False)
                    tokenized_str.extend(tokenized_sep)
                    images_seq_mask.extend([False] * len(tokenized_sep))

                    # If this text is followed by an <image> tag
                    if i < len(text_splits) - 1:
                        if image_idx >= len(images):
                            raise ValueError(
                                f"Data mismatch: Found '<image>' token but no corresponding image."
                            )

                        # Process the image
                        image = images[image_idx]
                        img_list, crop_list, spatial_crop, tok_img, _ = self.process_image(image)

                        images_list.extend(img_list)
                        images_crop_list.extend(crop_list)
                        images_spatial_crop.extend(spatial_crop)

                        # Add image placeholder tokens
                        tokenized_str.extend(tok_img)
                        images_seq_mask.extend([True] * len(tok_img))

                        image_idx += 1 # Move to the next image

            # --- 3. Validation and Final Prep ---
            if image_idx != len(images):
                raise ValueError(
                    f"Data mismatch: Found {len(images)} images but only {image_idx} '<image>' tokens were used."
                )

            # If we never found an assistant message, we're in a weird state
            # (e.g., user-only prompt). We mask everything.
            if not assistant_started:
                print("Warning: No assistant message found in sample. Masking all tokens.")
                prompt_token_count = len(tokenized_str)

            # Prepare image tensors
            images_ori = torch.stack(images_list, dim = 0)
            images_spatial_crop_tensor = torch.tensor(images_spatial_crop, dtype = torch.long)

            if images_crop_list:
                images_crop = torch.stack(images_crop_list, dim = 0)
            else:
                images_crop = torch.zeros((1, 3, self.base_size, self.base_size), dtype = self.dtype)

            return {
                "input_ids": torch.tensor(tokenized_str, dtype = torch.long),
                "images_seq_mask": torch.tensor(images_seq_mask, dtype = torch.bool),
                "images_ori": images_ori,
                "images_crop": images_crop,
                "images_spatial_crop": images_spatial_crop_tensor,
                "prompt_token_count": prompt_token_count, # This is now accurate
            }

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """Collate batch of samples"""
        batch_data = []

        # Process each sample
        for feature in features:
            try:
                processed = self.process_single_sample(feature['messages'])
                batch_data.append(processed)
            except Exception as e:
                print(f"Error processing sample: {e}")
                continue

        if not batch_data:
            raise ValueError("No valid samples in batch")

        # Extract lists
        input_ids_list = [item['input_ids'] for item in batch_data]
        images_seq_mask_list = [item['images_seq_mask'] for item in batch_data]
        prompt_token_counts = [item['prompt_token_count'] for item in batch_data]

        # Pad sequences
        input_ids = pad_sequence(input_ids_list, batch_first = True, padding_value = self.tokenizer.pad_token_id)
        images_seq_mask = pad_sequence(images_seq_mask_list, batch_first = True, padding_value = False)

        # Create labels
        labels = input_ids.clone()

        # Mask padding tokens
        labels[labels == self.tokenizer.pad_token_id] = -100

        # Mask image tokens (model shouldn't predict these)
        labels[images_seq_mask] = -100

        # Mask user prompt tokens when train_on_responses_only = True (only train on assistant responses)
        if self.train_on_responses_only:
            for idx, prompt_count in enumerate(prompt_token_counts):
                if prompt_count > 0:
                    labels[idx, :prompt_count] = -100

        # Create attention mask
        attention_mask = (input_ids != self.tokenizer.pad_token_id).long()

        # Prepare images batch (list of tuples)
        images_batch = []
        for item in batch_data:
            images_batch.append((item['images_crop'], item['images_ori']))

        # Stack spatial crop info
        images_spatial_crop = torch.cat([item['images_spatial_crop'] for item in batch_data], dim = 0)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "images": images_batch,
            "images_seq_mask": images_seq_mask,
            "images_spatial_crop": images_spatial_crop,
        }

import os
os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["NCCL_IB_DISABLE"] = "1"

from transformers import Trainer, TrainingArguments
from unsloth import is_bf16_supported
FastVisionModel.for_training(model) # Enable for training!
data_collator = DeepSeekOCRDataCollator(
    tokenizer = tokenizer,
    model = model,
    image_size = 640,
    base_size = 1024,
    crop_mode = True,
    train_on_responses_only = True,
)
trainer = Trainer(
    model = model,
    tokenizer = tokenizer,
    data_collator = data_collator, # Must use!
    train_dataset = converted_dataset,
    args = TrainingArguments(
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,
        warmup_steps = 5,
        #max_steps = 50,
        num_train_epochs = 1, # Set this instead of max_steps for full training runs
        learning_rate = 2e-4,
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.001,
        lr_scheduler_type = "linear",
        seed = 3407,
        fp16 = not is_bf16_supported(),  # Use fp16 if bf16 is not supported
        bf16 = is_bf16_supported(),  # Use bf16 if supported
        output_dir = "outputs",
        report_to = "none",     # For Weights and Biases
        dataloader_num_workers = 0,
        # You MUST put the below items for vision finetuning:
        remove_unused_columns = False,
    ),
)


trainer_stats = trainer.train()