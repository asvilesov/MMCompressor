import torch
import torch.nn as nn
from peft import get_peft_model
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoProcessor
import torch
from transformers import Qwen2VLForConditionalGeneration, Qwen2VLProcessor
from transformers.cache_utils import DynamicCache
from typing import List, Optional, Tuple, Union
import torch.nn.functional as F


def load_lora_parameters(model, lora_params_path):
    """
    Initialize the LoRA parameters.

    model (AutoModelForCausalLM): LLM with LoRA parameters.
    lora_params_path (str): Pretrained LoRA parameters path for initialization.
    """
    # load the pretrained LoRA parameters
    lora_params = torch.load(lora_params_path)

    # initialize the LoRA parameters and the compressed token in the LLM
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name in lora_params: 
                if 'lora' in name or 'memory_embeddings' in name:
                    param.copy_(lora_params[name])
                else:
                    print(f"No saved parameter for {name}")
            elif "lora" in name:
                print(f"No saved parameter for {name}")

class VideoCompressor(nn.Module):
    def __init__(self,
        num_mem,
        device,
        tokenizer, 
        lora_config=None,
        freeze_vision_encoder=False,
    ):
        """
        Create the compression model: LLM-LoRA + LLM.

        Args:
            llama_path (str): Path for the base LLM.
            max_context_length (int): Max number of context tokens to be compressed.
            lora_path (sttr): Pretrained LoRA parameters for initialization.
            lora_config (LoraConfig): LoRA configurations.
            num_mem (int): Number of compressed tokens.
            device (torch.device): CPU or GPU.
        """
        super(VideoCompressor, self).__init__()
        # load the original base LLaMA model
        model_id = "DAMO-NLP-SG/VideoLLaMA3-2B"

        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, 
            trust_remote_code=True,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
        )
        # self.model.resize_token_embeddings(len(tokenizer))
        # add LoRA parameters to the LLM
        if(lora_config is not None):
            self.model = get_peft_model(self.model, lora_config)
            # only LoRA parameters are trainable
            for name, param in self.model.named_parameters():
                param.requires_grad = False
                if 'lora' in name:
                    param.requires_grad = True
            print(f"Total parameters of phi: {sum(p.numel() for p in self.model.parameters())}")
        # Freeze models weights for the video encoder 
        if freeze_vision_encoder:
            for param in self.model.model.vision_encoder.parameters():
                param.requires_grad = False
            for param in self.model.model.mm_projector.parameters():
                param.requires_grad = False
        # load the tokenizer
        self.tokenizer = tokenizer
        print("phi tokenizer loaded.")
        # set the padding token
        self.tokenizer.pad_token = self.tokenizer.eos_token
        # max number of context tokens to be compressed
        self.criterion = nn.CrossEntropyLoss(ignore_index=-100)
        # initialize the LoRA parameters
        self.device = device
        # num memory tokens 
        self.num_mem = num_mem
    
    #needs vision encoder
#https://huggingface.co/DAMO-NLP-SG/VL3-SigLIP-NaViT

    def forward(self, **kwargs):
        """
        Multi-modal forward pass that processes memory across time steps using KV cache.

        The batch includes:
        - "memory_save": Multi-modal memory (text + image) for compression.
        - "QA": Downstream QA task with corresponding labels.
        
        Memory tokens are pre-handled in `collate_fn`, so we assume they are structured properly.
        """
        batch = kwargs
        num_time_steps = len(batch)
        new_kv = None  # KV cache to be passed across time steps
        total_loss = 0

        for t in range(num_time_steps):
            curr_batch = batch[f"T{t}"]
            
            # ----- A. Memory Save Phase -----
            # Forward pass using memory-aware inputs
            output = self.model(
                **curr_batch["memory_save"],
                past_key_values=new_kv,
                use_cache=True  # Ensure KV cache is stored
            )
            # ----- B. Update KV Cache -----
            past_key_values = output.past_key_values
            num_layers = len(past_key_values)
            new_kv = []
            num_tokens = past_key_values[0][0].shape[2]
            # Dynamically extract memory tokens from KV cache
            mem_token_start = num_tokens - 2 - self.num_mem  # Memory token position
            mem_token_end = num_tokens - 2
            for layer in range(num_layers):
                key_slice = past_key_values[layer][0][:, :, mem_token_start:mem_token_end, :]
                value_slice = past_key_values[layer][1][:, :, mem_token_start:mem_token_end, :]
                new_kv.append((key_slice, value_slice))
            new_kv = tuple(new_kv)
            kv_collater = DynamicCache()
            new_kv = kv_collater.from_legacy_cache(new_kv)
        
        for t in range(num_time_steps):
            new_kv_qa = kv_collater.from_legacy_cache(new_kv) # We create 2 KV caches since one gets updated during forward pass. 
            curr_batch = batch[f"T{t}"]
            # ----- C. QA Phase -----
            qa_input_ids = curr_batch["QA"]["input_ids"]
            qa_attention_mask = curr_batch["QA"]["attention_mask"]

            # Forward pass for QA using memory-enhanced KV cache
            qa_output = self.model(
                input_ids=qa_input_ids,
                attention_mask=qa_attention_mask,
                past_key_values=new_kv_qa,
                labels=curr_batch["QA"]["labels"],
                return_dict=True,
                use_cache=True  # No need to update KV cache further
            )
            total_loss += qa_output.loss

        # Return accumulated loss across all time steps
        qa_output.loss = total_loss
        return qa_output

        
    @torch.no_grad()
    def generate(self, **kwargs):
        batch = kwargs

        num_t = len(list(batch.keys()))
        new_kv = None
        output_dict = {}
        for i in range(num_t):
            curr_batch = batch[f"T{i}"]
            # A. Initial Forward Pass to get KV Cache
            output = self.model(**curr_batch["memory_save"],
                                past_key_values = new_kv,
                                use_cache=True)
            # B. Prepare new KV Cache (will form the input to the next forward pass through 'past_key_values')
            # KV_CACHE.shape = num_layer, key and value , batch_size, heads, seq_len, hidden_dim
            past_key_values = output.past_key_values
            num_layers = len(past_key_values)
            new_kv = []
            num_tokens = past_key_values[0][0].shape[2]
            # Dynamically extract memory tokens from KV cache
            mem_token_start = num_tokens - 2 - self.num_mem  # Memory token position
            mem_token_end = num_tokens - 2
            for j in range(num_layers):
                new_kv.append((past_key_values[j][0][:,:,mem_token_start:mem_token_end,:],
                                past_key_values[j][1][:,:,mem_token_start:mem_token_end,:]))
            new_kv = tuple(new_kv) #None
            kv_collater = DynamicCache()
            new_kv = kv_collater.from_legacy_cache(new_kv)
            new_kv_qa = kv_collater.from_legacy_cache(new_kv) # We create 2 KV caches since one gets updated during forward pass.
            # C. Generate Tokens using new KV Cache without image tokens
            input_ids = curr_batch["QA"]["input_ids"]
            labels = curr_batch["QA"]["labels"]
            idx_q = torch.sum(labels==-100)
            input_ids = input_ids[:,:idx_q].cuda()
            # generated_ids = self.model.generate(input_ids=input_ids, 
            #                                     max_new_tokens=50, past_key_values=new_kv, 
            #                                     do_sample=True, temperature=0.7,
            #                                     eos_token_id=self.tokenizer.eos_token_id, max_time=10)
            # D. Decode the generated tokens
            # output_text = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
            output_text = self.custom_generate(input_ids, 50, new_kv_qa)
            output_dict[f"T{i}"] = output_text
        
        return output_dict

    def custom_generate(self, input_ids, max_new_tokens, past_key_values):
        token_output_list = []
        #initial pass
        output = self.model.forward(input_ids=torch.tensor(input_ids).cuda(), past_key_values=past_key_values, use_cache=True)
    
        max_idx = [[198]]
        for i in range(max_new_tokens):
            output = self.model.forward(input_ids=torch.tensor(max_idx).cuda(), past_key_values=past_key_values, use_cache=True)
            max_idx=torch.argmax(output.logits, dim=-1)
            # print(self.tokenizer.convert_ids_to_tokens(max_idx))
            # print(max_idx)
            token_output_list.append(max_idx[0][0])
        output_text = self.tokenizer.decode(token_output_list)
        return output_text
        
    # def forward(self, **kwargs):
    #     """
    #     Multi-modal forward pass that processes memory across time steps using KV cache.

    #     The batch includes:
    #     - "memory_save": Multi-modal memory (text + image) for compression.
    #     - "QA": Downstream QA task with corresponding labels.
        
    #     Memory tokens are pre-handled in `collate_fn`, so we assume they are structured properly.
    #     """
    #     batch = kwargs
    #     num_time_steps = len(batch)
    #     new_kv = None  # KV cache to be passed across time steps
    #     total_loss = 0

    #     for t in range(num_time_steps):
    #         curr_batch = batch[f"T{t}"]
            
    #         # ----- A. Memory Save Phase -----
    #         # Forward pass using memory-aware inputs
    #         output = self.model(
    #             **curr_batch["memory_save"],
    #             past_key_values=new_kv,
    #             use_cache=True  # Ensure KV cache is stored
    #         )
            
    #         total_loss += output.loss

    #     # Return accumulated loss across all time steps
    #     output.loss = total_loss
    #     return output

    # @torch.no_grad()
    # def generate(self, **kwargs):
    #     batch = kwargs

    #     num_t = len(list(batch.keys()))
    #     new_kv = None
    #     output_dict = {}
    #     for i in range(num_t):
    #         curr_batch = batch[f"T{i}"]
            
    #         input_ids = curr_batch["memory_save"]["input_ids"]
    #         labels = curr_batch["memory_save"]["labels"]
    #         idx_q = torch.sum(labels==-100)
    #         input_ids = input_ids[:,:idx_q].cuda()
    #         curr_batch["memory_save"]["input_ids"] = input_ids
    #         curr_batch["memory_save"]["attention_mask"] = curr_batch["memory_save"]["attention_mask"][:,:idx_q].cuda()
    #         generated_ids = self.model.generate(**curr_batch["memory_save"],
    #                                             max_new_tokens=50, 
    #                                             do_sample=True, temperature=0.7,
    #                                             eos_token_id=self.tokenizer.eos_token_id, max_time=10)
    #         # D. Decode the generated tokens
    #         output_text = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    #         output_dict[f"T{i}"] = output_text
        
    #     return output_dict
