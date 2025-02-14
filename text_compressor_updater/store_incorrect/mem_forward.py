import torch
import torch.nn as nn
from peft import get_peft_model
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoProcessor
import torch
from transformers import Qwen2VLForConditionalGeneration, Qwen2VLProcessor
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

class PhiCompressor(nn.Module):
    def __init__(self,
        num_mem,
        device,
        tokenizer, 
        lora_config=None,
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
        super(PhiCompressor, self).__init__()
        # load the original base LLaMA model
        model_id = "microsoft/Phi-3.5-mini-instruct" 

        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, 
            device_map="cuda", 
            trust_remote_code=True, 
            torch_dtype="auto", 
            # _attn_implementation='flash_attention_2'    
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
        
        # load the tokenizer
        self.tokenizer = tokenizer
        print("phi tokenizer loaded.")
        # set the padding token
        self.tokenizer.pad_token = self.tokenizer.eos_token
        # max number of context tokens to be compressed
        self.criterion = nn.CrossEntropyLoss(ignore_index=-100)
        # initialize the LoRA parameters
        self.device = device
        # for every self.tau steps, decrease the number of tokens to compress to by half until 1 token is reached.
        self.current_step = 1000
        self.tau = 100 # only used for initial training for stability
        # num memory tokens 
        self.num_mem = num_mem
    
    def forward(self, **kwargs):
        batch = kwargs
        
        # A. Initial Forward Pass to get KV Cache
        output = self.model(**batch["memory_save"])

        # KV_CACHE.shape = num_layer, key and value , batch_size, heads, seq_len, hidden_dim
        past_key_values = output.past_key_values
        num_layers = len(past_key_values)
        # print(len(past_key_values), len(past_key_values[0]), past_key_values[0][0].shape)
        # return past_key_values

        # C. Prepare new KV Cache (will form the input to the next forward pass through 'past_key_values')
        new_kv = []
        start_idx = batch["memory_save"]["input_ids"].size(1) - self.num_mem - 2 #end tokens
        end_idx = batch["memory_save"]["input_ids"].size(1) - 2
        for i in range(num_layers):
            new_kv.append((past_key_values[i][0][:,:,start_idx:end_idx,:],
                            past_key_values[i][1][:,:,start_idx:end_idx,:]))
        new_kv = tuple(new_kv)
        
        # print("new_kv", new_kv[0][0].shape)

        batch["QA"]["attention_mask"] = torch.concatenate((torch.ones((batch["QA"]["attention_mask"].size(0)), self.num_mem).to("cuda"), batch["QA"]["attention_mask"]), dim=1)
        # D. Second Forward Pass using new KV Cache without image tokens
        output = self.model(input_ids=batch["QA"]["input_ids"],
            attention_mask=batch["QA"]["attention_mask"],
            position_ids=None,
            past_key_values=new_kv,
            inputs_embeds=None,
            use_cache=None,
            output_attentions=True,
            output_hidden_states=True,
            return_dict=True,
            labels=batch['labels'])

        return output
    
    @torch.no_grad()
    def generate(self, **kwargs):
        batch = kwargs

        # A. Initial Forward Pass to get KV Cache
        for key, value in batch["memory_save"].items():
            batch["memory_save"][key] = value.cuda()  # Direct assignment works best
        output = self.model(**batch["memory_save"])

        # KV_CACHE.shape = num_layer, key and value , batch_size, heads, seq_len, hidden_dim
        past_key_values = output.past_key_values
        num_layers = len(past_key_values)
        # print(len(past_key_values), len(past_key_values[0]), past_key_values[0][0].shape)
        # return past_key_values

        # C. Prepare new KV Cache (will form the input to the next forward pass through 'past_key_values')
        new_kv = []
        start_idx = batch["memory_save"]["input_ids"].size(1) - self.num_mem - 2 #end tokens
        end_idx = batch["memory_save"]["input_ids"].size(1) - 2
        for i in range(num_layers):
            new_kv.append((past_key_values[i][0][:,:,start_idx:end_idx,:],
                            past_key_values[i][1][:,:,start_idx:end_idx,:]))
        new_kv = tuple(new_kv)

        # D. Generate Tokens using new KV Cache without image tokens
        input_ids = batch["QA"]["input_ids"]
        labels = batch["labels"]
        idx_q = torch.sum(labels==-100)
        input_ids = input_ids[:,:idx_q].cuda()
        generated_ids = self.model.generate(input_ids=input_ids, 
                                            max_new_tokens=50, past_key_values=new_kv, 
                                            do_sample=True, temperature=0.7,
                                            eos_token_id=self.tokenizer.eos_token_id, max_time=10)
        
        # E. Decode the generated tokens
        output_text = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        
        return output_text
