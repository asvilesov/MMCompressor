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
        model_id = "microsoft/Phi-3.5-vision-instruct" 

        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, 
            device_map="cuda", 
            trust_remote_code=True, 
            torch_dtype="auto", 
            # _attn_implementation='flash_attention_2'    
        )
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
        self.processor = AutoProcessor.from_pretrained(model_id, 
            trust_remote_code=True, 
            num_crops=4
        ) 
        self.tokenizer = self.processor.tokenizer
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
    
    def forward(self, input_ids: torch.LongTensor = None,
                    attention_mask: Optional[torch.Tensor] = None,
                    position_ids: Optional[torch.LongTensor] = None,
                    past_key_values: Optional[List[torch.FloatTensor]] = None,
                    inputs_embeds: Optional[torch.FloatTensor] = None,
                    labels: Optional[torch.LongTensor] = None,
                    use_cache: Optional[bool] = None,
                    output_attentions: Optional[bool] = None,
                    output_hidden_states: Optional[bool] = None,
                    return_dict: Optional[bool] = None,
                    pixel_values: Optional[torch.Tensor] = None,
                    image_sizes: Optional[torch.Tensor] = None,
                    cache_position: Optional[torch.LongTensor] = None):
        
        # A. Initial Forward Pass to get KV Cache
        output = self.model(input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            pixel_values=pixel_values,
            image_sizes=image_sizes,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            labels=labels)

        # KV_CACHE.shape = num_layer, key and value , batch_size, heads, seq_len, hidden_dim
        past_key_values = output.past_key_values
        num_layers = len(past_key_values)
        
        # B. Find position of last image token in sequence
        ends = []
        for i in range(input_ids.size(0)):
            ends.append((input_ids[i] == -1).nonzero(as_tuple=True)[0][-1])

        # C. Prepare new KV Cache (will form the input to the next forward pass through 'past_key_values')
        new_kv = []
        if(self.model.training):
            self.current_step += 1
        else:
            pass
        start_idx = int(self.current_step / self.tau)+1
        if(2**start_idx < ends[0]): # if number of tokens to compress is greater than 1
            end_idx = ends[0]
            token_idxs = torch.concatenate((torch.arange(0, end_idx, 2**start_idx), torch.tensor([end_idx])))
            for i in range(num_layers):
                new_kv.append([past_key_values[i][0][:,:,token_idxs,:],   # key
                                past_key_values[i][1][:,:,token_idxs,:]]) # value
        else: # if number of tokens to compress is 1
            start_idx = ends[0]
            end_idx = ends[0]+1
            for i in range(num_layers):
                new_kv.append([past_key_values[i][0][:,:,start_idx:end_idx,:],
                                past_key_values[i][1][:,:,start_idx:end_idx,:]])
        
        # D. Second Forward Pass using new KV Cache without image tokens
        output = self.model(input_ids=input_ids[:,ends[0]+1:],
            attention_mask=attention_mask[:,ends[0]+1:],
            position_ids=None,
            past_key_values=new_kv,
            inputs_embeds=None,
            pixel_values=None,
            image_sizes=None,
            use_cache=use_cache,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=return_dict,
            labels=labels[:,ends[0]+1:])

        return output
    
    @torch.no_grad()
    def generate(self, inputs):

        # A. Find position of last image token in sequence
        ends = []
        input_ids = inputs["input_ids"]
        for i in range(input_ids.size(0)):
            ends.append((input_ids[i] == -1).nonzero(as_tuple=True)[0][-1])
        
        # B. Initial Forward Pass to get KV Cache
        tensor_dict_input = {key: value[:,:ends[0]+1].to("cuda") for key, value in inputs.items()}
        output = self.model.forward(**tensor_dict_input)

        past_key_values = output.past_key_values
        num_layers = len(past_key_values)
        
        # C. Prepare new KV Cache (will form the input to the next forward pass through 'past_key_values') 
        new_kv = []
        start_idx = int(self.current_step / self.tau)+1
        if(2**start_idx < ends[0]):
            end_idx = ends[0]
            token_idxs = torch.concatenate((torch.arange(0, end_idx, 2**start_idx), torch.tensor([end_idx])))
            for i in range(num_layers):
                new_kv.append([past_key_values[i][0][:,:,token_idxs,:],
                                past_key_values[i][1][:,:,token_idxs,:]])
        else:
            start_idx = ends[0]
            end_idx = ends[0]+1
            for i in range(num_layers):
                new_kv.append([past_key_values[i][0][:,:,start_idx:end_idx,:],
                                past_key_values[i][1][:,:,start_idx:end_idx,:]])

        # D. Generate Tokens using new KV Cache without image tokens
        tensor_dict_output = {key: value[:,ends[0]+1:].to("cuda") for key, value in inputs.items()}
        generated_ids = self.model.generate(input_ids=tensor_dict_output["input_ids"], 
                                            max_new_tokens=50, past_key_values=new_kv, 
                                            do_sample=True, temperature=0.7,
                                            eos_token_id=self.processor.tokenizer.eos_token_id, max_time=10)
        
        # E. Decode the generated tokens
        output_text = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        
        return output_text
