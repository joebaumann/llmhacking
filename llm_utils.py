import copy
import os
from munch import Munch
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
import torch
from openai import OpenAI
import pdb
import json
import time

from pathlib import Path

from dotenv import load_dotenv
load_dotenv()


def free_gpu_memory(items_to_delete):
    for item in items_to_delete:
        del item
    torch.cuda.empty_cache()


def move_to_device(model):
    # Move model to the correct device
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    print(f"Moving model to device: {device}")
    if model.device.type != device:
        model.to(device)


class ChatBot:
    def __init__(self, model_name_or_path, config=None, api_key=None, seed=None):
        if config is None:
            config = Munch({'SEED': seed})
        args = [model_name_or_path, config, api_key]
        if 'gpt' in model_name_or_path:
            self.chatbot = _OpenAI(*args)
        else:
            self.chatbot = _HFModel(*args)

    def update_seed(self, seed):
        if seed is not None:
            self.seed = seed

    def initialize_prompt_history(self, prompt):
        # check if prompt is a string or a list of messages
        if isinstance(prompt, str):
            return [{"role": "user", "content": prompt}]
        elif isinstance(prompt, list):
            return [self.initialize_prompt_history(p) for p in prompt]
        elif isinstance(prompt, dict):
            if "role" not in prompt:
                prompt["role"] = "user"
            elif prompt["role"] not in ["user", "assistant", "system"]:
                raise ValueError(f"Invalid role: {prompt['role']}")
            if set(prompt.keys()) != set(["role", "content"]):
                raise ValueError(
                    f"Prompt has invalid keys: {prompt.keys()}. Only 'role' and 'content' are allowed, right?")
            return prompt
        else:
            raise ValueError(f"Invalid prompt type: {type(prompt)}")

    def get_temperature_range(self):
        raise NotImplementedError(
            "This method should be implemented in subclasses.")

    def get_temperature(self, temperature):
        if temperature is None:
            return None
        min_temperature, max_temperature = self.get_temperature_range()
        if temperature == 'min':
            return min_temperature
        elif temperature == 'max':
            return max_temperature
        elif temperature > max_temperature:
            raise ValueError(
                f"Temperature {temperature} is higher than the maximum allowed temperature {max_temperature}(min temperature: {min_temperature}).")
        else:
            return temperature


class _HFModel(ChatBot):
    def __init__(self, model_name_or_path, config=None, api_key=None, seed=None):
        self.model_name = model_name_or_path
        if seed is not None:
            set_seed(seed)
        elif config is not None and config.SEED is not None:
            set_seed(config.SEED)
        if api_key is None:
            api_key = os.getenv('HF_ACCESS_TOKEN')
        local_model_directory = os.getenv('LOCAL_MODEL_PATH')
        self.model, self.tokenizer = self.load_model_and_tokenizer(
            model_name_or_path, token=api_key, local_model_directory=local_model_directory)

    def load_model_and_tokenizer(self, model_name, token, local_model_directory, **kwargs):
        # print(f"\n-------\n  Loading model and tokenizer {model_name}\n-------\n")
        model_kwargs = {
            "device_map": "auto",
            "trust_remote_code": True,
        }
        if local_model_directory is not None:
            model_path = local_model_directory + model_name.replace("/", "--")
        else:
            model_path = model_name
        model_kwargs.update(kwargs)
        if token is not None:
            model_kwargs.update({"token": token})
        tokenizer_kwargs = copy.deepcopy(model_kwargs)
        model, tokenizer = None, None

        if "google/gemma-3-" in model_name:
            # fixing gemma 3 error https://github.com/google-deepmind/gemma/issues/169
            torch.backends.cuda.enable_mem_efficient_sdp(False)
            torch.backends.cuda.enable_flash_sdp(False)
            torch.backends.cuda.enable_math_sdp(True)

            model_kwargs.update({'torch_dtype': torch.bfloat16})

        elif "Qwen" in model_name:
            model_kwargs.update({'torch_dtype': 'auto'})
        else:
            # print("No special model handling")
            model_kwargs.update(
                {'torch_dtype': torch.bfloat16 if self._is_bf16_compatible() else torch.float16})
        if tokenizer is None:
            tokenizer = AutoTokenizer.from_pretrained(
                model_path, **tokenizer_kwargs)
        if model is None:
            model = AutoModelForCausalLM.from_pretrained(
                model_path, **model_kwargs)

        # Add pad token to the tokenizer if it doesn't already exist
        self._set_pad_token_id(tokenizer)
        model.eval()
        move_to_device(model)
        return model, tokenizer

    def _set_pad_token_id(self, tokenizer):
        """Add end-of-sentence pad token to the model and tokenizer if it doesn't already exist.
        """
        if tokenizer.pad_token_id is None:
            # If there's no pad_token_id, default to eos_token_id or any value of your choice
            self.pad_token_id = tokenizer.eos_token_id
        else:
            self.pad_token_id = tokenizer.pad_token_id
        if tokenizer.pad_token is None:
            tokenizer.add_special_tokens({'pad_token': tokenizer.eos_token})

    def update_seed(self, seed):
        if seed is not None:
            set_seed(seed)

    def __call__(self, prompt, decoding_params, num_return_sequences=1, seed=None, enable_thinking=False):
        self.update_seed(seed)
        prompt = self.initialize_prompt_history(prompt)
        prompt = self.ensure_prompt_is_compatible_with_template(prompt)

        if "qwen3" in self.model_name.lower() and enable_thinking == False:
            tokens = self.tokenizer.apply_chat_template(
                prompt, return_dict=True, return_tensors="pt", add_generation_prompt=True,
                enable_thinking=False
            )
        else:
            tokens = self.tokenizer.apply_chat_template(
                prompt, return_dict=True, return_tensors="pt", add_generation_prompt=True
            )

        tokens = {k: v.to(self.model.device) for k, v in tokens.items()}
        prompt_length = tokens['input_ids'].shape[1]

        generation_config = {
            "num_return_sequences": num_return_sequences,
            "pad_token_id": self.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }
        generation_config.update(decoding_params)

        temperature = generation_config.get("temperature", None)
        if temperature is not None:
            temperature = self.get_temperature(temperature)
            if float(temperature) == 0.0:
                generation_config["do_sample"] = False
                generation_config["temperature"] = None
                generation_config["top_p"] = None
            else:
                generation_config["do_sample"] = True

        decoding_params = generation_config

        # run the following to check the exact prompt being used:
        # print(self.tokenizer.decode(tokens['input_ids'][0], skip_special_tokens=False))

        # Generate new tokens
        generated_ids = self.model.generate(
            **tokens,
            **generation_config,
        )

        # print(f"\n\n\n\n   Generated text:\n{self.tokenizer.decode(generated_ids[0])}\n\n\n\n")

        if num_return_sequences == 1:
            return self.tokenizer.decode(generated_ids[0][prompt_length:], skip_special_tokens=True)
        else:
            return [
                self.tokenizer.decode(
                    generated_id[prompt_length:], skip_special_tokens=True)
                for generated_id in generated_ids
            ]

    def ensure_prompt_is_compatible_with_template(self, prompt):
        """
        Ensure the input prompt is compatible with the chat template.
        Fixes issues with unsupported roles or improper role alternation.
        """
        def is_compatible(test_prompt):
            """
            Check if the given prompt is compatible with the chat template.
            Returns True if compatible, False otherwise.
            """
            try:
                _ = self.tokenizer.apply_chat_template(
                    test_prompt, return_dict=True, return_tensors="pt", add_generation_prompt=True
                )
                return True
            except Exception as e:
                return False

        # Step 1: Replace 'system' role with 'user' if needed
        if not is_compatible([{'role': 'system', 'content': 'Hello!'}]):
            prompt = self.replace_system_with_user(prompt)

        # Step 2: Concatenate consecutive roles if needed
        test_prompt2 = [{'role': 'user', 'content': 'Hello!'},
                        {'role': 'user', 'content': 'Hello again!'}]
        if not is_compatible(test_prompt2):
            prompt = self.concat_consecutive_roles(prompt)

        return prompt

    def replace_system_with_user(self, prompt):
        prompt = [{"role": "user" if p["role"] == "system" else p["role"],
                   "content": p["content"]} for p in prompt]
        return prompt

    def concat_consecutive_roles(self, prompt):
        """
        Concatenate consecutive user or assistant roles in the prompt.
        """
        concatenated_prompt = []
        current_role = None
        accumulated_content = ""

        for message in prompt:
            if message["role"] == current_role:
                # Accumulate content if the role is the same as the previous one
                accumulated_content += "\n" + message["content"]
            else:
                if current_role is not None:
                    # Append the previous accumulated message
                    concatenated_prompt.append(
                        {"role": current_role, "content": accumulated_content.strip()})
                # Start accumulating content for the new role
                current_role = message["role"]
                accumulated_content = message["content"]

        # Append the last accumulated message
        if current_role is not None:
            concatenated_prompt.append(
                {"role": current_role, "content": accumulated_content.strip()})

        return concatenated_prompt

    def get_temperature_range(self):
        return (0.0, 100.0)

    def _is_bf16_compatible(self) -> bool:
        """Checks if the current environment is bfloat16 compatible."""
        return torch.cuda.is_available() and torch.cuda.is_bf16_supported()


class _OpenAI(ChatBot):
    def __init__(self, model_name_or_path, config=None, api_key=None, seed=None):
        if seed is not None:
            self.seed = seed
        elif config is not None and config.SEED is not None:
            self.seed = config.SEED
        self.model_name = model_name_or_path
        self.api_key = api_key or self.get_api_key()
        self.client = self.initialize_client()
        self.tokenizer = None

    def get_api_key(self):
        return os.getenv('OPENAI_API_KEY')

    def initialize_client(self):
        try:
            return OpenAI(api_key=self.api_key)
        except Exception as e:
            print(f'Could not set up OpenAI API client: {e}')
            return None

    def is_reasoning_model(self):
        if any(m in self.model_name for m in [
            'o1',
            'o3',
            'o4'
        ]):
            return True
        else:
            return False

    def __call__(self, prompt, do_sample=False, temperature=None, top_p=None, max_new_tokens=10, num_return_sequences=1, seed=None, reasoning_params=None):
        self.update_seed(seed)
        prompt = self.initialize_prompt_history(prompt)
        prompt = [
            {
                "role": p["role"],
                "content": [{"type": "text", "text": p["content"]}],
            }
            for p in prompt]
        generation_config = {
            "model": self.model_name,
            "messages": prompt,
            "seed": seed,
            "max_tokens": max_new_tokens,
            "n": num_return_sequences,
        }
        if temperature is not None:
            generation_config["temperature"] = self.get_temperature(
                temperature)
        if do_sample:
            if top_p is not None:
                generation_config["top_p"] = top_p

        if self.is_reasoning_model():
            # Remove keys from generation_config that are not supported by reasoning models
            generation_config.pop("temperature", None)
            generation_config.pop("top_p", None)
            generation_config.pop("max_tokens", None)
            generation_config["n"] = 1
            if reasoning_params is not None:
                generation_config.update(reasoning_params)

        response = self.client.chat.completions.create(
            **generation_config
        )
        if num_return_sequences == 1:
            full_response_object = response.to_dict()
            response_text = response.choices[0].message.content
            return full_response_object, response_text
        else:
            pass

    # batch processing functions

    def prepare_batch_request(self, custom_id, prompt, do_sample=False, temperature=None, top_p=None, max_new_tokens=10, num_return_sequences=1, seed=None, reasoning_params=None):
        """Prepare a batch request without making the API call."""
        self.update_seed(seed)
        prompt = self.initialize_prompt_history(prompt)
        prompt = [
            {
                "role": p["role"],
                "content": [{"type": "text", "text": p["content"]}],
            }
            for p in prompt]

        body = {
            "model": self.model_name,
            "messages": prompt,
            "seed": seed,
            "max_tokens": max_new_tokens,
            "n": num_return_sequences,
        }

        if temperature is not None:
            body["temperature"] = self.get_temperature(temperature)
        if do_sample and top_p is not None:
            body["top_p"] = top_p

        if self.is_reasoning_model():
            body.pop("temperature", None)
            body.pop("top_p", None)
            body.pop("max_tokens", None)
            body["n"] = 1
            if reasoning_params is not None:
                body.update(reasoning_params)

        return {
            "custom_id": custom_id,
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": body
        }

    def create_batch_file(self, requests_list, filepath):
        """Create JSONL file for batch processing."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w') as f:
            for request in requests_list:
                f.write(json.dumps(request) + '\n')
        return filepath

    def submit_batch(self, filepath, endpoint="/v1/chat/completions"):
        """Upload file and create batch."""
        while True:
            file_created = False
            try:
                # Upload file
                with open(filepath, 'rb') as f:
                    file_response = self.client.files.create(
                        file=f,
                        purpose="batch"
                    )
                file_created = True

                # Create batch
                batch_response = self.client.batches.create(
                    input_file_id=file_response.id,
                    endpoint=endpoint,
                    completion_window="24h"
                )

                return batch_response
            except Exception as e:
                print(f"Error submitting batch: {e}")
                if not file_created:
                    print(f'File could not be uploaded: {filepath}')
                print('Press c+enter to retry...')
                pdb.set_trace()

    def check_batch_status(self, batch_id):
        """Check status of a batch."""
        try:
            return self.client.batches.retrieve(batch_id)
        except Exception as e:
            print(f"Error checking batch status: {e}")
            pdb.set_trace()
            return None

    def retrieve_batch_results(self, batch_id, output_path):
        """Download batch results to file."""
        try:
            batch = self.client.batches.retrieve(batch_id)
            if batch.status != "completed":
                print(f"Batch not completed. Status: {batch.status}")
                return None

            if batch.output_file_id is None:
                print("No output file available")
                return None

            # Download results
            file_response = self.client.files.content(batch.output_file_id)
            file_contents = file_response.text

            # Save to file
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, 'w') as f:
                f.write(file_contents)

            return output_path
        except Exception as e:
            print(f"Error retrieving batch results: {e}")
            pdb.set_trace()
            return None

    def process_batch_response(self, batch_response_line):
        """Normalize batch response to match sync response format."""
        try:
            response_data = json.loads(batch_response_line)
            if response_data.get("error"):
                print(f"Batch request error: {response_data['error']}")
                pdb.set_trace()
                return None

            # Extract the response body which matches the sync API format
            return response_data["response"]["body"]
        except Exception as e:
            print(f"Error processing batch response: {e}")
            pdb.set_trace()
            return None

    def wait_for_batch_completion(self, batch_id, timeout=86400, poll_interval=60):
        """Wait for batch to complete with polling."""
        start_time = time.time()
        while time.time() - start_time < timeout:
            batch = self.check_batch_status(batch_id)
            if batch is None:
                return None

            print(f"Batch {batch_id} status: {batch.status}")

            if batch.status == "completed":
                return batch
            elif batch.status in ["failed", "expired", "cancelled"]:
                print(f"Batch failed with status: {batch.status}")
                pdb.set_trace()
                return batch

            time.sleep(poll_interval)

        print(f"Batch timeout after {timeout} seconds")
        return None

    def get_temperature_range(self):
        return (0.0, 2.0)


set_seed(42)
