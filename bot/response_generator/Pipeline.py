from transformers import Pipeline

class ResponseGeneratorPipeline(Pipeline):
    def _sanitize_parameters(self, **kwargs):
        preprocess_arguments: dict = {}
        forward_arguments: dict = {}
        postprocess_arguments: dict = {}

        try:
            forward_arguments["model"] = kwargs["model"]
        except ValueError:
            print("model is required as an argument")

        preprocess_arguments["tokenizer"] = kwargs["tokenizer"] if kwargs["tokenizer"] is not None else kwargs["model"]
        postprocess_arguments["tokenizer"] = preprocess_arguments["tokenizer"]

        forward_arguments["generation_config"] = kwargs["generation_config"]

        preprocess_arguments["device"] = kwargs["device"]

        return preprocess_arguments, forward_arguments, postprocess_arguments

    def preprocess(self, sample, tokenizer, device):
        tokenized_prompt = tokenizer.apply_chat_template(sample["prompt"], tokenize=False, add_generation_prompt=True,
                                                         return_tensors="pt").to(device)
        return {"model_input": tokenized_prompt}

    def _forward(self, model_inputs, model, generation_config):
        model_outputs = model.generate(**model_inputs, generation_config=generation_config)
        return model_outputs

    def postprocess(self, model_outputs, tokenizer):
        response_raw = tokenizer.decode(model_outputs[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)
        response = response_raw.replace(response_raw.split("[/INST]")[0], "").removeprefix("[/INST]").strip()
        return response
