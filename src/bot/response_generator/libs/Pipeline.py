from typing import Any, Dict

from transformers import Conversation, ConversationalPipeline


class ResponseGeneratorPipeline(ConversationalPipeline):
	def preprocess(self, conversation: Conversation, min_length_for_response=32) -> Dict[str, Any]:
		input_ids = self.tokenizer.apply_chat_template(
			conversation,
			tokenize=True,
			padding=True,
			add_generation_prompt=True,
			return_tensors="pt"
		)

		return {"input_ids": input_ids, "conversation": conversation}

	def postprocess(self, model_outputs, clean_up_tokenization_spaces=True):
		output_ids = model_outputs["output_ids"]
		answer = self.tokenizer.decode(
			output_ids[0],
			skip_special_tokens=True,
			clean_up_tokenization_spaces=clean_up_tokenization_spaces
		)
		conversation = model_outputs["conversation"]
		conversation[-1]["content"]["dialog"] = answer.replace(self.tokenizer.eos_token, "").strip()
		return conversation
