from typing import Dict, Any

from transformers import ConversationalPipeline, Conversation


class ResponseGeneratorPipeline(ConversationalPipeline):
    def preprocess(self, conversation: Conversation, min_length_for_response=32) -> Dict[str, Any]:
        input_ids = self.tokenizer.apply_chat_template(conversation,
                                                       tokenize=True,
                                                       padding=True,
                                                       max_length=1024,
                                                       add_generation_prompt=True,
                                                       return_tensors="pt")

        return {"input_ids": input_ids, "conversation": conversation}
