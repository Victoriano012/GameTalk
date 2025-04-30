from trl import SFTTrainer

class CustomSTaRTrainer(SFTTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        dataset.keep_partial_conversations = False

    def training_step(self, model, inputs, num_items_in_batch = None):
        inputs = filter(inputs)
        input_ids, attention_mask = tokenizer.process(inputs)

        self.step(input_ids=input_ids, attention_mask=attention_mask)