
class XSumPreprocessor:
    def __init__(self, tokenizer, max_input_length, max_target_length, prefix='summarize'):
        self.tokenizer = tokenizer
        self.max_input_length = max_input_length
        self.max_target_length = max_target_length
        self.prefix = prefix
  
    def preprocess(self, examples):
        # encode the code-docstring pairs
        texts = examples['document']
        summaries = examples['summary']
        
        inputs = [self.prefix + text for text in texts]
        model_inputs = self.tokenizer(inputs, max_length=self.max_input_length, padding="max_length", truncation=True)

        # encode the summaries
        labels = self.tokenizer(summaries, max_length=self.max_target_length, padding="max_length", truncation=True).input_ids

        # important: we need to replace the index of the padding tokens by -100
        # such that they are not taken into account by the CrossEntropyLoss
        labels_with_ignore_index = []
        for labels_example in labels:
            labels_example = [label if label != 0 else -100 for label in labels_example]
            labels_with_ignore_index.append(labels_example)
        
        model_inputs["labels"] = labels_with_ignore_index

        return model_inputs

  