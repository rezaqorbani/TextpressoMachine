
from datasets import load_dataset


class XSumDataLoader():
    """
    XSUM dataset loader 
    """
    def __init__(self, tokenizer, max_input_length, max_target_length, prefix='summarize'):
      
        self.MAX_INPUT_LENGTH = max_input_length
        self.MAX_TARGET_LENGTH = max_target_length
        self.prefix = prefix

        ## Load data
        self.datasets = load_dataset("xsum")
        
        ## Load pretrained Tokenizer
        self.tokenizer = tokenizer
        
        ## Tokenize data
        self.tokenized_datasets=self.datasets.map(self.tokenize, batched=True)
        
        
        
        
        
    def tokenize(self, examples):
        # Tokenize input texts
        inputs = [self.prefix + doc for doc in examples["document"]]
        model_inputs = self.tokenizer(inputs, max_length=self.MAX_INPUT_LENGTH, truncation=True)

        # tokenize summaries
        labels = self.tokenizer(examples["summary"], max_length=self.MAX_TARGET_LENGTH, truncation=True)
        model_inputs["labels"] = labels["input_ids"]

        return model_inputs
    
    
                
        

