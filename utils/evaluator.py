from collections import Counter
from datasets import load_metric


class Evaluator:
    def __init__(self, hypothesis, reference):
        self.hypothesis = hypothesis
        self.reference = reference
        self.metrics={
        'rouge1': self.rouge_N(1),
        'rouge2': self.rouge_N(2),
        'rougeL': self.rouge_L(),
        'rougeLsum': self.rouge_L_sum(),
        'bert': self.bert_Score()
    }
        
    def rouge_L(self):
        metric = load_metric("rouge")
        metric_type = 'rougeL'
        rg_score = metric.compute(predictions=self.hypothesis, references=self.reference, rouge_types=[metric_type])['rougeL'].mid
        return {"precision": rg_score.precision, "recall": rg_score.recall, "f1": rg_score.fmeasure}
        
    
    def rouge_N(self, n=1):
        metric = load_metric("rouge")
        metric_type = f'rouge{n}'
        rg_score = metric.compute(predictions=self.hypothesis, references=self.reference, rouge_types=[metric_type])[metric_type].mid
        return {"precision": rg_score.precision, "recall": rg_score.recall, "f1": rg_score.fmeasure}
		
		
    def rouge_L_sum(self):
        metric = load_metric("rouge")
        metric_type = 'rougeLsum'
        rg_score = metric.compute(predictions=self.hypothesis, references=self.reference, rouge_types=[metric_type])[metric_type].mid
        return {"precision": rg_score.precision, "recall": rg_score.recall, "f1": rg_score.fmeasure}
		
        
    def bert_Score(self):
        metric = load_metric("bertscore")
        bert_score = metric.compute(predictions=self.hypothesis, references=self.reference, lang="eng")
        return {"precision": bert_score['precision'][0], "recall": bert_score['recall'][0], "f1": bert_score['f1'][0]}
    
    
    def rouge_L_evaluation(self):
        # Tokenize hypothesos and reference sentences
        hypothesis_tokens = self.hypothesis.split()
        reference_tokens = self.reference.split()

        # Compute the length of the longest common subsequence
        lcs = lcs_length(hypothesis_tokens, reference_tokens)

        # Compute precision, recall, and f1 score
        precision = lcs / len(hypothesis_tokens)
        recall = lcs / len(reference_tokens)
        f1_score = 2 * ((precision * recall) / (precision + recall + 1e-7))

        return {"precision": precision, "recall": recall, "f1": f1_score}


    def rouge_N_evaluation(self, n=1):
        # split sentences into n-grams
        def ngrams(sentence, n):
            # use a list comprehension to generate n-grams
            return Counter([tuple(sentence[i:i+n]) for i in range(len(sentence) - n + 1)])

        # compute the n-grams for the candidate and reference sentences
        hypothesis_ngrams = ngrams(self.hypothesis.split(" "), n)
        reference_ngrams = ngrams(self.reference.split(" "), n)

        # count the number of shared n-grams
        shared_ngrams = hypothesis_ngrams & reference_ngrams
        shared_count = sum(shared_ngrams.values())

        # calculate precision, recall, and f1 score
        precision = shared_count / sum(hypothesis_ngrams.values())
        recall = shared_count / sum(reference_ngrams.values())
        f1_score = 2 * ((precision * recall) / (precision + recall + 1e-7))

        return {"precision": precision, "recall": recall, "f1": f1_score}
    
    def rouge_L_sum_evaluation(self):
        # Tokenize candidate and reference summaries
        hypothesis_tokens = self.hypothesis.split()
        reference_tokens = self.reference.split()

        # Compute the length of the longest common subsequence for summarizations
        lcs_sum = lcs_length(hypothesis_tokens, reference_tokens)

        # Compute precision, recall, and f1 score
        precision = lcs_sum / len(hypothesis_tokens)
        recall = lcs_sum / len(reference_tokens)
        f1_score = 2 * ((precision * recall) / (precision + recall + 1e-7))

        return {"precision": precision, "recall": recall, "f1": f1_score}
    
    @staticmethod
    def lcs_length(s1, s2):
            m, n = len(s1), len(s2)
            dp = [[0] * (n + 1) for _ in range(m + 1)]

            for i in range(1, m + 1):
                for j in range(1, n + 1):
                    if s1[i - 1] == s2[j - 1]:
                        dp[i][j] = dp[i - 1][j - 1] + 1
                    else:
                        dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

            return dp[m][n]
        
        
def evaluate_model(test_data, model, tokenizer, metric):
        precision, recall, f1 = [], [], []

        for example in test_data:
            input_text = example["input_ids"]
            reference_summary = example["summary"]
            generated_summary = model.generate(torch.tensor([input_text]).to(device))  # Generate summary using your model's generate function
            gen_summary=tokenizer.decode(generated_summary[0], skip_special_tokens=True)
            scores_dict = Evaluator([gen_summary], [reference_summary]).metrics[metric]  # Use evaluation function from previous examples
            precision.append(scores_dict['precision'])
            recall.append(scores_dict['recall'])
            f1.append(scores_dict['f1'])
            
        avg_precision = sum(precision) / len(precision)
        avg_recall = sum(recall) / len(recall)
        avg_f1 = sum(f1) / len(f1)

        return avg_precision, avg_recall, avg_f1
    