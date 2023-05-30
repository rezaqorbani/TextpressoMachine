# TextpressoMachine
All the relevant code can be found under the notebooks directory. In the following we will describe the use of each notebooks.

Please use the requirements.txt file to install the required packages.

1. **gpt2-summarizer**: This notebook trains and saves the gp2 model fine-tunder for summarization. The notebook can also load the saved checkpoints and can generate summaries (from the chosen test document). To get a simple run and one summarization generation, simply run the notebook (run all). 
2. **gpt2-Summarizer_Evaluation**: This notebook evaluates the saved gpt2 summarizer model with different evaluation metrics as described in the report. You shoul simply run the notebook and specify the checkpoint you want to run.
3. **t5-summerizer**: same as 1. but for t5 model.
4. **t5-summerizer-Evaluation**: same as 2. but for t5 model.
5. **t5-Baseline-Evaluation**: same as 2. but for t5 baseline (pre-trained not fine-tuned) model.
