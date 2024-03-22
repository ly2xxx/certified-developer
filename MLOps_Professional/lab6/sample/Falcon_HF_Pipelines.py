# from transformers import AutoTokenizer
#, AutoModelForCausalLM
from transformers import AutoModel
import transformers
import torch
import argparse
import time


def main(FLAGS):
    #Constructs the model identifier based on the Falcon version specified in the command-line arguments.
    model = f"tiiuae/falcon-{FLAGS.falcon_version}"
    
    #Initializes a tokenizer for the specified model, allowing text inputs to be processed for the model.
    # tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
    tokenizer = AutoModel.from_pretrained(model, trust_remote_code=True)
    
    #Creates a text generation pipeline using the transformers library. This pipeline abstracts model loading, tokenization, and inference
#     Arguments provided to the pipeline creation:
# "text-generation": Specifies the pipeline type.
# model=model: Specifies the pre-trained model to be used.
# tokenizer=tokenizer: Specifies the tokenizer to be used.
# torch_dtype=torch.bfloat16: Sets the data type for torch tensors to bfloat16.
# trust_remote_code=True: Allows remote code execution for the tokenizer.
# device_map="auto": Automatically selects the device for inference.    
    generator = transformers.pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map="auto",
    )

#     User Interaction Loop:
# user_input = "start": Initializes the user input to begin the loop.
# while user_input != "stop":: Starts a loop that continues until the user enters "stop".
    user_input = "start"

    while user_input != "stop":
        #Prompts the user to input a text string for the model to generate text based on
        user_input = input(f"Provide Input to {model} parameter Falcon (not tuned): ")
        
        start = time.time()
        #Generates text sequences using the pre-trained model and provided input.
        # f""" {user_input}""": Formats the user input for model processing.
        # max_length=FLAGS.max_length: Specifies the maximum length of the generated text.
        # do_sample=False: Disables sampling during text generation.
        # top_k=FLAGS.top_k: Sets the number of highest probability tokens to consider.-
        # num_return_sequences=1: Specifies the number of generated sequences to return.
        # eos_token_id=tokenizer.eos_token_id: Specifies the end-of-sequence token ID.
        if user_input != "stop":
            sequences = generator( 
            f""" {user_input}""",
            max_length=FLAGS.max_length,
            do_sample=False,
            top_k=FLAGS.top_k,
            num_return_sequences=1,
            eos_token_id=tokenizer.eos_token_id,)

        inference_time = time.time() - start
        
        for seq in sequences:
         print(f"Result: {seq['generated_text']}")
         
        print(f'Total Inference Time: {inference_time} seconds')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument('-fv',
                        '--falcon_version',
                        type=str,
                        default="7b",
                        help="select 7b or 40b version of falcon")
    parser.add_argument('-ml',
                        '--max_length',
                        type=int,
                        default="25",
                        help="used to control the maximum length of the generated text in text generation tasks")
    parser.add_argument('-tk',
                        '--top_k',
                        type=int,
                        default="5",
                        help="specifies the number of highest probability tokens to consider at each step")
    
    FLAGS = parser.parse_args()
    main(FLAGS)