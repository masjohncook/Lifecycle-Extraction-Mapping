from llama_index.llms import Ollama
# llm = Ollama(model="vicuna:13b")  # Run locally
# llm = Ollama(model="llama2:13b")  # Run locally
llm = Ollama(model="mistral")  # Run locally
# llm = Ollama(model="gemma")  # Run locally

import time
import pandas as pd

# Load the Excel file
train_file_path = 'dataset_ref/test_data_bef_aug.csv'
# train_file_path = 'dataset_ref/test.csv'

tech_map_df = pd.read_csv(train_file_path)

combined_df = pd.DataFrame(columns=['tech_num', 'tech_name', 'sentence'])


# Updated placeholder function to return a list of 5 sentences
def generate_answer(sentence, tech_num, tech_name):
    global combined_df

    print ("Tech name: ", tech_name)
    prompt = """
    Rephase :
    """
    desc =  sentence
    new = prompt + desc
    sentences = []


    response = llm.complete(new)
    hold = str(response)
    print ("Augemented data: ", hold)
    sentences.append(hold)
    time.sleep(1)


    # Example new data to append
    new_rows = {
        'tech_num': tech_num,  # Make sure this is a list
        'tech_name': tech_name,  # Make sure this is a list
        'sentence': sentences  # Assume 'sentences' is a string or a list of sentences to be joined
    }


    # Create a DataFrame from the new rows
    temp_df = pd.DataFrame(new_rows)

    # Append the new DataFrame to the existing one
    combined_df = pd.concat([combined_df, temp_df], ignore_index=True)




tech_map_df.apply(lambda row: generate_answer(row['sentence'], row["label"], row["text_label"]), axis=1)



# Define the path for the updated Excel file
updated_output_file_path = 'testing_data_augmented_mistral7b.csv'

# Save the DataFrame to the updated Excel file
combined_df.to_csv(updated_output_file_path, index=False)


