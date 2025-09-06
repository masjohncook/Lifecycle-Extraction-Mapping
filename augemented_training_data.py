from llama_index.llms import Ollama
# llm = Ollama(model="vicuna:13b")  # Run locally
# llm = Ollama(model="llama2:13b")  # Run locally
# llm = Ollama(model="mistral")  # Run locally
llm = Ollama(model="gemma")  # Run locally

import time
import pandas as pd

# Load the Excel file
train_file_path = 'dataset_ref/train_data_bef_aug.csv'
#train_file_path = 'dataset_ref/test.csv'
ref_file_path = "dataset_ref/tech_count.csv"
tech_map_df = pd.read_csv(train_file_path)
df = pd.read_csv(ref_file_path)
ref_count = df.set_index('label').T.to_dict('list')
combined_df = pd.DataFrame(columns=['tech_num', 'tech_name', 'sentence'])


# Updated placeholder function to return a list of 5 sentences
def generate_answer(sentence, tech_num, tech_name):
    global combined_df, ref_count

    print ("Tech name: ", tech_name)
    print("Count: ", ref_count[tech_num][0])
    prompt = """
    Rephase :
    """
    desc =  sentence
    new = prompt + desc
    sentences = []

    for i in range (ref_count[tech_num][0]):
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

# # Select the required columns for the new Excel file
# output_df = tech_map_df[['tech_num', 'tech_name', 'sentence']]

# Define the path for the updated Excel file
updated_output_file_path = 'training_data_augmented_gemma.csv'

# Save the DataFrame to the updated Excel file
combined_df.to_csv(updated_output_file_path, index=False)




# # Split the content by lines
# lines = response.split('\n')

# Extract sentences
# sentences = []
# for line in lines:
#     # Check if line starts with a number (indicating a new sentence)
#     if line.strip() and line[0].isdigit():
#         # Extract the sentence without the leading number
#         sentence = ' '.join(line.split()[1:])
#         sentences.append(sentence)
#
# print(sentences)