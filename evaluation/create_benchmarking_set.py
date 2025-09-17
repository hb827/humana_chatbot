from transformers import pipeline
import json
import uuid
import re

def generate_questions(prompt_template, model_pipeline):
    messages = [
        {"role": "user", "content": prompt_template},
    ]
    return model_pipeline(messages, max_new_tokens=2048)

def numbered_list_to_python_list(text_block):
    lines = text_block.strip().split('\n')
    result_list = []
    for line in lines:
        processed_line = re.sub(r'^\d+\.\s*', '', line).strip()
        if processed_line:  # Only add non-empty lines to the list
            result_list.append(processed_line)
    return result_list

def main():
    with open("../knowledge_base/parsed_documents/slamon_etal.txt", "r") as file:
        content = file.read()

    prompt_template_general = "Generate 10 questions that a non-expert might ask an AI chatbot about breast cancer. Your response should contain only the questions, separated by new line characters, and no other text. Number the questions 1 to 10. Each question should be a complete sentence."
    prompt_template_expert = "Generate 10 questions that an expert, such as a medical doctor or someone with an advanced degree in biology or genetics, might ask an AI chatbot about breast cancer. Your response should contain only the questions, separated by new line characters, and no other text. Number the questions 1 to 10. Each question should be a complete sentence."
    prompt_template_patient = "Generate 10 questions that a breast cancer patient might ask an AI chatbot about breast cancer. Your response should contain only the questions, separated by new line characters, and no other text. Number the questions 1 to 10. Each question should be a complete sentence."

    prompt_types = {"testing":"Generate a question about breast cancer.","general":prompt_template_general,"expert":prompt_template_expert,"patient":prompt_template_patient}

    pipe = pipeline("text-generation", model="Qwen/Qwen3-0.6B")
    
    all_questions = []
    for user,prompt in prompt_types.items():
        print(f'Generating potential questions asked by a(n) {user} user...')
        try:
            llm_output=generate_questions(prompt,pipe)
            print(llm_output)

            generated_text = llm_output[0]["generated_text"]

            response = [unit['content'] for unit in generated_text if unit['role']=='assistant']
            thinking = response[0].split('</think>')[0].split('<think>')[1].strip()
            response = response[0].split('</think>')[1].strip()

            question_list = [{"id":str(uuid.uuid4()),"user_type":user,"thinking":thinking,"question":question} for question in numbered_list_to_python_list(response)]
            all_questions.extend(question_list)
        except Exception as e:
            print(f'Error: {e}')

    with open("questions.json", "w") as json_file:
        json.dump(all_questions, json_file, indent=4)

if __name__ == "__main__":
    main()
