# humana_chatbot

## Running the chatbot

1. Clone the repository and create a conda environment based on the environment.yml file
2. streamlit run app.py

The app should launch in a browser tab. It may take a moment for the pdf to load, then you can ask questions. Note that the app uses gpt2 as a foundation model so that it can run on a cpu. High quality responses should not be expected; this app is intended only as a prototype.

## Viewing the evaluation

The file eval_report.html contains information about the chatbot's performance across five dimensions. For further information about how the evaluation was conducted, and for parallel evaluations of gpt2-based and llama-based chatbots using the same RAG system, view the contents of the evaluation directory.

## Reflection

### Overview of the chatbot

This chatbot was developed to answer questions about Slamon et al. (1987) using a RAG system. In order to prepare the pdf for ingestion into the RAG system, the following steps were taken: The pdf of Slamon et al. (1987) was parsed and segmented into chunks for retrieval. Embeddings for those chunks were created using the all-MiniLM-L6-v2 sentence transformer, and an index was created using FAISS. When the chatbot is prompted, the user prompt is then used to retrieve relevant chunks from the knowledge base, and those chunks are inserted into a prompt template, along with the user prompt. Because the entire system is designed to run on a cpu, only very small models can be used. GPT-2 was selected because it allowed response speeds of approximately 8 seconds, making it a functional prototype. Note, however, that the accuracy of GPT-2's responses is extremely low and the outputs are often incoherent. 

### Overview of the evaluation procedure

The chatbot was evaluated using a small synthetic dataset of 30 questions. The questions were generated using Qwen/Qwen3-0.6B. Again, in order to run on a CPU, small models had to be selected. A model from a separate family was selected in order to prevent a biased evaluation.

The model was prompted three times, each time to generate 10 questions that a potential user might ask about the target article (Slamon et al., 1987). Each of the prompts instructed the model to simulate a different hypothetical user: general readers, domain experts (such as those with an advanced degree in medicine, biology, or genetics), and breast cancer patients. The 30 questions generated this way were then used to prompt the chatbot (once per prompt), and responses were recorded and evaluated.

Chatbot responses were evaluated based on five dimensions: response time, retrieval relevance, response groundedness, response helpfulness, and response accuracy. Dimensions which were evaluated using an LLM as a judge used Qwen/Qwen3-0.6B.

- Response Time: This dimension was evaluated automatically by recording the timestamp immediately before and after generating the chatbot’s response to the query.
- Retrieval Relevance: This dimension was evaluated using an LLM as a judge. The LLM was prompted to provide a numerical relevance rating on a scale from 1 to 5 and was provided with both the query and the retrieved passages. Higher scores indicate that the retrieved passages were more relevant to the query, defined as containing the answer to the query.
- Response Groundedness: This dimension was evaluated using an LLM as a judge. The LLM was prompted to provide a numerical groundedness rating on a scale from 1 to 5 and was provided with both the chatbot response and the retrieved passages. Higher scores indicate that the chatbot response was more grounded in the retrieved passages; lower scores indicate the presence of hallucinations.
- Response Helpfulness: This dimension was evaluated using an LLM as a judge. The LLM was prompted to provide a numerical helpfulness rating on a scale from 1 to 5 and was provided with both the query and the chatbot response. Higher scores indicate that the response was more helpful to the hypothetical user who submitted the query.
- Response Accuracy: This dimension was evaluated using an LLM as a judge. The LLM was prompted to provide a numerical accuracy rating on a scale from 1 to 5 and was provided with the query, the retrieved passages, and the chatbot response. Higher scores indicate that the response was a more accurate response to the query, given the retrieved passages.

While the evalutation dimensions have some overlap, they capture distinct potential patterns of failure, which can be helpful for identifying points of failure in the system. For example, in cases where the answer to the user’s query is not contained in the retrieved passages, retrieval relevance will be low. In such a case, a chatbot response like “I’m sorry, I couldn’t find the answer to your query in the document” would receive low ratings for helpfulness but high ratings for groundedness and accuracy; a chatbot response that correctly summarises the information contained in the (irrelevant) retrieved passages would receive low ratings for accuracy and helpfulness but high ratings for groundedness; a chatbot response that directly answers the query using information not contained in the retrieved passages would receive low ratings for accuracy and groundedness but high ratings for helpfulness.

The evaluation procedure was run twice: once with GPT-2 as the foundation model supporting the chatbot and once with Llama-3.2-1B. The comparison of these two evaluations reveals that substantial gains can be made in response quality by using a bigger model, though this of course comes with timing costs.

### Discussion of ClinVec

1. What are the 3 takeaways for this research?
3. How could this research be important for Humana?
4. How can you include this research into your Q/A chatbot?
