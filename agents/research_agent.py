#====Changes to be made in this file====
# 1. Find an alternative to the ibm_watsonx ModelInference. 


#from ibm_watsonx_ai.foundation_models import ModelInference
#from ibm_watsonx_ai import Credentials, APIClient
from typing import Dict, List
from langchain.schema import Document
from config.settings import settings
import json
import ollama

#credentials = Credentials(
#                   url = "https://us-south.ml.cloud.ibm.com",
#                  )
#client = APIClient(credentials)

# ResearchAgent generates initial draft response based on retrieved documents.
class ResearchAgent:
    def __init__(self):
        """
        Initialize the research agent with the IBM WatsonX ModelInference.
        """
        # Initialize the WatsonX ModelInference
        # print("Initializing ResearchAgent with IBM WatsonX ModelInference...")
        # self.model = ModelInference(
        #     model_id="meta-llama/llama-3-2-90b-vision-instruct", 
        #     credentials=credentials,
        #     project_id="skills-network",
        #     params={
        #         "max_tokens": 300,            # Adjust based on desired response length
        #         "temperature": 0.3,           # Controls randomness; lower values make output more deterministic
        #     }
        # )

        #Set Ollama model to be used to generate response
        self.model = "llama3.2-vision:11b"

        print("ModelInference initialized successfully.")

    #Purpose: Clean LLM response
    def sanitize_response(self, response_text: str) -> str:
        """
        Sanitize the LLM's response by stripping unnecessary whitespace.
        """
        return response_text.strip()
    

    #Purpose: Generate a structured prompt based on prompt and context (retrieved documents)
    def generate_prompt(self, question: str, context: str) -> str:
        """
        Generate a structured prompt for the LLM to generate a precise and factual answer.
        """
        prompt = f"""
        You are an AI assistant designed to provide precise and factual answers based on the given context.

        **Instructions:**
        - Answer the following question using only the provided context.
        - Be clear, concise, and factual.
        - Return as much information as you can get from the context.
        
        **Question:** {question}
        **Context:**
        {context}

        **Provide your answer below:**
        """
        return prompt

    #Purpose: Generate a response by querying the LLM with question and relevant documents
    def generate(self, question: str, documents: List[Document]) -> Dict:
        """
        Generate an initial answer using the provided documents.
        """
        print(f"ResearchAgent.generate called with question='{question}' and {len(documents)} documents.")

        # Combine the top document contents into one string
        context = "\n\n".join([doc.page_content for doc in documents])
        print(f"Combined context length: {len(context)} characters.")

        # Create a prompt for the LLM
        prompt = self.generate_prompt(question, context)
        print("Prompt created for the LLM.")

        # Call the LLM to generate the answer
        try:
            print("Sending prompt to the model...")
            # response = self.model.chat(
            #     messages=[
            #         {
            #             "role": "user",
            #             "content": prompt  # Ensure content is a string
            #         }
            #     ]
            # )

            response = ollama.chat(
                model = self.model,
                messages = [
                    {
                    "role": "user",
                    "content": prompt
                    }
                ]
            )

            print("LLM response received.")
        except Exception as e:
            print(f"Error during model inference: {e}")
            raise RuntimeError("Failed to generate answer due to a model error.") from e

        # Extract and process the LLM's response
        try:
            llm_response = response['message']['content'].strip()
            print(f"Raw LLM response:\n{llm_response}")
        except (IndexError, KeyError) as e:
            print(f"Unexpected response structure: {e}")
            llm_response = "I cannot answer this question based on the provided documents."

        # Sanitize the response
        draft_answer = self.sanitize_response(llm_response) if llm_response else "I cannot answer this question based on the provided documents."

        print(f"Generated answer: {draft_answer}")

        return {
            "draft_answer": draft_answer,
            "context_used": context
        }
    
# === Stand-alone execution entrypoint ===
if __name__ == "__main__":
    # Create an agent
    agent = ResearchAgent()

    # Sample question
    question = "What is the capital of Bui Division?"

    # Sample "documents" -- minimal test
    docs = [
        Document(page_content="Bui Division is found in the NorthWest of Cameroon. And Its capital is Nkuv.")
    ]

    # Generate a response
    result = agent.generate(question, docs)

    print("\n=== Final Output ===")
    print("Answer:", result["draft_answer"])
    print("Context Used:", result["context_used"])
