import openai
import os
from dotenv import load_dotenv
import numpy as np
from sklearn.preprocessing import normalize

# Load environment variables from .env file
load_dotenv()

# Get credentials and model info from environment variables
openai_model_type = os.getenv("OPENAI_AZURE_MODEL")  # Example: "gpt-4o-mini"
azure_openai_api_key = os.getenv("OPENAI_API_AZURE_KEY")
azure_openai_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
openai_model_version = os.getenv("OPENAI_API_VERSION")  # Example: "003"
openai_embeddings_model = os.getenv("OPENAI_API_EMBEDDINGS_MODEL")  # Example: "text-embedding-ada-002"

# Set OpenAI API credentials
openai.api_key = azure_openai_api_key
openai.api_base = azure_openai_endpoint

# Load persona from a text file
def load_persona():
    try:
        with open("persona.txt", "r") as file:
            persona = file.read()
        return persona
    except FileNotFoundError:
        return "Error: Persona file not found."

# Get Python assistance (response from OpenAI API)
def chat_completition(query):
    try:
        persona = load_persona()  # Load the persona from the file
        # Create the prompt by combining the persona and the query
        prompt = persona + "\n" + "Question: " + query + "\nAnswer:"

        # Get response from OpenAI chat model using ChatCompletion (gpt-4o-mini)
        completion  = openai.chat.completions.create(
            model=openai_model_type,  # Using the model type from .env (e.g., "gpt-4o-mini")
            messages=[{"role": "system", "content": prompt}, {"role": "user", "content": query}],
            max_tokens=150,          # Adjust the number of tokens for more/less text
            temperature=0.7,         # Creativity in response
        )        
        # Return the assistant's response as text
        return completion.choices[0].message.content

    except Exception as e:
        return f"Error: {e}"
    


# Convert text to vector using OpenAI's Embedding API
def text_to_vector(text):
    try:
        # Use the specified embeddings model from .env
        embedd = openai.embeddings.create(
            model=openai_embeddings_model,  # Using the embedding model from .env
            input=text
        )
        print(embedd.data[0].embedding)
        # Extract and return the embedding vector
        vector = embedd.data[0].embedding
        return np.array(vector)

    except Exception as e:
        print(f"Error: {e}")
        return None

# Personal assistant loop
def personal_assistant():
    print("Hello! I'm Sneek Assistant. Ask me anything related to Indonesian Law.")
    print("Type 'exit' to quit.")
    
    # Load persona at the beginning of the session
    persona = load_persona()
    if "Error" in persona:
        print(persona)  # Handle case where the persona file is missing
        return

    # Print the loaded persona (this is sent once at the start)
    print("Persona loaded.")
    
    while True:
        query = input("\nAsk about your cases: ")
        
        if query.lower() == 'exit':
            print("Goodbye!")
            break

        # Convert the question to a vector before sending it for the best response
        query_vector = text_to_vector(query)
        
        if query_vector is None:
            print("Error converting question to vector.")
            continue

        # Get Python assistance (text response)
        answer_text = chat_completition(query)
        print(f"\nAssistant Response (Text): {answer_text}")

        answer_text = chat_completition(query_vector)
        print(f"\nAssistant Response (Text): {answer_text}")
        # Optionally, normalize vectors for comparison
        normalized_query_vector = normalize(query_vector.reshape(1, -1))

        # Show a preview of the first 5 elements of the vector for both query and answer
        print("\nAssistant Response (Vector):")
        print(f"Query Vector: {normalized_query_vector[0][:5]}...")  # Showing first 5 elements of the query vector

if __name__ == "__main__":
    personal_assistant()

def chat_response(query):
    response = openai.responses.create(
            model=openai_model_type,  # Using the model type from .env (e.g., "gpt-4o-mini")
            input="what is the capital of italy",
            
            max_output_tokens=100,  # Limit the response length to 100 tokens

        )
    
    return response.output_text