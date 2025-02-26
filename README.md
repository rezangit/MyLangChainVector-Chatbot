# Project Setup

1.  **Clone the repository:**
    ```bash
    git clone <your_repository_url>
    ```

2.  **Create a `.env` file:**
    *   Copy the contents of `.env.template` into a new file named `.env` in the same directory.
    *   Replace the placeholder values in `.env` with your actual API keys and region.
        *  **`OPENAI_API_KEY`**: your openai api key.
        * **`PINECONE_API_KEY`**: your pinecone api key.
        * **`PINECONE_ENV`**: your pinecone enviorment.

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Activate your virtual environment:**
    ```bash
    # Replace 'env-hf' with the name of your virtual environment if you used a different name
    env-hf\Scripts\activate
    ```

5.  **Run the application:**
    ```bash
    python main.py
    ```

