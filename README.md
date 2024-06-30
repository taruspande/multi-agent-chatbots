## Local Configuration

- Setup Virtual Environment & Install Dependencies
    ```bash
    python -m venv env
    source env/bin/activate
    pip install -r requirements.txt
    ```

- Replace the template code with your api key in line 8 of ```main.py```.
    ```python
    os.environ["OPENAI_API_KEY"] = "YOUR_API_KEY_GOES_HERE"
    ```

- Run the file
    ```bash
    chainlit run main.py
    ```