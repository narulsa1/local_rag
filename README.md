# local_rag
Chat with your Documents with off-line local RAG, build on your system

1. Python Version

Use: python 3.10 or 3.11

❌ Avoid: python 3.13 / 3.14  (causes numpy / build issues)

2. Install Properly

pip install --upgrade pip
pip install -r requirements.txt

3. Torch CPU Install
   
pip install torch --index-url https://download.pytorch.org/whl/cpu

After this Run:

python ingest.py
streamlit run app.py
