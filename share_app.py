import os
import time
import threading
from pyngrok import ngrok

def run_gradio():
    os.system("python app.py")

def start_ngrok():
    print("Starting Gradio app...")
    # Give the app a moment to start
    time.sleep(5)
    
    try:
        # Open a HTTP tunnel on Gradio default port 7860
        public_url = ngrok.connect(7860).public_url
        print("\n" + "="*60)
        print(f"  \033[92m>>> YOUR LIVE LINK: {public_url} <<<\033[0m")
        print("  Share this link with your client. The app must stay running.")
        print("="*60 + "\n")
        
        # Keep the script running
        input("Press Enter to exit...\n")
    except Exception as e:
        print(f"\nError starting ngrok: {e}")
        print("\nNOTE: You may need a free ngrok account.")
        print("1. Go to https://dashboard.ngrok.com/signup")
        print("2. Get your Authtoken")
        print("3. Run: ngrok config add-authtoken <your_token>")

if __name__ == "__main__":
    # Start Gradio app in a separate thread
    thread = threading.Thread(target=run_gradio)
    thread.daemon = True
    thread.start()

    start_ngrok()
