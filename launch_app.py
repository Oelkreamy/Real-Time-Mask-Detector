"""
Launch script for Streamlit Mas        print("🌐 Starting web server...")
        print("🔗 URL: http://localhost:8502")
        print("\n📱 If the browser doesn't open automatically:")
        print("   1. Copy this URL: http://localhost:8502")
        print("   2. Paste it in your web browser")
        print("   3. Or try: http://127.0.0.1:8502")ction App
This script will start the Streamlit web application
"""

import subprocess
import sys
import os
import webbrowser
import time

def launch_streamlit():
    """Launch the Streamlit application"""
    
    print("🚀 Launching Streamlit Mask Detection App...")
    print("=" * 50)
    
    # Check if streamlit is installed
    try:
        import streamlit
        print(f"✅ Streamlit version: {streamlit.__version__}")
    except ImportError:
        print("❌ Streamlit not found. Please run setup_app.py first.")
        return
    
    # Check if model files exist
    model_weights = "model/weights/anti_overfit/best_anti_overfit.pt"
    if not os.path.exists(model_weights):
        print(f"❌ Model weights not found: {model_weights}")
        print("Please make sure you have trained the model first.")
        return
    
    print("✅ Model files found")
    
    # Launch Streamlit
    try:
        print("🌐 Starting web server...")
        print("� URL: http://localhost:8501")
        print("\n📱 If the browser doesn't open automatically:")
        print("   1. Copy this URL: http://localhost:8501")
        print("   2. Paste it in your web browser")
        print("   3. Or try: http://127.0.0.1:8501")
        print("\n⚠️  To stop the app, press Ctrl+C in this terminal")
        print("=" * 50)
        
        # Try to open browser (but don't fail if it doesn't work)
        def try_open_browser():
            time.sleep(4)  # Give server time to start
            try:
                # Try different methods to open browser
                import webbrowser
                
                # Method 1: Default browser
                if webbrowser.open('http://localhost:8502'):
                    print("✅ Browser opened successfully!")
                    return
                
                # Method 2: Try different browsers
                browsers = ['chrome', 'firefox', 'edge', 'safari']
                for browser in browsers:
                    try:
                        webbrowser.get(browser).open('http://localhost:8502')
                        print(f"✅ Opened in {browser}!")
                        return
                    except:
                        continue
                
                print("⚠️  Could not auto-open browser. Please open manually:")
                print("   http://localhost:8502")
                
            except Exception as e:
                print(f"⚠️  Browser auto-open failed: {e}")
                print("   Please open your browser manually and go to:")
                print("   http://localhost:8502")
        
        import threading
        threading.Thread(target=try_open_browser, daemon=True).start()
        
        # Run Streamlit
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", "streamlit_app.py",
            "--server.port", "8502",
            "--server.headless", "false"
        ])
        
    except KeyboardInterrupt:
        print("\n🛑 Stopping the application...")
    except Exception as e:
        print(f"❌ Error launching app: {e}")

if __name__ == "__main__":
    launch_streamlit()
