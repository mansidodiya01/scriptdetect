import threading
import subprocess  # Ensure subprocess is imported

def run_babydetect():
    subprocess.run(["python3", "frame.py"])

def run_cryclassify():
    subprocess.run(["python3", "audioinput.py"])

if __name__ == "__main__":
    # Create and start threads
    thread1 = threading.Thread(target=run_babydetect)
    thread2 = threading.Thread(target=run_cryclassify)

    thread1.start()
    thread2.start()

    # Wait for threads to complete
    thread1.join()
    thread2.join()
