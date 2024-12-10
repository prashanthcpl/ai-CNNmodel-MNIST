from http.server import HTTPServer, SimpleHTTPRequestHandler
import threading
import webbrowser

class CustomHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=".", **kwargs)

def run_server():
    server_address = ('', 8000)
    httpd = HTTPServer(server_address, CustomHandler)
    print("Server running at http://localhost:8000")
    httpd.serve_forever()

if __name__ == '__main__':
    # Start server in a separate thread
    server_thread = threading.Thread(target=run_server)
    server_thread.daemon = True
    server_thread.start()
    
    # Open browser
    webbrowser.open('http://localhost:8000')
    
    # Keep main thread running
    try:
        while True:
            input()
    except KeyboardInterrupt:
        print("\nShutting down server...") 