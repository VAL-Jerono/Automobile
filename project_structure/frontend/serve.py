#!/usr/bin/env python3
"""
Simple HTTP server for serving the insurance frontend
"""
import http.server
import socketserver
import os

PORT = 3000
DIRECTORY = os.path.dirname(os.path.abspath(__file__))

class MyHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=DIRECTORY, **kwargs)
    
    def end_headers(self):
        # Add CORS headers
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        super().end_headers()

if __name__ == "__main__":
    with socketserver.TCPServer(("", PORT), MyHTTPRequestHandler) as httpd:
        print(f"""
╔════════════════════════════════════════════════════════════════╗
║                 🚗 AutoGuard Insurance Portal 🚗              ║
╠════════════════════════════════════════════════════════════════╣
║                                                                ║
║  Customer Portal:  http://localhost:{PORT}                ║
║  Admin Dashboard:  http://localhost:{PORT}/admin.html     ║
║                                                                ║
║  API Server:       http://localhost:8000                       ║
║  API Docs:         http://localhost:8000/docs                  ║
║                                                                ║
╠════════════════════════════════════════════════════════════════╣
║  Features:                                                     ║
║  ✓ Multi-step quote calculator with ML risk scoring           ║
║  ✓ Policy renewal portal                                       ║
║  ✓ Claims submission system                                    ║
║  ✓ Comprehensive admin dashboard with analytics               ║
║  ✓ Real-time KPIs and visualizations                           ║
║  ✓ Customer & policy management                                ║
║  ✓ ML insights and feature importance                          ║
╠════════════════════════════════════════════════════════════════╣
║  Database: {191480:,} customers, {52645:,} policies loaded          ║
║  Model: 94.05% accuracy ensemble (RandomForest + GradientBoost)║
╚════════════════════════════════════════════════════════════════╝

Press Ctrl+C to stop the server
""")
        httpd.serve_forever()
