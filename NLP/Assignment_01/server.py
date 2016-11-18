# -*- coding: utf-8 -*-
from http.server import BaseHTTPRequestHandler, HTTPServer
from json import dumps
from model import *
from codecs import open

PORT_NUMBER = 8080


class imeHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/":
            self.send_response(200)
            self.send_header('Content-type','text/html')
            self.end_headers()
            with open('web/index.html', 'r', 'utf-8') as f:
                self.wfile.write(bytes(f.read(), "utf-8"))
        elif self.path == "/main.js":
            self.send_response(200)
            self.send_header('Content-type','text/javascript')
            self.end_headers()
            with open('web/main.js', 'r', 'utf-8') as f:
                self.wfile.write(bytes(f.read(), "utf-8"))
        elif self.path.startswith("/candidates/"):
            key = self.path[len("/candidates/"):]
            print("Input: %s" % key)
            split_pinyin = pinyin_seperator(key)
            candidates = cal_candidates(split_pinyin)
            response = {
                'part': split_pinyin,
                'candidates': candidates
            }
            self.send_response(200)
            self.send_header('Content-type', 'text/json')
            self.end_headers()
            output = dumps(response)
            self.wfile.write(bytes(output, "utf-8"))
        else:
            self.send_response(404)
            self.end_headers()

        return

try:
    server = HTTPServer(('', PORT_NUMBER), imeHandler)
    print('Started httpserver on port ', PORT_NUMBER)
    server.serve_forever()
except KeyboardInterrupt:
    print('^C received, shutting down the web server')
    server.socket.close()
