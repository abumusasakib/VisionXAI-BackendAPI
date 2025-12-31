print('starting test client...')
from fastapi.testclient import TestClient
import importlib.util
# Import main module from file to avoid ModuleNotFound issues
import os
main_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'main.py'))
print('main_path:', main_path)
import sys
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)
    print('inserted repo_root to sys.path')
spec = importlib.util.spec_from_file_location('main', main_path)
main = importlib.util.module_from_spec(spec)
spec.loader.exec_module(main)
print('imported main')
client = TestClient(main.app)
print('created TestClient')
headers = {'Origin': 'http://localhost:3000', 'Access-Control-Request-Method': 'POST', 'Access-Control-Request-Headers': 'content-type'}
r = client.options('/caption', headers=headers)
print('status_code:', r.status_code)
print('access-control-allow-origin:', r.headers.get('access-control-allow-origin'))
print('access-control-allow-methods:', r.headers.get('access-control-allow-methods'))
print('access-control-allow-headers:', r.headers.get('access-control-allow-headers'))
print('body:', r.text)
