import socket
import subprocess
import time
import requests

# Utility to find free port
def find_free_port():
    s = socket.socket()
    s.bind(('', 0)) 
    port = s.getsockname()[1]
    s.close()
    return port

distributdNode = []

def disttributed(a, b, c, d):
    distributdNode.extend([a, b, c, d])
    return distributdNode

if __name__ == '__main__':
    # Allocate four distinct ports
    p1, p2, p3, p4 = find_free_port(), find_free_port(), find_free_port(), find_free_port()
    # Ensure ports are unique
    while len({p1, p2, p3, p4}) < 4:
        p2 = find_free_port()
        p3 = find_free_port()
        p4 = find_free_port()

    node1_url = f'http://127.0.0.1:{p1}'
    node2_url = f'http://127.0.0.1:{p2}'
    node3_url = f'http://127.0.0.1:{p3}'
    node4_url = f'http://127.0.0.1:{p4}'

    all_nodes = [node1_url, node2_url, node3_url, node4_url]
    all_nodes_str = ','.join(all_nodes)

    # Start nodes as subprocesses
    cmd1 = ['python', 'market_sim/Blockchain/marketTransactionsledger.py', str(p1), all_nodes_str]
    cmd2 = ['python', 'market_sim/Blockchain/marketTransactionsledger.py', str(p2), all_nodes_str]
    cmd3 = ['python', 'market_sim/Blockchain/marketTransactionsledger.py', str(p3), all_nodes_str]
    cmd4 = ['python', 'market_sim/Blockchain/marketTransactionsledger.py', str(p4), all_nodes_str]

    print(f"Starting Node 1 on port {p1}")
    p1_proc = subprocess.Popen(cmd1)
    time.sleep(1)
    print(f"Starting Node 2 on port {p2}")
    p2_proc = subprocess.Popen(cmd2)
    time.sleep(1)
    print(f"Starting Node 3 on port {p3}")
    p3_proc = subprocess.Popen(cmd3)
    time.sleep(1)
    print(f"Starting Node 4 on port {p4}")
    p4_proc = subprocess.Popen(cmd4)
    time.sleep(1)

    # Auto-register peers: register *all other nodes* for each node
    try:
        for idx, node_url in enumerate(all_nodes):
            peers = [url for i, url in enumerate(all_nodes) if i != idx]
            reg = requests.post(f'{node_url}/nodes/register', json={'nodes': peers})
            print(f'Node{idx+1} register response:', reg.json())
    except Exception as e:
        print('Peer registration failed:', e)

    # Trigger consensus to sync chains
    try:
        for idx, node_url in enumerate(all_nodes):
            res = requests.get(f'{node_url}/nodes/resolve')
            print(f'Node{idx+1} resolve response:', res.json())
    except Exception as e:
        print('Consensus resolve failed:', e)

    print(f"Node1 -> {node1_url}")
    print(f"Node2 -> {node2_url}")
    print(f"Node3 -> {node3_url}")
    print(f"Node4 -> {node4_url}")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Stopping nodes...")
        p1_proc.terminate()
        p2_proc.terminate()
        p3_proc.terminate()
        p4_proc.terminate()
