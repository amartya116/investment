import hashlib
import json
import time
import sys
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Set, Optional
import requests
from uuid import uuid4
from threading import Lock
from collections import Counter

# Blockchain logic
class Blockchain:
    def __init__(self):
        self.chain: List[dict] = []
        self.current_transactions: List[dict] = []
        self.lock = Lock()
        # Create genesis block
        self.new_block(proof=100, previous_hash='1')

    def new_block(self, proof: int, previous_hash: Optional[str] = None) -> dict:
        with self.lock:
            block = {
                'index': len(self.chain) + 1,
                'timestamp': time.time(),
                'transactions': list(self.current_transactions),
                'proof': proof,
                'previous_hash': previous_hash or self.hash(self.chain[-1]),
            }
            self.current_transactions.clear()
            self.chain.append(block)
            return block

    def new_transaction(self, account_number: str, stock_symbol: str, transaction_type: str,
                        share_quantity: int, price_per_share: float, commission_fee: float,
                        total_cost: float, net_amount: float, trade_date: str, settlement_date: str) -> int:
        with self.lock:
            self.current_transactions.append({
                'account': account_number,
                'stock_symbol': stock_symbol,
                'transaction_type': transaction_type,
                'quantity': share_quantity,
                'price_per_share': price_per_share,
                'commission': commission_fee,
                'total_cost': total_cost,
                'net_amount': net_amount,
                'trade_date': trade_date,
                'settlement_date': settlement_date
            })
            return self.last_block['index'] + 1

    @staticmethod
    def hash(block: dict) -> str:
        block_string = json.dumps(block, sort_keys=True)
        return hashlib.sha256(block_string.encode()).hexdigest()

    @property
    def last_block(self) -> dict:
        return self.chain[-1]

    def valid_proof(self, last_proof: int, proof: int) -> bool:
        guess = f"{last_proof}{proof}".encode()
        guess_hash = hashlib.sha256(guess).hexdigest()
        return guess_hash[:4] == "0000"

    def proof_of_work(self, last_proof: int) -> int:
        proof = 0
        while not self.valid_proof(last_proof, proof):
            proof += 1
        return proof

    def valid_chain(self, chain: List[dict]) -> bool:
        for i in range(1, len(chain)):
            prev = chain[i-1]
            curr = chain[i]
            if curr['previous_hash'] != self.hash(prev):
                return False
            if not self.valid_proof(prev['proof'], curr['proof']):
                return False
        return True

# FastAPI setup
app = FastAPI()
blockchain = Blockchain()
PEERS: Set[str] = set()
NODE_ID = str(uuid4()).replace('-', '')

# Pydantic model for transactions
class Transaction(BaseModel):
    account_number: str
    stock_symbol: str
    transaction_type: str
    share_quantity: int
    price_per_share: float
    commission_fee: float
    total_cost: float
    net_amount: float
    trade_date: str
    settlement_date: str

@app.post("/transactions/new")
def add_transaction(tx: Transaction):
    idx = blockchain.new_transaction(
        tx.account_number, tx.stock_symbol, tx.transaction_type,
        tx.share_quantity, tx.price_per_share, tx.commission_fee,
        tx.total_cost, tx.net_amount, tx.trade_date, tx.settlement_date
    )
    return {"message": f"Transaction will be added to block {idx}"}

@app.get("/mine")
def mine():
    proof = blockchain.proof_of_work(blockchain.last_block['proof'])
    block = blockchain.new_block(proof)
    return block

@app.get("/chain")
def get_chain():
    return {"chain": blockchain.chain, "length": len(blockchain.chain)}

@app.get("/nodes/register")
def get_peers():
    return {"nodes": list(PEERS)}

@app.get("/nodes/synchronize")
def synchronize():
    def clean_chain(chain):
        """Remove timestamp and hashes for comparison purposes"""
        return json.dumps([
            {k: v for k, v in block.items() if k not in ["timestamp", "previous_hash"]} for block in chain
        ], sort_keys=True)

    # Start with our local chain
    local_chain_str = clean_chain(blockchain.chain)
    chain_votes = {local_chain_str: {"count": 1, "chain": blockchain.chain}}
    
    # Total nodes in network = this node + all peers
    total_nodes = len(PEERS)
    
    # Get chains from all peers
    for peer in PEERS:
        try:
            response = requests.get(f"{peer}/chain", timeout=5)
            if response.status_code == 200:
                peer_chain = response.json()["chain"]
                peer_chain_str = clean_chain(peer_chain)
                
                if peer_chain_str in chain_votes:
                    chain_votes[peer_chain_str]["count"] += 1
                else:
                    chain_votes[peer_chain_str] = {"count": 1, "chain": peer_chain}
        except Exception as e:
            print(f"Could not contact peer {peer}: {str(e)}")
            continue
    
    # Find the chain with the most votes
    majority_chain_str = None
    max_votes = 0
    majority_chain = None
    
    for chain_str, data in chain_votes.items():
        if data["count"] > max_votes:
            max_votes = data["count"]
            majority_chain_str = chain_str
            majority_chain = data["chain"]
    
    # Check if we have a majority (more than half of total nodes)
    required_majority = (total_nodes // 2) + 1
    
    if max_votes >= required_majority:
        # We have a majority, check if our chain needs updating
        if majority_chain_str != local_chain_str:
            # Validate the majority chain before updating
            if blockchain.valid_chain(majority_chain):
                with blockchain.lock:
                    blockchain.chain = majority_chain
                    # Clear pending transactions since we're adopting a new chain
                    blockchain.current_transactions.clear()
                
                return {
                    "updated": True, 
                    "message": f"Chain updated by majority consensus ({max_votes}/{total_nodes} nodes agree)",
                    "total_nodes": total_nodes,
                    "majority_votes": max_votes
                }
            else:
                return {
                    "updated": False, 
                    "message": "Majority chain is invalid, keeping local chain",
                    "total_nodes": total_nodes,
                    "majority_votes": max_votes
                }
        else:
            return {
                "updated": False, 
                "message": f"Chain already matches majority ({max_votes}/{total_nodes} nodes agree)",
                "total_nodes": total_nodes,
                "majority_votes": max_votes
            }
    else:
        return {
            "updated": False, 
            "message": f"No majority consensus reached ({max_votes}/{total_nodes} nodes agree, need {required_majority})",
            "total_nodes": total_nodes,
            "majority_votes": max_votes
        }


if __name__ == '__main__':
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
    peers_arg = sys.argv[2] if len(sys.argv) > 2 else ''
    if peers_arg:
        PEERS.update(peers_arg.split(','))
    print(f"Node starting on port {port}, peers={PEERS}")
    uvicorn.run(app, host='127.0.0.1', port=port)