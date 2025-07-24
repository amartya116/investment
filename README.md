This project implements a modular, event-driven trading architecture that simulates the core functionality of a decentralized exchange (DEX). At the center is a robust Matching Engine responsible for processing incoming user orders, matching buy/sell pairs, and generating trades. The system is built using FastAPI for API communication and SQLAlchemy for database interactions, with all confirmed trades stored in a PostgreSQL database for persistence and auditability.
<img width="603" height="695" alt="image" src="https://github.com/user-attachments/assets/b603dd13-f9cf-4f64-9d45-19e5e334ea50" />

🧩 System Architecture
Once a user submits an order via the API, the Matching Engine handles the logic to determine if the order results in a trade. When a trade is successfully generated, it is broadcast to a network of distributed nodes (Node 1–4). Each node maintains a local copy of the ledger and is responsible for adding blocks containing the new trade data.

To ensure consistency across the network, the nodes implement a majority-voting consensus mechanism. When a new block is proposed, it must be accepted by more than 50% of the nodes to be considered valid. Once consensus is reached, the block is committed across all nodes, preserving the integrity and synchronization of the distributed ledger.

Simultaneously, the finalized trade is saved to a centralized PostgreSQL database, enabling downstream analytics, reporting, and audit logging. This separation of execution logic (Matching Engine) and distributed ledger maintenance (Nodes) creates a scalable and fault-tolerant foundation for decentralized trade processing.

🛠️ Tech Stack
FastAPI – RESTful APIs for user and node interaction

SQLAlchemy – ORM for PostgreSQL-backed trade and order management

PostgreSQL – Centralized, durable storage for all trades

Python – Core language for matching logic, node communication, and consensus

Custom Consensus Protocol – Majority voting among distributed nodes

✅ Key Features
Central Matching Engine with order book management

Blockchain-inspired nodes with majority voting consensus

Real-time trade generation and block broadcasting

Persistent trade storage using PostgreSQL

Modular architecture ready for future integration with gRPC, Raft, or smart contracts
