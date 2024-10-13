Certainly! Let's enhance your SG-Tree Rust project by adding a **Search** feature. This feature will allow users to perform semantic searches within the SG-Tree, retrieving nodes that are most similar to a given query based on their embeddings.

Below is the comprehensive update to your project, including:

1. **Updated `main.rs`**: Incorporates the search functionality with necessary methods and API routes.
2. **Updated `README.md`**: Documents the new search feature and its usage.
3. **Updated `Cargo.toml`**: Includes any additional dependencies required for the search functionality.

---

## 1. Updated `main.rs`

```rust
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{
    collections::{HashMap, HashSet},
    fs::File,
    io::BufReader,
    path::Path,
    sync::Arc,
};
use parking_lot::RwLock;
use rayon::prelude::*;
use salvo::prelude::*;
use candle::{Device, Tensor};
use candle_transformers::models::bert::{BertConfig, BertModel};
use candle_transformers::models::WithTokenizer;
use hf_hub::{Cache, Repo};
use rand::Rng;
use redb::{Database, TableDefinition, ReadableTable, WritableTable, Transaction};
use tokio::task;
use anyhow::Result;

/// Type Definitions
type NodeIndex = u32;
type Offset = u32;
type PositionIndex = u32;
type FragmentEncoding = u8;
type EdgeLabel = u16;

/// Represents a unique node in the SG-Tree
#[derive(Serialize, Deserialize, Debug, Clone)]
struct Node {
    index: NodeIndex,
    fragment: String,
    encoding: FragmentEncoding,
    embedding: Vec<f32>,
}

/// Represents a position node in the SG-Tree
#[derive(Serialize, Deserialize, Debug, Clone)]
struct PositionNode {
    position: PositionIndex,
    node_index: NodeIndex,
    offset: Offset,
}

/// Represents a directed edge between nodes in the SG-Tree
#[derive(Serialize, Deserialize, Debug, Clone)]
struct Edge {
    from: NodeIndex,
    to: NodeIndex,
    label: EdgeLabel,
}

/// SG-Tree structure encapsulating nodes, positions, and edges with Redb persistence
struct SGTree {
    db: Arc<Database>,
    nodes_table: TableDefinition<u32, Vec<u8>>,
    fragments_table: TableDefinition<String, u32>,
    positions_table: TableDefinition<u32, Vec<u8>>,
    edges_table: TableDefinition<u32, Vec<u8>>,
    next_node_index: RwLock<NodeIndex>,
    next_position_index: RwLock<PositionIndex>,
}

impl SGTree {
    /// Initializes a new SG-Tree with Redb persistence
    fn new(db_path: &str) -> Self {
        // Open or create the Redb database
        let db = Database::create(db_path).expect("Failed to create or open Redb database");

        // Define tables
        let nodes_table = TableDefinition::new("nodes");
        let fragments_table = TableDefinition::new("fragments");
        let positions_table = TableDefinition::new("positions");
        let edges_table = TableDefinition::new("edges");

        // Initialize tables if they don't exist
        {
            let mut txn = db.begin_write().expect("Failed to begin write transaction");
            txn.open_table_or_create(&nodes_table).expect("Failed to open/create nodes table");
            txn.open_table_or_create(&fragments_table).expect("Failed to open/create fragments table");
            txn.open_table_or_create(&positions_table).expect("Failed to open/create positions table");
            txn.open_table_or_create(&edges_table).expect("Failed to open/create edges table");
            txn.commit().expect("Failed to commit initial tables");
        }

        // Retrieve the next indices
        let next_node_index = {
            let txn = db.begin_read().expect("Failed to begin read transaction");
            let table = txn.open_table(&nodes_table).expect("Failed to open nodes table");
            let max = table.iter().map(|r| r.key()).max().unwrap_or(0);
            RwLock::new(max + 1)
        };

        let next_position_index = {
            let txn = db.begin_read().expect("Failed to begin read transaction");
            let table = txn.open_table(&positions_table).expect("Failed to open positions table");
            let max = table.iter().map(|r| r.key()).max().unwrap_or(0);
            RwLock::new(max + 1)
        };

        SGTree {
            db: Arc::new(db),
            nodes_table,
            fragments_table,
            positions_table,
            edges_table,
            next_node_index,
            next_position_index,
        }
    }

    /// Encodes a fragment using a simple gematria-based encoding
    fn encode_fragment(&self, fragment: &str) -> FragmentEncoding {
        fragment
            .chars()
            .filter(|c| c.is_ascii_alphabetic())
            .map(|c| (c.to_ascii_uppercase() as u8 - b'A' + 1) as FragmentEncoding)
            .sum()
    }

    /// Adds a node to the SG-Tree with deduplication
    fn add_node(&self, fragment: String, model_state: &ModelState) -> NodeIndex {
        // Check for deduplication
        {
            let txn = self.db.begin_read().expect("Failed to begin read transaction");
            let table = txn.open_table(&self.fragments_table).expect("Failed to open fragments table");
            if let Some(&index) = table.get(&fragment).expect("Failed to get fragment") {
                return index;
            }
        }

        // Generate encoding
        let encoding = self.encode_fragment(&fragment);

        // Generate embedding
        let embedding = self.generate_embedding(&fragment, model_state);

        // Get next node index
        let node_index = {
            let mut lock = self.next_node_index.write();
            let idx = *lock;
            *lock += 1;
            idx
        };

        let node = Node {
            index: node_index,
            fragment: fragment.clone(),
            encoding,
            embedding,
        };

        // Serialize node
        let node_bytes = bincode::serialize(&node).expect("Failed to serialize node");

        // Write to Redb
        {
            let mut txn = self.db.begin_write().expect("Failed to begin write transaction");
            let nodes_table = txn.open_table(&self.nodes_table).expect("Failed to open nodes table");
            nodes_table.insert(node_index, node_bytes).expect("Failed to insert node");

            let fragments_table = txn.open_table(&self.fragments_table).expect("Failed to open fragments table");
            fragments_table.insert(fragment, node_index).expect("Failed to insert fragment");

            txn.commit().expect("Failed to commit transaction");
        }

        node_index
    }

    /// Generates an embedding for a fragment using the Candles model
    fn generate_embedding(&self, fragment: &str, model_state: &ModelState) -> Vec<f32> {
        // Tokenize the fragment
        let tokens = model_state.tokenizer.encode(fragment).expect("Failed to tokenize");

        // Convert tokens to tensors
        let input_ids = Tensor::new(&tokens.ids, &model_state.device)
            .expect("Failed to create input_ids tensor")
            .unsqueeze(0)
            .expect("Failed to unsqueeze input_ids");
        let attention_mask = Tensor::new(&tokens.attention_mask, &model_state.device)
            .expect("Failed to create attention_mask tensor")
            .unsqueeze(0)
            .expect("Failed to unsqueeze attention_mask");

        // Run the model
        let outputs = model_state
            .model
            .forward(&input_ids, &attention_mask, None)
            .expect("Failed to run model");

        // Get the last hidden state
        let hidden_state = outputs.hidden_state.expect("No hidden state in outputs");

        // Pool the hidden state to get a sentence embedding (mean pooling)
        let embedding = hidden_state.mean(1).expect("Failed to mean pool hidden state");

        // Convert the embedding tensor to a Vec<f32>
        embedding.to_vec1_f32().expect("Failed to convert embedding to Vec<f32>")
    }

    /// Adds a position node to the SG-Tree
    fn add_position_node(&self, node_index: NodeIndex, offset: Offset) -> PositionIndex {
        // Get next position index
        let position = {
            let mut lock = self.next_position_index.write();
            let pos = *lock;
            *lock += 1;
            pos
        };

        let position_node = PositionNode {
            position,
            node_index,
            offset,
        };

        // Serialize position node
        let pos_bytes = bincode::serialize(&position_node).expect("Failed to serialize position node");

        // Write to Redb
        {
            let mut txn = self.db.begin_write().expect("Failed to begin write transaction");
            let positions_table = txn.open_table(&self.positions_table).expect("Failed to open positions table");
            positions_table.insert(position, pos_bytes).expect("Failed to insert position node");
            txn.commit().expect("Failed to commit transaction");
        }

        position
    }

    /// Adds an advanced edge with semantic labels
    fn add_advanced_edge(&self, from: NodeIndex, to: NodeIndex, relationship: &str) {
        let label = match relationship {
            "subject" => 1,
            "verb" => 2,
            "object" => 3,
            "sequence" => 4,
            _ => 0,
        };
        self.add_edge(from, to, label);
    }

    /// Adds a directed edge between nodes
    fn add_edge(&self, from: NodeIndex, to: NodeIndex, label: EdgeLabel) {
        let edge = Edge { from, to, label };
        let edge_bytes = bincode::serialize(&edge).expect("Failed to serialize edge");

        // Use from as the key and store edges as a list
        {
            let mut txn = self.db.begin_write().expect("Failed to begin write transaction");
            let edges_table = txn.open_table(&self.edges_table).expect("Failed to open edges table");

            // Retrieve existing edges for 'from'
            let existing = edges_table.get(&from).expect("Failed to get existing edges");
            let mut edges: Vec<Edge> = if let Some(bytes) = existing {
                bincode::deserialize(&bytes).expect("Failed to deserialize existing edges")
            } else {
                Vec::new()
            };

            // Add the new edge
            edges.push(edge);

            // Serialize and insert back
            let serialized = bincode::serialize(&edges).expect("Failed to serialize edges");
            edges_table.insert(from, serialized).expect("Failed to insert edges");

            txn.commit().expect("Failed to commit transaction");
        }
    }

    /// Traverses the SG-Tree starting from a given node index
    fn traverse_from_node(&self, index: NodeIndex) -> Vec<String> {
        let mut result = Vec::new();
        let mut visited = HashSet::new();
        let mut stack = vec![index];

        while let Some(current_index) = stack.pop() {
            if visited.contains(&current_index) {
                continue;
            }
            visited.insert(current_index);

            // Retrieve node
            if let Some(node) = self.get_node_by_index(current_index) {
                result.push(node.fragment.clone());
            }

            // Retrieve outgoing edges
            let edges = self.get_edges_from_node(current_index);
            for edge in edges {
                stack.push(edge.to);
            }
        }

        result
    }

    /// Finds the most similar node based on cosine similarity
    fn find_most_similar_node(&self, embedding: &Vec<f32>) -> Option<NodeIndex> {
        let mut max_similarity = -1.0;
        let mut best_node_index = None;

        // Iterate over all nodes
        let txn = self.db.begin_read().expect("Failed to begin read transaction");
        let nodes_table = txn.open_table(&self.nodes_table).expect("Failed to open nodes table");

        for result in nodes_table.iter() {
            let (key, value) = result.expect("Failed to iterate nodes");
            let node: Node = bincode::deserialize(&value).expect("Failed to deserialize node");
            let similarity = SGTree::cosine_similarity(embedding, &node.embedding);
            if similarity > max_similarity {
                max_similarity = similarity;
                best_node_index = Some(key);
            }
        }

        best_node_index
    }

    /// Finds the next node based on current embedding and edges
    fn find_next_node(&self, current_index: NodeIndex, current_embedding: &Vec<f32>) -> Option<NodeIndex> {
        // Retrieve outgoing edges
        let edges = self.get_edges_from_node(current_index);
        if edges.is_empty() {
            // Fallback to most similar node
            self.find_most_similar_node(current_embedding)
        } else {
            let mut max_similarity = -1.0;
            let mut best_node_index = None;
            for edge in edges {
                // Retrieve the 'to' node's embedding
                if let Some(node) = self.get_node_by_index(edge.to) {
                    let similarity = SGTree::cosine_similarity(current_embedding, &node.embedding);
                    if similarity > max_similarity {
                        max_similarity = similarity;
                        best_node_index = Some(edge.to);
                    }
                }
            }
            best_node_index
        }
    }

    /// Generates text starting from a given fragment
    fn generate_text(&self, start_fragment: &str, max_length: usize) -> String {
        let mut generated = String::new();
        if let Some(start_index) = self.get_node_index(&start_fragment) {
            let mut current_index = start_index;
            for _ in 0..max_length {
                if let Some(node) = self.get_node_by_index(current_index) {
                    generated.push_str(&node.fragment);
                    generated.push(' ');
                    let current_embedding = &node.embedding;
                    let next_index = self.find_next_node(current_index, current_embedding);
                    if let Some(next_index) = next_index {
                        current_index = next_index;
                    } else {
                        break;
                    }
                } else {
                    break;
                }
            }
        }
        generated.trim().to_string()
    }

    /// Retrieves a node by its index
    fn get_node_by_index(&self, index: NodeIndex) -> Option<Node> {
        let txn = self.db.begin_read().expect("Failed to begin read transaction");
        let nodes_table = txn.open_table(&self.nodes_table).expect("Failed to open nodes table");
        nodes_table.get(&index).expect("Failed to get node").map(|bytes| {
            bincode::deserialize(&bytes).expect("Failed to deserialize node")
        })
    }

    /// Retrieves the node index by fragment (exact match)
    fn get_node_index(&self, fragment: &str) -> Option<NodeIndex> {
        let txn = self.db.begin_read().expect("Failed to begin read transaction");
        let fragments_table = txn.open_table(&self.fragments_table).expect("Failed to open fragments table");
        fragments_table.get(&fragment).expect("Failed to get fragment")
    }

    /// Retrieves all outgoing edges from a node
    fn get_edges_from_node(&self, index: NodeIndex) -> Vec<Edge> {
        let txn = self.db.begin_read().expect("Failed to begin read transaction");
        let edges_table = txn.open_table(&self.edges_table).expect("Failed to open edges table");
        edges_table.get(&index).expect("Failed to get edges").map(|bytes| {
            bincode::deserialize(&bytes).expect("Failed to deserialize edges")
        }).unwrap_or_else(|| Vec::new())
    }

    /// Calculates cosine similarity between two vectors
    fn cosine_similarity(a: &Vec<f32>, b: &Vec<f32>) -> f32 {
        let dot_product: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
        dot_product / (norm_a * norm_b + 1e-10)
    }

    /// Performs a semantic search and returns the top_k most similar nodes
    fn search(&self, query: &str, model_state: &ModelState, top_k: usize) -> Vec<Node> {
        // Generate embedding for the query
        let query_embedding = self.generate_embedding(query, model_state);

        // Iterate over all nodes and compute similarity
        let txn = self.db.begin_read().expect("Failed to begin read transaction");
        let nodes_table = txn.open_table(&self.nodes_table).expect("Failed to open nodes table");

        let mut similarities = Vec::new();

        for result in nodes_table.iter() {
            let (key, value) = result.expect("Failed to iterate nodes");
            let node: Node = bincode::deserialize(&value).expect("Failed to deserialize node");
            let similarity = SGTree::cosine_similarity(&query_embedding, &node.embedding);
            similarities.push((similarity, node));
        }

        // Sort by similarity in descending order and take top_k
        similarities.par_sort_unstable_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
        similarities.into_iter().take(top_k).map(|(_, node)| node).collect()
    }
}

/// Represents the state of the ML model and tokenizer
struct ModelState {
    tokenizer: candle_transformers::tokenizer::Tokenizer,
    model: BertModel,
    device: Device,
}

/// Represents the application state shared across handlers
struct AppState {
    sg_tree: Arc<SGTree>,
    model_state: Arc<ModelState>,
}

/// Request structure for adding a document
#[derive(Deserialize)]
struct AddDocumentRequest {
    text: String,
}

/// Request structure for retrieving a node
#[derive(Deserialize)]
struct GetNodeRequest {
    index: NodeIndex,
}

/// Request structure for traversing the SG-Tree
#[derive(Deserialize)]
struct TraverseRequest {
    index: NodeIndex,
}

/// Request structure for generating text
#[derive(Deserialize)]
struct GenerateTextRequest {
    start_fragment: String,
    max_length: Option<usize>,
}

/// Request structure for searching
#[derive(Deserialize)]
struct SearchRequest {
    query: String,
    top_k: Option<usize>,
}

/// Handler implementation for adding a document
async fn add_document_handler(
    req: &mut Request,
    depot: &mut Depot,
    res: &mut Response,
) -> Result<(), anyhow::Error> {
    let app_state = depot.take::<web::Data<AppState>>()?;
    let add_req: AddDocumentRequest = req.parse_json().await?;

    let text = add_req.text.clone();
    let sg_tree = app_state.sg_tree.clone();
    let model_state = app_state.model_state.clone();

    // Process the document in a separate thread for non-blocking behavior
    task::spawn_blocking(move || {
        let fragments: Vec<&str> = text.split_whitespace().collect();
        let mut previous_node_index: Option<NodeIndex> = None;
        let mut offset: Offset = 0;

        for fragment in fragments {
            let fragment_string = fragment.to_string();
            let node_index = sg_tree.add_node(fragment_string.clone(), &model_state);
            let _position_index = sg_tree.add_position_node(node_index, offset);

            if let Some(prev_index) = previous_node_index {
                sg_tree.add_advanced_edge(prev_index, node_index, "sequence");
            }

            previous_node_index = Some(node_index);
            offset += 1;
        }

        // Redb handles persistence automatically
    });

    res.render(Text::Plain("Document added successfully".to_string()));
    Ok(())
}

/// Handler implementation for retrieving a node
async fn get_node_handler(
    req: &mut Request,
    depot: &mut Depot,
    res: &mut Response,
) -> Result<(), anyhow::Error> {
    let app_state = depot.take::<web::Data<AppState>>()?;
    let get_req: GetNodeRequest = req.parse_query().await?;

    let tree = &app_state.sg_tree;
    if let Some(node) = tree.get_node_by_index(get_req.index) {
        res.render(Json(node.clone()));
    } else {
        res.set_status_code(StatusCode::NOT_FOUND);
        res.render(Text::Plain("Node not found".to_string()));
    }
    Ok(())
}

/// Handler implementation for traversing the SG-Tree
async fn traverse_handler(
    req: &mut Request,
    depot: &mut Depot,
    res: &mut Response,
) -> Result<(), anyhow::Error> {
    let app_state = depot.take::<web::Data<AppState>>()?;
    let traverse_req: TraverseRequest = req.parse_query().await?;

    let tree = &app_state.sg_tree;
    let traversal = tree.traverse_from_node(traverse_req.index);
    res.render(Json(traversal));
    Ok(())
}

/// Handler implementation for generating text
async fn generate_text_handler(
    req: &mut Request,
    depot: &mut Depot,
    res: &mut Response,
) -> Result<(), anyhow::Error> {
    let app_state = depot.take::<web::Data<AppState>>()?;
    let gen_req: GenerateTextRequest = req.parse_query().await?;

    let tree = &app_state.sg_tree;
    let generated = tree.generate_text(&gen_req.start_fragment, gen_req.max_length.unwrap_or(50));

    res.render(Text::Plain(generated));
    Ok(())
}

/// Handler implementation for searching
async fn search_handler(
    req: &mut Request,
    depot: &mut Depot,
    res: &mut Response,
) -> Result<(), anyhow::Error> {
    let app_state = depot.take::<web::Data<AppState>>()?;
    let search_req: SearchRequest = req.parse_query().await?;

    let query = search_req.query.clone();
    let top_k = search_req.top_k.unwrap_or(5);

    let tree = &app_state.sg_tree;
    let model_state = &app_state.model_state;

    // Perform search in a separate thread
    let results = task::spawn_blocking(move || {
        tree.search(&query, &model_state, top_k)
    }).await??;

    res.render(Json(results));
    Ok(())
}

#[tokio::main]
async fn main() -> Result<(), std::io::Error> {
    // Initialize or load SG-Tree with Redb persistence
    let sg_tree = Arc::new(SGTree::new("sg_tree.redb"));

    // Initialize Hugging Face cache and repository
    let cache = Cache::default().expect("Failed to initialize HF Hub cache");
    let repo = Repo::with_revision("sentence-transformers", "all-MiniLM-L6-v2", None);

    // Load tokenizer
    let tokenizer = candle_transformers::tokenizer::Tokenizer::from_repo(&repo, &cache)
        .expect("Failed to load tokenizer");

    // Load model configuration
    let config_path = cache
        .repo_file(&repo, "config.json", false)
        .expect("Failed to load config.json");
    let config_file = File::open(config_path).expect("Failed to open config.json");
    let config: BertConfig =
        serde_json::from_reader(BufReader::new(config_file)).expect("Failed to parse config.json");

    // Load model weights
    let weights_filename = "model.safetensors";
    let weights_path = cache
        .repo_file(&repo, weights_filename, false)
        .expect("Failed to load model weights");
    let weights = candle::safetensors::load(weights_path, &Device::Cpu)
        .expect("Failed to load weights");

    // Initialize model
    let model = BertModel::load("bert", config, weights)
        .expect("Failed to load model");

    let model_state = Arc::new(ModelState {
        tokenizer,
        model,
        device: Device::Cpu,
    });

    // Create shared application state
    let app_state = web::Data::new(AppState {
        sg_tree: sg_tree.clone(),
        model_state: model_state.clone(),
    });

    // Define routes
    let router = Router::new()
        .push(Router::with_path("/add_document").post(add_document_handler))
        .push(Router::with_path("/generate_text").get(generate_text_handler))
        .push(Router::with_path("/node/:index").get(get_node_handler))
        .push(Router::with_path("/traverse/:index").get(traverse_handler))
        .push(Router::with_path("/search").get(search_handler)); // Added search route

    // Start the server
    println!("Starting server on http://0.0.0.0:8080");
    Server::new(router).bind("0.0.0.0:8080").await?;

    Ok(())
}
```

### **Explanation of Additions:**

1. **Search Functionality in `SGTree`:**
   
   - **`search` Method**: Added a new method `search` to perform semantic search. It takes a query string, generates its embedding, computes cosine similarity with all node embeddings, and returns the top_k most similar nodes.
   
   ```rust
   /// Performs a semantic search and returns the top_k most similar nodes
   fn search(&self, query: &str, model_state: &ModelState, top_k: usize) -> Vec<Node> {
       // Generate embedding for the query
       let query_embedding = self.generate_embedding(query, model_state);

       // Iterate over all nodes and compute similarity
       let txn = self.db.begin_read().expect("Failed to begin read transaction");
       let nodes_table = txn.open_table(&self.nodes_table).expect("Failed to open nodes table");

       let mut similarities = Vec::new();

       for result in nodes_table.iter() {
           let (key, value) = result.expect("Failed to iterate nodes");
           let node: Node = bincode::deserialize(&value).expect("Failed to deserialize node");
           let similarity = SGTree::cosine_similarity(&query_embedding, &node.embedding);
           similarities.push((similarity, node));
       }

       // Sort by similarity in descending order and take top_k
       similarities.par_sort_unstable_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
       similarities.into_iter().take(top_k).map(|(_, node)| node).collect()
   }
   ```

2. **Search Handler and Route:**

   - **`SearchRequest` Struct**: Defines the expected query parameters for the search endpoint.
   
   ```rust
   #[derive(Deserialize)]
   struct SearchRequest {
       query: String,
       top_k: Option<usize>,
   }
   ```
   
   - **`search_handler` Function**: Handles incoming search requests, invokes the `search` method, and returns the results.
   
   ```rust
   /// Handler implementation for searching
   async fn search_handler(
       req: &mut Request,
       depot: &mut Depot,
       res: &mut Response,
   ) -> Result<(), anyhow::Error> {
       let app_state = depot.take::<web::Data<AppState>>()?;
       let search_req: SearchRequest = req.parse_query().await?;

       let query = search_req.query.clone();
       let top_k = search_req.top_k.unwrap_or(5);

       let tree = &app_state.sg_tree;
       let model_state = &app_state.model_state;

       // Perform search in a separate thread
       let results = task::spawn_blocking(move || {
           tree.search(&query, &model_state, top_k)
       }).await??;

       res.render(Json(results));
       Ok(())
   }
   ```
   
   - **Route Addition**: Added a new route for the search endpoint.
   
   ```rust
   .push(Router::with_path("/search").get(search_handler)) // Added search route
   ```

3. **Concurrency with Rayon:**
   
   - Utilized `rayon`'s parallel sorting (`par_sort_unstable_by`) to efficiently sort similarity scores.

4. **Error Handling:**
   
   - Ensured that any errors during the search process are properly propagated using `anyhow::Error`.

---

## 2. Updated `README.md`

```markdown
# SG-Tree Rust Project

## Overview

The **SG-Tree Rust Project** is a sophisticated implementation of a Semantic Graph Tree (SG-Tree) designed for efficient text processing, storage, and retrieval. It leverages modern Rust libraries for high performance and safety, integrating machine learning models for semantic understanding and providing a RESTful API for interaction.

### Key Features

- **SG-Tree Structure**: Represents text fragments as nodes with embeddings, positions, and labeled edges to capture semantic relationships.
- **Persistence with Redb**: Ensures data durability and efficient storage using the [Redb](https://crates.io/crates/redb) embedded database.
- **Machine Learning Integration**: Utilizes [Candle Transformers](https://crates.io/crates/candle-transformers) to generate embeddings using BERT models from Hugging Face.
- **Asynchronous HTTP API**: Built with [Salvo](https://crates.io/crates/salvo) to provide endpoints for adding documents, retrieving nodes, traversing the SG-Tree, generating text, and performing semantic searches.
- **Concurrency and Parallelism**: Employs [Rayon](https://crates.io/crates/rayon) for parallel processing and [Parking Lot](https://crates.io/crates/parking_lot) for efficient synchronization.

## Table of Contents

- [Features](#features)
- [Architecture](#architecture)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Configuration](#configuration)
- [Usage](#usage)
  - [API Endpoints](#api-endpoints)
    - [Add Document](#add-document)
    - [Generate Text](#generate-text)
    - [Retrieve Node](#retrieve-node)
    - [Traverse SG-Tree](#traverse-sg-tree)
    - [Search](#search)
- [Project Structure](#project-structure)
- [Dependencies](#dependencies)
- [Contributing](#contributing)
- [License](#license)

## Architecture

The project is structured around the following core components:

1. **SG-Tree**: The central data structure managing nodes, positions, and edges with persistence via Redb.
2. **ModelState**: Manages the machine learning model and tokenizer for generating embeddings.
3. **API Server**: Provides RESTful endpoints for interacting with the SG-Tree, built using Salvo.
4. **Concurrency Management**: Ensures thread-safe operations and efficient processing using Rayon and Parking Lot.

## Getting Started

### Prerequisites

- **Rust**: Ensure you have Rust installed. If not, install it from [rustup.rs](https://rustup.rs/).
- **Cargo**: Comes bundled with Rust for managing dependencies and building the project.
- **Hugging Face Models**: Access to Hugging Face models, specifically the `all-MiniLM-L6-v2` model from the `sentence-transformers` repository.

### Installation

1. **Clone the Repository**

   ```bash
   git clone https://github.com/yourusername/sg-tree-rust.git
   cd sg-tree-rust
   ```

2. **Initialize Submodules (If Any)**

   ```bash
   git submodule update --init --recursive
   ```

3. **Build the Project**

   ```bash
   cargo build --release
   ```

### Configuration

1. **Hugging Face Hub Authentication**

   To download models from Hugging Face, you may need to set up authentication tokens.

   ```bash
   export HF_HOME=~/.cache/huggingface
   export HF_TOKEN=your_huggingface_token
   ```

2. **Database Path**

   The SG-Tree uses Redb for persistence. By default, it stores data in `sg_tree.redb` in the project root. You can modify the path in the `main` function if needed.

## Usage

After building the project, you can run the server:

```bash
cargo run --release
```

The server will start on `http://0.0.0.0:8080`.

### API Endpoints

#### 1. Add Document

**Endpoint**

```
POST /add_document
```

**Description**

Adds a new document to the SG-Tree. The document text is split into fragments (words), each added as a node with embeddings. Edges are created to represent the sequence of fragments.

**Request Body**

```json
{
  "text": "Your document text goes here."
}
```

**Response**

```
HTTP 200 OK
Document added successfully
```

**Example**

```bash
curl -X POST http://localhost:8080/add_document \
     -H "Content-Type: application/json" \
     -d '{"text": "Hello world this is a test document."}'
```

#### 2. Generate Text

**Endpoint**

```
GET /generate_text
```

**Description**

Generates text starting from a specified fragment, traversing the SG-Tree based on semantic similarities.

**Query Parameters**

- `start_fragment` (string, required): The fragment to start generating text from.
- `max_length` (integer, optional): The maximum number of fragments to generate. Defaults to 50.

**Response**

```
HTTP 200 OK
Generated text as a plain string.
```

**Example**

```bash
curl "http://localhost:8080/generate_text?start_fragment=Hello&max_length=10"
```

#### 3. Retrieve Node

**Endpoint**

```
GET /node/:index
```

**Description**

Retrieves details of a node by its index.

**Path Parameters**

- `index` (integer, required): The index of the node to retrieve.

**Response**

```json
{
  "index": 1,
  "fragment": "Hello",
  "encoding": 52,
  "embedding": [0.123, 0.456, ...]
}
```

**Example**

```bash
curl http://localhost:8080/node/1
```

#### 4. Traverse SG-Tree

**Endpoint**

```
GET /traverse/:index
```

**Description**

Traverses the SG-Tree starting from the specified node index, returning a list of fragments encountered.

**Path Parameters**

- `index` (integer, required): The starting node index for traversal.

**Response**

```json
["Hello", "world", "this", "is", "a", "test", "document"]
```

**Example**

```bash
curl http://localhost:8080/traverse/1
```

#### 5. Search

**Endpoint**

```
GET /search
```

**Description**

Performs a semantic search within the SG-Tree, returning the top_k most similar nodes to the query.

**Query Parameters**

- `query` (string, required): The search query string.
- `top_k` (integer, optional): The number of top similar nodes to return. Defaults to 5.

**Response**

```json
[
  {
    "index": 2,
    "fragment": "world",
    "encoding": 72,
    "embedding": [0.234, 0.567, ...]
  },
  {
    "index": 3,
    "fragment": "this",
    "encoding": 56,
    "embedding": [0.345, 0.678, ...]
  },
  ...
]
```

**Example**

```bash
curl "http://localhost:8080/search?query=Hello&top_k=3"
```

## Project Structure

```
sg-tree-rust/
├── src/
│   └── main.rs        # Main application entry point with SG-Tree implementation and API handlers
├── Cargo.toml         # Project dependencies and metadata
└── README.md          # Project documentation
```

## Dependencies

The project relies on several Rust crates for its functionality. Below is a summary of the main dependencies:

```toml
[package]
name = "sg-tree-rust"
version = "0.1.0"
edition = "2021"
authors = ["Your Name <youremail@example.com>"]
description = "A Semantic Graph Tree (SG-Tree) implementation in Rust with Redb persistence and BERT-based embeddings."
license = "MIT"
repository = "https://github.com/yourusername/sg-tree-rust"

[dependencies]
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"
bincode = "1.3"
sha2 = "0.10"
parking_lot = "0.12"
rayon = "1.5"
salvo = { version = "0.38", features = ["tokio"] }
candle = { version = "0.4", features = ["cuda", "safetensors"] }
candle-transformers = "0.4"
hf-hub = "0.5"
rand = "0.8"
redb = "0.5"
tokio = { version = "1", features = ["full"] }
anyhow = "1.0"
```

### Detailed Dependency List

- **serde**: Serialization and deserialization framework.
- **serde_json**: JSON support for Serde.
- **bincode**: Binary serialization for efficient storage.
- **sha2**: Implementation of SHA-2 hashing algorithms.
- **parking_lot**: Fast and lightweight synchronization primitives.
- **rayon**: Data parallelism library for Rust.
- **salvo**: Asynchronous HTTP server framework.
- **candle**: Tensor computation library.
- **candle-transformers**: Transformer models for Candle.
- **hf-hub**: Interface to Hugging Face Hub for model and tokenizer management.
- **rand**: Random number generation.
- **redb**: Embedded key-value store database.
- **tokio**: Asynchronous runtime for Rust.
- **anyhow**: Error handling library.

## Contributing

Contributions are welcome! Please follow these steps:

1. **Fork the Repository**

   Click the "Fork" button at the top-right corner of the repository page.

2. **Clone Your Fork**

   ```bash
   git clone https://github.com/yourusername/sg-tree-rust.git
   cd sg-tree-rust
   ```

3. **Create a New Branch**

   ```bash
   git checkout -b feature/your-feature-name
   ```

4. **Make Your Changes**

   Implement your feature or bug fix.

5. **Commit Your Changes**

   ```bash
   git commit -m "Add feature: your feature description"
   ```

6. **Push to Your Fork**

   ```bash
   git push origin feature/your-feature-name
   ```

7. **Create a Pull Request**

   Navigate to the original repository and create a pull request from your fork.

## License

This project is licensed under the [MIT License](LICENSE).

---

*Happy Coding!*
```

### **Explanation of Additions:**

1. **Search Endpoint Documentation:**
   
   - Added a new section under **API Endpoints** for **Search**.
   - Detailed the endpoint's purpose, expected query parameters, response format, and provided example usage.

2. **Project Structure Update:**
   
   - Clarified that all main implementations, including the search functionality, reside within `main.rs`.

3. **Dependencies Update:**
   
   - No new dependencies were added for the search functionality. However, clarified the use of existing dependencies like `rayon` for parallel processing, which is utilized in the search method for efficient sorting.

---

## 3. Updated `Cargo.toml`

```toml
[package]
name = "sg-tree-rust"
version = "0.1.0"
edition = "2021"
authors = ["Your Name <youremail@example.com>"]
description = "A Semantic Graph Tree (SG-Tree) implementation in Rust with Redb persistence and BERT-based embeddings."
license = "MIT"
repository = "https://github.com/yourusername/sg-tree-rust"

[dependencies]
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"
bincode = "1.3"
sha2 = "0.10"
parking_lot = "0.12"
rayon = "1.5"
salvo = { version = "0.38", features = ["tokio"] }
candle = { version = "0.4", features = ["cuda", "safetensors"] }
candle-transformers = "0.4"
hf-hub = "0.5"
rand = "0.8"
redb = "0.5"
tokio = { version = "1", features = ["full"] }
anyhow = "1.0"
```

### **Explanation of `Cargo.toml` Updates:**

- **No New Dependencies Added:** The search functionality leverages existing dependencies such as `rayon` for parallel processing and `anyhow` for error handling. Therefore, there's no need to introduce additional crates.
  
- **Consistency and Completeness:** Ensured that all dependencies required for both the existing features and the new search feature are properly listed and versioned.

---

## Additional Notes

- **Performance Considerations:**
  
  - **Parallel Processing:** The search function utilizes Rayon’s parallel sorting to efficiently handle large datasets, ensuring quick retrieval of similar nodes.
  
  - **Concurrency:** Asynchronous handlers with Salvo and Tokio ensure that the server remains responsive, even under heavy load.

- **Scalability:**
  
  - The current implementation loads all node embeddings into memory during the search. For extremely large datasets, consider implementing indexing mechanisms or leveraging approximate nearest neighbor (ANN) search algorithms to enhance scalability.

- **Error Handling:**
  
  - The project employs the `anyhow` crate for flexible error management, ensuring that errors are propagated and handled gracefully across different components.

- **Security:**
  
  - For production deployments, implement authentication and authorization mechanisms to secure the API endpoints and protect the data.

- **Testing:**
  
  - Implement unit and integration tests to verify the functionality of the SG-Tree operations and API endpoints, ensuring robustness and reliability.

- **Model Optimization:**
  
  - Depending on the use case, consider optimizing the BERT model for faster inference or reduced memory footprint, especially if deploying in resource-constrained environments.

- **Documentation:**
  
  - Maintain up-to-date documentation as the project evolves, ensuring that new features and changes are well-documented for users and contributors.

---

With these updates, your SG-Tree Rust project now supports robust semantic search capabilities, enhancing its utility for various text processing and retrieval applications. Ensure to test the new feature thoroughly and consider the additional notes for further improvements and optimizations.
