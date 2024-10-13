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
use half::f16;
use ann::{AnnIndex, Metric, Searcher};

// Type Definitions
type NodeIndex = u32;
type Offset = u32;
type PositionIndex = u32;
type FragmentEncoding = u8;
type EdgeLabel = u16;

// Represents a unique node in the SG-Tree
#[derive(Serialize, Deserialize, Debug, Clone)]
struct Node {
    index: NodeIndex,
    fragment: String,
    encoding: FragmentEncoding,
    embedding: Vec<f16>, // Changed from Vec<f32> to Vec<f16>
}

// Represents a position node in the SG-Tree
#[derive(Serialize, Deserialize, Debug, Clone)]
struct PositionNode {
    position: PositionIndex,
    node_index: NodeIndex,
    offset: Offset,
}

// Represents a directed edge between nodes in the SG-Tree
#[derive(Serialize, Deserialize, Debug, Clone)]
struct Edge {
    from: NodeIndex,
    to: NodeIndex,
    label: EdgeLabel,
}

// SG-Tree structure encapsulating nodes, positions, and edges with Redb persistence
struct SGTree {
    db: Arc<Database>,
    nodes_table: TableDefinition<u32, Vec<u8>>,
    fragments_table: TableDefinition<String, u32>,
    positions_table: TableDefinition<u32, Vec<u8>>,
    edges_table: TableDefinition<u32, Vec<u8>>,
    embeddings_table: TableDefinition<u32, Vec<u8>>, // New table for embeddings
    next_node_index: RwLock<NodeIndex>,
    next_position_index: RwLock<PositionIndex>,
    ann_index: RwLock<AnnIndex<u32, Vec<f16>>>, // ANN index for embeddings
}

impl SGTree {
    /// Initializes a new SG-Tree with Redb persistence and ANN index
    fn new(db_path: &str) -> Self {
        // Open or create the Redb database
        let db = Database::create(db_path).expect("Failed to create or open Redb database");

        // Define tables
        let nodes_table = TableDefinition::new("nodes");
        let fragments_table = TableDefinition::new("fragments");
        let positions_table = TableDefinition::new("positions");
        let edges_table = TableDefinition::new("edges");
        let embeddings_table = TableDefinition::new("embeddings"); // Initialize embeddings table

        // Initialize tables if they don't exist
        {
            let mut txn = db.begin_write().expect("Failed to begin write transaction");
            txn.open_table_or_create(&nodes_table).expect("Failed to open/create nodes table");
            txn.open_table_or_create(&fragments_table).expect("Failed to open/create fragments table");
            txn.open_table_or_create(&positions_table).expect("Failed to open/create positions table");
            txn.open_table_or_create(&edges_table).expect("Failed to open/create edges table");
            txn.open_table_or_create(&embeddings_table).expect("Failed to open/create embeddings table"); // Create embeddings table
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

        // Initialize ANN index (using Euclidean distance as an example)
        let ann_index = AnnIndex::new(768, Metric::Euclidean); // Assuming original embeddings are 768-dimensional
        RwLock::new(ann_index)

        SGTree {
            db: Arc::new(db),
            nodes_table,
            fragments_table,
            positions_table,
            edges_table,
            embeddings_table,
            next_node_index,
            next_position_index,
            ann_index: RwLock::new(AnnIndex::new(768, Metric::Euclidean)),
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

        // Convert embedding from Vec<f32> to Vec<f16>
        let embedding_f16: Vec<f16> = embedding.iter().map(|&x| f16::from_f32(x)).collect();

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
            embedding: embedding_f16.clone(),
        };

        // Serialize node without embedding to save space
        let node_clone = Node {
            embedding: vec![], // Exclude embedding from node serialization
            ..node.clone()
        };
        let node_bytes = bincode::serialize(&node_clone).expect("Failed to serialize node");

        // Serialize embedding separately
        let embedding_bytes = bincode::serialize(&embedding_f16).expect("Failed to serialize embedding");

        // Write to Redb
        {
            let mut txn = self.db.begin_write().expect("Failed to begin write transaction");
            let nodes_table = txn.open_table(&self.nodes_table).expect("Failed to open nodes table");
            nodes_table.insert(node_index, node_bytes).expect("Failed to insert node");

            let fragments_table = txn.open_table(&self.fragments_table).expect("Failed to open fragments table");
            fragments_table.insert(fragment, node_index).expect("Failed to insert fragment");

            let embeddings_table = txn.open_table(&self.embeddings_table).expect("Failed to open embeddings table");
            embeddings_table.insert(node_index, embedding_bytes).expect("Failed to insert embedding");

            txn.commit().expect("Failed to commit transaction");
        }

        // Add embedding to ANN index
        {
            let mut ann = self.ann_index.write();
            ann.add(node_index, embedding_f16.clone()).expect("Failed to add to ANN index");
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

    /// Finds the most similar node based on cosine similarity using ANN
    fn find_most_similar_node(&self, embedding: &Vec<f16>) -> Option<NodeIndex> {
        let ann = self.ann_index.read();
        let searcher = ann.searcher();

        // Perform ANN search (top 1)
        if let Some((score, node_index)) = searcher.nearest(embedding, 1).next() {
            Some(*node_index)
        } else {
            None
        }
    }

    /// Finds the next node based on current embedding and edges using ANN
    fn find_next_node(&self, current_index: NodeIndex, current_embedding: &Vec<f16>) -> Option<NodeIndex> {
        // Retrieve outgoing edges
        let edges = self.get_edges_from_node(current_index);
        if edges.is_empty() {
            // Fallback to most similar node
            self.find_most_similar_node(current_embedding)
        } else {
            let mut best_node_index = None;
            let mut max_similarity = -1.0;

            // Iterate through edges to find the most similar 'to' node
            for edge in edges {
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

    /// Performs a semantic search and returns the top_k most similar nodes
    fn search(&self, query: &str, model_state: &ModelState, top_k: usize) -> Vec<Node> {
        // Generate embedding for the query
        let query_embedding = self.generate_embedding(query, model_state);

        // Convert to f16
        let query_embedding_f16: Vec<f16> = query_embedding.iter().map(|&x| f16::from_f32(x)).collect();

        // Perform ANN search
        let ann = self.ann_index.read();
        let searcher = ann.searcher();
        let mut results = Vec::new();

        for (score, node_index) in searcher.nearest(&query_embedding_f16, top_k) {
            if let Some(node) = self.get_node_by_index(*node_index) {
                results.push(node);
            }
        }

        results
    }

    /// Retrieves a node by its index
    fn get_node_by_index(&self, index: NodeIndex) -> Option<Node> {
        let txn = self.db.begin_read().expect("Failed to begin read transaction");
        let nodes_table = txn.open_table(&self.nodes_table).expect("Failed to open nodes table");
        nodes_table.get(&index).expect("Failed to get node").map(|bytes| {
            let mut node: Node = bincode::deserialize(&bytes).expect("Failed to deserialize node");
            // Retrieve embedding separately
            let embeddings_table = txn.open_table(&self.embeddings_table).expect("Failed to open embeddings table");
            if let Some(embedding_bytes) = embeddings_table.get(&index).expect("Failed to get embedding") {
                node.embedding = bincode::deserialize(&embedding_bytes).expect("Failed to deserialize embedding");
            }
            node
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
    fn cosine_similarity(a: &Vec<f16>, b: &Vec<f16>) -> f32 {
        let a_f32: Vec<f32> = a.iter().map(|&x| x.to_f32()).collect();
        let b_f32: Vec<f32> = b.iter().map(|&x| x.to_f32()).collect();

        let dot_product: f32 = a_f32.iter().zip(b_f32.iter()).map(|(x, y)| x * y).sum();
        let norm_a: f32 = a_f32.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm_b: f32 = b_f32.iter().map(|x| x * x).sum::<f32>().sqrt();
        dot_product / (norm_a * norm_b + 1e-10)
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

        fragments.par_iter().for_each(|&fragment| {
            let fragment_string = fragment.to_string();
            let node_index = sg_tree.add_node(fragment_string.clone(), &model_state);
            let _position_index = sg_tree.add_position_node(node_index, offset);

            if let Some(prev_index) = previous_node_index {
                sg_tree.add_advanced_edge(prev_index, node_index, "sequence");
            }

            // Update previous_node_index safely
            // Note: This is a simplified example; in a real-world scenario, proper synchronization is needed
            // Here, we assume that the order of processing does not affect the relationships
            previous_node_index = Some(node_index);
            offset += 1;
        });

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
