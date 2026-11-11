use std::collections::{HashMap, HashSet};

// Represents one section of the PDF.
#[derive(Clone)]
pub struct Chunk {
    pub id: usize,
    pub text: String,
}

// Stores all PDF chunks and their TF-IDF index.
#[derive(Clone)]
pub struct RagIndex {
    pub chunks: Vec<Chunk>,
    idf: HashMap<String, f64>,
    chunk_vectors: Vec<HashMap<String, f64>>,
}

// ------------------------------------------
// TEXT CHUNKING
// ------------------------------------------

pub fn chunk_text(text: &str, chunk_size: usize, overlap: usize) -> Vec<Chunk> {
    assert!(chunk_size > 0, "Chunk size must be greater than zero.");
    assert!(
        overlap < chunk_size,
        "Overlap must be smaller than chunk size."
    );

    let characters: Vec<char> = text.chars().collect();

    let mut chunks = Vec::new();
    let mut start = 0;

    while start < characters.len() {
        let end = (start + chunk_size).min(characters.len());

        let content: String = characters[start..end].iter().collect();

        if !content.trim().is_empty() {
            chunks.push(Chunk {
                id: chunks.len(),
                text: content,
            });
        }

        if end == characters.len() {
            break;
        }

        start = end - overlap;
    }

    chunks
}

// ------------------------------------------
// TOKENIZATION
// ------------------------------------------

fn tokenize(text: &str) -> Vec<String> {
    text.to_lowercase()
        .split(|c: char| !c.is_alphanumeric())
        .filter(|word| word.len() > 2)
        .filter(|word| {
            !matches!(
                *word,
                "the"
                    | "and"
                    | "for"
                    | "with"
                    | "this"
                    | "that"
                    | "from"
                    | "are"
                    | "was"
                    | "were"
                    | "what"
                    | "which"
                    | "how"
                    | "about"
                    | "into"
                    | "your"
                    | "you"
                    | "can"
                    | "have"
                    | "has"
                    | "does"
                    | "tell"
                    | "please"
                    | "document"
                    | "pdf"
                    | "who"
                    | "why"
                    | "when"
                    | "where"
                    | "did"
                    | "not"
                    | "but"
                    | "all"
                    | "its"
            )
        })
        .map(|word| word.to_string())
        .collect()
}

// ------------------------------------------
// VECTOR NORMALIZATION
// ------------------------------------------

fn normalize(vector: &mut HashMap<String, f64>) {
    let magnitude = vector
        .values()
        .map(|value| value * value)
        .sum::<f64>()
        .sqrt();

    if magnitude > 0.0 {
        for value in vector.values_mut() {
            *value /= magnitude;
        }
    }
}

// ------------------------------------------
// BUILD RAG INDEX
// ------------------------------------------

pub fn build_index(text: &str) -> RagIndex {
    // Split PDF text into 1200-character chunks.
    // Keep 200 characters overlapping between chunks.
    let chunks = chunk_text(text, 1200, 200);

    let total_chunks = chunks.len() as f64;

    let mut document_frequency: HashMap<String, usize> = HashMap::new();

    let mut token_counts = Vec::new();

    // Calculate term frequency for each chunk.
    for chunk in &chunks {
        let mut counts: HashMap<String, usize> = HashMap::new();

        for token in tokenize(&chunk.text) {
            *counts.entry(token).or_insert(0) += 1;
        }

        // Count each term only once per chunk.
        let unique_terms: HashSet<String> = counts.keys().cloned().collect();

        for term in unique_terms {
            *document_frequency.entry(term).or_insert(0) += 1;
        }

        token_counts.push(counts);
    }

    // Calculate inverse document frequency.
    let idf: HashMap<String, f64> = document_frequency
        .into_iter()
        .map(|(term, frequency)| {
            let weight = ((total_chunks + 1.0) / (frequency as f64 + 1.0)).ln() + 1.0;

            (term, weight)
        })
        .collect();

    // Create a normalized TF-IDF vector for every chunk.
    let chunk_vectors: Vec<HashMap<String, f64>> = token_counts
        .into_iter()
        .map(|counts| {
            let mut vector = HashMap::new();

            for (term, frequency) in counts {
                let tf = 1.0 + (frequency as f64).ln();

                let idf_weight = idf.get(&term).copied().unwrap_or(1.0);

                vector.insert(term, tf * idf_weight);
            }

            normalize(&mut vector);

            vector
        })
        .collect();

    println!("RAG index created with {} chunks", chunks.len());

    RagIndex {
        chunks,
        idf,
        chunk_vectors,
    }
}

// ------------------------------------------
// RETRIEVE RELEVANT CHUNKS
// ------------------------------------------

pub fn retrieve(index: &RagIndex, question: &str, top_k: usize) -> Vec<Chunk> {
    if index.chunks.is_empty() || top_k == 0 {
        return Vec::new();
    }

    // Convert question into keyword frequencies.
    let mut query_counts: HashMap<String, usize> = HashMap::new();

    for token in tokenize(question) {
        *query_counts.entry(token).or_insert(0) += 1;
    }

    let mut query_vector = HashMap::new();

    // Apply the same TF-IDF weighting to the question.
    for (term, frequency) in query_counts {
        if let Some(idf_weight) = index.idf.get(&term) {
            let tf = 1.0 + (frequency as f64).ln();

            query_vector.insert(term, tf * idf_weight);
        }
    }

    normalize(&mut query_vector);

    // Calculate cosine similarity between question
    // and each PDF chunk.
    let mut scores: Vec<(usize, f64)> = index
        .chunk_vectors
        .iter()
        .enumerate()
        .map(|(chunk_index, vector)| {
            let score: f64 = query_vector
                .iter()
                .map(|(term, weight)| weight * vector.get(term).copied().unwrap_or(0.0))
                .sum();

            (chunk_index, score)
        })
        .collect();

    // Highest similarity first.
    scores.sort_by(|a, b| b.1.total_cmp(&a.1));

    // If there are no matching keywords, use the first chunks
    // as a fallback instead of returning an empty context.
    let has_match = scores
        .first()
        .map(|(_, score)| *score > 0.0)
        .unwrap_or(false);

    if !has_match {
        return index.chunks.iter().take(top_k).cloned().collect();
    }

    // Return the top relevant chunks.
    scores
        .into_iter()
        .filter(|(_, score)| *score > 0.0)
        .take(top_k)
        .map(|(chunk_index, _)| index.chunks[chunk_index].clone())
        .collect()
}
