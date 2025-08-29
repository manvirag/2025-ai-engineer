
### Ai usecases 2025: 

1. Perception
What it does: Extract features from images/videos.
Industries: Healthcare (medical imaging), Automotive (autonomous driving), Security (surveillance).
Tech: CNNs, Vision Transformers (ViT), Image Segmentation (U-Net); Deep Learning (Computer Vision).

2. Speech Recognition
What it does: Convert spoken audio to text.
Industries: Customer Support, Consumer Electronics (voice assistants), Automotive (voice commands).
Tech: RNNs, LSTMs, Transformers, CTC loss; Deep Learning (Speech Processing/NLP).

3. Text Understanding
What it does: Comprehend text intent, entities, sentiment.
Industries: Finance (document analysis), Legal (contract review), Customer Service.
Tech: Transformers (BERT, RoBERTa), Named Entity Recognition (NER); Deep Learning (NLP).

4. Text Generation
What it does: Produce coherent language output.
Industries: Marketing (content creation), Media (summarization), Education (tutoring).
Tech: Autoregressive Transformers (GPT family), Seq2Seq models; Deep Learning (NLP).

5. Knowledge Retrieval
What it does: Retrieve relevant external info for tasks.
Industries: Tech Support, Research, Healthcare.
Tech: Dense vector retrieval with k-NN, embedding models (BERT embeddings), combined with LLMs (RAG); ML + DL (Information Retrieval + NLP).

6. Multimodal Fusion
What it does: Align and integrate multiple data types.
Industries: Retail (visual search), Entertainment (video captioning), Autonomous Systems.
Tech: Multimodal Transformers, Cross-Attention; Deep Learning (Multimodal AI).

7. Prediction
What it does: Forecast or detect anomalies from data.
Industries: Finance (fraud detection), Manufacturing (predictive maintenance), Energy (demand forecasting).
Tech: Regression, Random Forest, Gradient Boosting, LSTM; Machine Learning + Deep Learning (Time-Series Analysis).

8. Decision Making
What it does: Optimize actions/plans based on goals.
Industries: Logistics (route planning), Robotics, Gaming.
Tech: Reinforcement Learning (Q-learning, Policy Gradients), Heuristic Search; Machine Learning (Reinforcement Learning).

9. Generative Content Creation
What it does: Create new images, audio, code, etc.
Industries: Advertising, Software Dev, Arts & Music.
Tech: GANs, Diffusion Models, Autoregressive models (Codex); Deep Learning (Generative Models).

10. Autonomous Agents
What it does: Autonomous perception, reasoning, and action.
Industries: Autonomous Vehicles, Virtual Assistants, Industrial Automation.
Tech: Integration of CNNs, Transformers, RL, Planning Algorithms; AI Systems + ML + DL (Agent-based AI).



## Different section in AI: 

###  Artificial Intelligence (AI)
AI is the broad field of creating intelligent systems that can mimic human behavior.

#### 1. Symbolic AI / Classical AI
- Rule-Based Systems
- Knowledge Graphs
- Expert Systems

#### 2. Machine Learning (ML)
ML is a subset of AI focused on systems that learn from data.

#### 2.1 Learning Paradigms
- **Supervised Learning**
  - Tasks: Regression, Classification
  - Algorithms:
    - Linear Regression
    - Logistic Regression
    - Decision Trees
    - Random Forest
    - Support Vector Machine (SVM)

- **Unsupervised Learning**
  - Tasks: Clustering, Dimensionality Reduction
  - Algorithms:
    - K-Means
    - DBSCAN
    - PCA
    - t-SNE

- **Semi-Supervised Learning**

- **Self-Supervised Learning**

- **Reinforcement Learning (RL)**
  - Algorithms:
    - Q-Learning
    - SARSA
    - Deep Q-Network (DQN)
    - Proximal Policy Optimization (PPO)
    - A3C, DDPG, etc.

#### 2.2 ML Techniques
- **Classical ML** – Uses above algorithms
- **Deep Learning (DL)** – Uses neural networks:
  - Feedforward Neural Networks (FNN / DNN)
  - Convolutional Neural Networks (CNN)
  - Recurrent Neural Networks (RNN)
    - LSTM, GRU
  - Autoencoders (AE, VAE)
  - Generative Adversarial Networks (GANs)
  - Transformers
    - Used in NLP, LLMs, Vision, Speech

---

### Application Domains

#### 3.1 Natural Language Processing (NLP)
Processes and understands human language.

- **Tasks:**
  - Text Classification
  - Named Entity Recognition (NER)
  - Machine Translation
  - Summarization, QA

- **Models:**
  - RNN, LSTM
  - BERT, RoBERTa
  - GPT Series
  - T5, XLNet

- **Includes: Large Language Models (LLMs)**
  - GPT-3, GPT-4, GPT-4o
  - LLaMA, Claude, PaLM
  - Used in chatbots, agents, RAG systems

#### 3.2 Computer Vision (CV)
Processes and understands visual data (images, videos).

- **Tasks:**
  - Image Classification
  - Object Detection
  - Image Segmentation
  - Image Generation

- **Models:**
  - CNNs: VGG, ResNet, EfficientNet
  - Vision Transformers (ViT, DINO)
  - GANs: StyleGAN, CycleGAN

- **Applications:**
  - Facial Recognition, OCR, Medical Imaging

### 3.3 Speech / Audio Processing
Processes audio and speech.

- **Tasks:**
  - ASR (Automatic Speech Recognition)
  - TTS (Text-to-Speech)
  - Speaker Identification

- **Models:**
  - RNNs, CNNs
  - Transformers (e.g., Whisper)
  - WaveNet, Tacotron

#### 3.4 Multimodal AI
Combines multiple input types: text + image + audio + video.

- **Examples:**
  - CLIP (text + image)
  - Whisper (speech + text)
  - Flamingo, GPT-4o, Gemini, Sora

#### 3.5 Retrieval-Augmented Generation (RAG)
Combines LLMs with external data sources.

- **Components:**
  - Embedding Models
  - Vector Databases (e.g., FAISS, Pinecone)
  - LLMs for answer generation

- **Use Cases:**
  - Chat over documents
  - Internal knowledge bots
  - QA over web, PDFs, databases

## NLP vs LLM  (Brief)

- NLP is the science of understanding and working with language.
- LLMs are advanced tools (like ChatGPT) used within NLP to understand and generate text.
- Gen AI is the bigger umbrella that includes LLMs and also tools that make:
  - Images (like DALL·E, Midjourney)
  - Music (like Suno)
  - Videos (like Sora)
  - Code (like GitHub Copilot)
![](image.png)

## NLP Learning:

### Text Preprocessing (Clean Data)
1. **Tokenization** - Split text into units
   - Word: "Hello world!" → ["Hello", "world", "!"]
   - Subword: "unhappiness" → ["un", "happy", "ness"]
   - Character: "cat" → ["c", "a", "t"]
   - Tools: `nltk`, `spacy`, `transformers`

2. **Normalization** - Standardize format
   - Lowercase, remove accents, expand contractions
   - Handle URLs/emails → [URL], [EMAIL]

3. **Stopword Removal** - Filter common words
   - English: the, and, or, but, in, on, at, to, for
   - Keep "not" for sentiment

4. **Stemming vs Lemmatization** - Reduce words to root/base form
   - **Stemming**: Crude chopping (fast, imperfect)
     - "running" → "run" (Porter algorithm)
     - "better" → "better" (unchanged)
     - "studies" → "studi" (incorrect)
     - When: Speed matters, rough matching OK

   - **Lemmatization**: Smart reduction using grammar (slow, accurate)
     - "running" → "run" (verb)
     - "better" → "good" (adjective, comparative)
     - "studies" → "study" (correct noun/verb)
     - "went" → "go" (past → present)
     - When: Accuracy matters, POS tags available

### Embeddings (Text → Numbers)
1. **Word Embeddings** - Convert words to vectors
   - **One-hot**: Sparse vectors, no semantic meaning
     - "cat" = [1,0,0,0], "dog" = [0,1,0,0], "car" = [0,0,1,0]
     - Problem: No relationship between words, huge vectors

   - **Word2Vec**: Dense vectors, learns word relationships
     - "king" - "man" + "woman" ≈ "queen" (famous example)
     - "Paris" - "France" + "Italy" ≈ "Rome"
     - Uses context windows to learn meanings

   - **GloVe**: Global co-occurrence statistics
     - Counts word pairs in large corpus
     - "ice" and "cold" appear together often
     - Captures global word relationships

   - **FastText**: Handles out-of-vocabulary words
     - Breaks words into character n-grams
     - "unhappiness" → "un", "nh", "ha", "ap", "pp", "pi", "in", "ne", "es", "ss"
     - Can create vectors for unseen words

2. **Sentence Embeddings** - Convert full sentences to vectors
   - **Average word vectors**: Simple but loses word order
     - Take all word vectors in sentence, average them
     - "I love cats" → average of [I, love, cats] vectors
     - Fast but ignores sentence structure

   - **Doc2Vec**: Learns document-level representations
     - Like Word2Vec but for entire documents
     - Captures document-level context
     - Good for document classification

   - **BERT/RoBERTa**: Contextual sentence embeddings
     - Uses transformer architecture
     - "[CLS]" token represents whole sentence
     - Bidirectional context understanding
     - "I love cats" vs "Cats love I" = different vectors

   - **Sentence-BERT**: Optimized for sentence similarity
     - Fine-tuned BERT for sentence tasks
     - Better at semantic similarity than base BERT
     - Faster and more accurate for sentence comparison

3. **Similarity Measures**
   - Cosine similarity (-1 to 1)
   - Dot product (faster)
   - Euclidean distance

### Transformers & Attention
1. **What is Attention?** - AI's way of focusing on important information

   **Think of it like this**: Imagine you're reading a book and need to understand what "cat" means in this sentence. Your brain automatically looks at the words around it to get context.

   **How Attention Works (Simple Version)**:
   - **Step 1**: Current word (like "cat") asks "What should I focus on?"
   - **Step 2**: All other words answer "I can help!" or "Not really"
   - **Step 3**: "cat" listens more to helpful words, less to others
   - **Step 4**: "cat" combines information from helpful words

   **Everyday Example**: "I love my fluffy cat"
   - Word "cat" focuses most on "fluffy" (description)
   - Also pays attention to "my" (ownership) and "love" (emotion)
   - Ignores "I" (not very helpful for understanding "cat")

   **Where is Attention Used?**
   - **ChatGPT/Chatbots**: Understanding what you mean
   - **Google Translate**: Connecting words in different languages
   - **Voice Assistants**: Figuring out your commands
   - **Image Recognition**: Finding important parts of pictures

   **Why Beginners Should Care**:
   - Attention helps AI understand relationships between words
   - Without attention, AI treats each word separately (like a dictionary)
   - With attention, AI understands context (like a human reader)

   **Simple Attention in Action**:
   ```
   Sentence: "The big black cat sat on the red mat"

   Word "cat" pays attention to:
   - "big" and "black" (what it looks like) - HIGH attention
   - "sat" (what it did) - HIGH attention
   - "The" and "on" (connecting words) - MEDIUM attention
   - "red" and "mat" (location info) - LOW attention
   ```

   **The Magic**: Attention lets AI read like humans do - connecting ideas instead of just seeing individual words!

2. **Multi-Head Attention** - Multiple ways of paying attention at once

   **Simple Explanation**: Instead of looking at things one way, AI looks at the same information from 8 different angles, like having multiple cameras filming the same scene.

   **Why Multiple Heads?**
   - Language has many types of relationships between words
   - One head alone might miss important connections
   - Different heads catch different patterns

   **8 Different Ways AI Pays Attention**:
   - **Head 1**: Who did what? ("cat" → "sat")
   - **Head 2**: What describes what? ("fluffy" → "cat")
   - **Head 3**: What happened where? ("sat" → "mat")
   - **Head 4**: Who owns what? ("my" → "cat")
   - **Head 5**: How did it happen? ("quickly" → "jumped")
   - **Head 6**: What colors/things? ("black" → "cat")
   - **Head 7**: Connect distant ideas ("The" → "mat")
   - **Head 8**: What qualities? ("soft" → "mat")

   **Real Example**: "My fluffy black cat quickly jumped onto the soft red mat"
   ```
   Head 1 sees: cat → jumped (the main action)
   Head 2 sees: fluffy + black → cat (description)
   Head 3 sees: jumped → mat (location)
   Head 4 sees: My → cat (ownership)
   All heads work together for complete understanding!
   ```

   **The Result**: AI gets a full picture of relationships, just like how humans understand sentences from multiple angles.

3. **BERT vs GPT** - Two different AI reading styles

   **Simple Comparison**:

   **BERT (Reading Expert)**:
   - **Reads**: The entire sentence at once (sees past AND future words)
   - **Like**: Reading a whole book first, then answering questions about it
   - **Best for**: Understanding existing text, finding facts
   - **Example**: "The fluffy [BLANK] jumped" → predicts "cat" (uses all context)

   **GPT (Writing Expert)**:
   - **Reads**: One word at a time, left to right (only past words)
   - **Like**: Writing a story one word at a time, predicting what comes next
   - **Best for**: Creating new text, conversations
   - **Example**: "The fluffy cat" → predicts "sat" (logical next word)

   **Key Difference**:
   - **BERT**: "I understand the whole story" (sees everything at once)
   - **GPT**: "I write the next word" (builds sequentially)

   **Real Use Cases**:
   - **BERT**: Search engines, fact-checking, document analysis
   - **GPT**: Chatbots, content creation, writing assistance
   - **Both**: Modern AI systems combine both for best results

### Semantic Search
1. **Traditional vs Semantic**
   - Keyword: exact matches only
   - Semantic: understands meaning
   - Hybrid: BM25 + dense vectors

2. **Use Cases**
   - Medical: "chest pain" → "cardiac discomfort"
   - Legal: "contract breach" → "agreement violation"

### Vector Databases
1. **Why Special?** - Fast similarity search
   - PostgreSQL: slow full scan
   - Vector DB: HNSW, IVF indexes

2. **Popular Options**
   - FAISS: local, fastest
   - Pinecone: cloud, managed
   - Chroma: Python-friendly
   - Weaviate: GraphQL, ML models
   - Qdrant: Rust, fast filtering

3. **Index Types**
   - Flat: exact, slow
   - HNSW: hierarchical, fast approx
   - IVF: inverted file, large datasets

### Chunking Strategies
1. **Why Important?** - LLM context limits (4k-128k tokens)

2. **Methods**
   - Fixed-size: simple, crude
   - Sentence-aware: preserves meaning
   - Paragraph-based: natural structure
   - Semantic: topic changes
   - Recursive: sentences → paragraphs → fixed

3. **Overlap Strategy** - 10-20% overlap preserves context

### Prompt Engineering
1. **Basic Principles**
   - Be specific, provide context
   - Use examples, set constraints

2. **RAG-Specific**
   - System prompt, context injection
   - Answer constraints: "Based only on context"

3. **Advanced Techniques**
   - Chain-of-thought, self-consistency
   - Few-shot, role prompting

### Key Remember Points
- **Retrieval Scoring**: BM25 + dense vectors = hybrid search
- **Context Management**: Token limits (4k, 8k, 128k), smart truncation
- **Evaluation**: RAGAS (faithfulness, relevance), A/B testing
- **Advanced**: Query rewriting, multi-step retrieval, metadata filtering

Start building now!


## RAG Deep dive:

### RAG Deep Dive 2025 (Complete Learning Path)

![](/arch.png)
1. General Overview
    - Ingestion pipeline ( put documents in db).
      - chunks
      - embeddings
      - insert in db. 
    - Retrieval 
      - question 
      - chunk
      - embedding generation.
      - find near by embeddings, topk type.
    - Generation
      - send nearby embeddings to llm and questions.
      - generate response. 
 ![](/rag.png)
2. Optimisation: ( Advance RAG )
    - Query translation
      - multiple query
      - query fusion
      - etc.
    - routing
    - query contruction
    - indexing

Ref for later and deep diving: 
- pdf attached.
- https://github.com/langchain-ai/rag-from-scratch?tab=readme-ov-file
- https://mallahyari.github.io/rag-ebook/04_advanced_rag.html
- https://www.coursera.org/learn/retrieval-augmented-generation-rag



## AI agents deep dive:

## MCP server:

## High level understanding of multi modal:

## LLM architecture deep dive:


