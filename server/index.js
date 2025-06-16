import express from 'express';
import cors from 'cors';
import multer from 'multer';
import dotenv from 'dotenv';
import fetch from 'node-fetch';
import fs from 'fs';
import path from 'path';
import { Queue } from 'bullmq';
import { QdrantVectorStore } from '@langchain/qdrant';
import { HuggingFaceInferenceEmbeddings } from '@langchain/community/embeddings/hf';

dotenv.config();

const app = express();
app.use(cors());
app.use(express.json());

const uploadsDir = path.join(process.cwd(), 'uploads');
if (!fs.existsSync(uploadsDir)) {
  fs.mkdirSync(uploadsDir);
}

const queue = new Queue('file-upload-queue', {
  connection: {
  host: process.env.REDIS_HOST,
  port: parseInt(process.env.REDIS_PORT || '6379'),
   username: 'default',
  password: process.env.REDIS_PASSWORD,
  tls: {}  // Enables TLS for Upstash (rediss://)
}

});

const storage = multer.diskStorage({
  destination: function (req, file, cb) {
    cb(null, uploadsDir);
  },
  filename: function (req, file, cb) {
    const uniqueSuffix = Date.now() + '-' + Math.round(Math.random() * 1e9);
    cb(null, `${uniqueSuffix}-${file.originalname}`);
  },
});
const upload = multer({ storage });

app.get('/', (req, res) => {
  res.json({ status: 'All Good!' });
});

app.post('/upload/pdf', upload.single('pdf'), async (req, res) => {
  if (!req.file) return res.status(400).json({ error: 'No file uploaded' });

  await queue.add('file-ready', {
    filename: req.file.originalname,
    destination: req.file.destination,
    path: req.file.path,
  });

  return res.json({ message: 'uploaded' });
});

app.get('/chat', async (req, res) => {
  const userQuery = req.query.message;
  if (!userQuery) return res.status(400).json({ error: 'Missing user query' });

  const embeddings = new HuggingFaceInferenceEmbeddings({
    apiKey: process.env.HF_TOKEN,
    model: 'sentence-transformers/all-mpnet-base-v2',
    maxRetries: 5,
    config: { timeout: 40000 },
  });

  const vectorStore = await QdrantVectorStore.fromExistingCollection(embeddings, {
    url: process.env.QDRANT_URL || 'http://localhost:6333',
    apiKey: process.env.QDRANT_API_KEY,
    collectionName: 'langchainjs-testing',
  });

  const retriever = vectorStore.asRetriever({ k: 2 });
  const docs = await retriever.invoke(userQuery);

  const context = docs.map((d, i) => `Context ${i + 1}:\n${d.pageContent}`).join('\n\n');

  const prompt = `You are a helpful AI Assistant. Use the following context to answer the user's query.\n\n${context}\n\nUser: ${userQuery}\n\nAssistant:`;

  const hfResponse = await fetch(
    'https://api-inference.huggingface.co/models/mistralai/Mixtral-8x7B-Instruct-v0.1',
    {
      method: 'POST',
      headers: {
        Authorization: `Bearer ${process.env.HF_TOKEN}`,
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        inputs: prompt,
        parameters: {
          temperature: 0.8,
          top_p: 0.7,
          max_new_tokens: 300,
        },
      }),
    }
  );

  if (!hfResponse.ok) {
    const error = await hfResponse.text();
    return res.status(500).json({ error: 'Hugging Face API failed', details: error });
  }

  const data = await hfResponse.json();
  const message =
    data[0]?.generated_text?.split('Assistant:')[1]?.trim() ||
    data[0]?.generated_text ||
    'No response';

  return res.json({ message, docs });
});

const PORT = process.env.PORT || 8000;
app.listen(PORT, () => console.log(`🚀 Server running on port ${PORT}`));
