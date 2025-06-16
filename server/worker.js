import { Worker } from 'bullmq';
import { HuggingFaceInferenceEmbeddings } from '@langchain/community/embeddings/hf';
import { QdrantVectorStore } from '@langchain/qdrant';
import { PDFLoader } from '@langchain/community/document_loaders/fs/pdf';
import { CharacterTextSplitter } from '@langchain/textsplitters';
import dotenv from 'dotenv';

dotenv.config();

const worker = new Worker(
  'file-upload-queue',
  async (job) => {
    try {
      console.log(`📥 Received job:`, job.data);

      const data = typeof job.data === 'string' ? JSON.parse(job.data) : job.data;

      if (!data.path) {
        throw new Error('Missing file path in job data');
      }

      // Load the PDF
      const loader = new PDFLoader(data.path);
      const rawDocs = await loader.load();

      // Split into chunks
      const splitter = new CharacterTextSplitter({
        separator: '\n',
        chunkSize: 800,
        chunkOverlap: 100,
      });
      const docs = await splitter.splitDocuments(rawDocs);

      // Create embeddings
      const embeddings = new HuggingFaceInferenceEmbeddings({
        apiKey: process.env.HF_TOKEN,
        model: 'sentence-transformers/all-mpnet-base-v2',
        maxRetries: 3,
        config: {
          timeout: 30000,
        },
      });

      const vectorStore = await QdrantVectorStore.fromExistingCollection(embeddings, {
        url: process.env.QDRANT_URL || 'http://localhost:6333',
        apiKey: process.env.QDRANT_API_KEY,
        collectionName: 'langchainjs-testing',
      });

      await vectorStore.addDocuments(docs);

      console.log(`✅ Successfully added ${docs.length} chunks to vector store`);
    } catch (err) {
      console.error('❌ Job failed:', err);
    }
  },
  {
    concurrency: 100,
    connection: {
      host: process.env.REDIS_HOST || 'localhost',
      port: parseInt(process.env.REDIS_PORT || '6379'),
      password: process.env.REDIS_PASSWORD,
    },
  }
);
