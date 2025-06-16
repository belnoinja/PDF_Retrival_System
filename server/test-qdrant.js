import { QdrantClient } from '@qdrant/js-client-rest';
import { pipeline } from '@xenova/transformers';
import { v4 as uuidv4 } from 'uuid';

// Helper to average nested vectors [1 x N x 384] → [384]
function averagePooling(nestedArray) {
  const vectors = nestedArray[0]; // shape: [N x 384]
  const length = vectors[0].length;
  const pooled = Array(length).fill(0);

  for (const vec of vectors) {
    for (let i = 0; i < length; i++) {
      pooled[i] += vec[i];
    }
  }

  for (let i = 0; i < length; i++) {
    pooled[i] /= vectors.length;
  }

  return pooled;
}

async function run() {
  const COLLECTION_NAME = 'langchainjs-testings';
  const VECTOR_SIZE = 384;

  const client = new QdrantClient({ url: 'http://localhost:6333' });

  try {
    await client.createCollection(COLLECTION_NAME, {
      vectors: { size: VECTOR_SIZE, distance: 'Cosine' }
    });
    console.log('✅ Collection created');
  } catch (e) {
    if (e.message.includes('already exists')) {
      console.log('ℹ️ Collection already exists');
    } else {
      console.error('❌ Collection creation failed:', e);
      return;
    }
  }

  const extractor = await pipeline('feature-extraction', 'Xenova/all-MiniLM-L6-v2');

  const text = 'This is a test vector for Qdrant.';

  const output = await extractor(text);
  const vector = averagePooling(output);

  console.log('✅ Vector length:', vector.length); // ✅ 384

  const point = {
    id: uuidv4(),
    vector,
    payload: { text },
  };

  const result = await client.upsert(COLLECTION_NAME, {
    points: [point],
  });

  console.log('✅ Upsert result:', result);

  const count = await client.count(COLLECTION_NAME);
  console.log('✅ Qdrant now has', count.count, 'points.');
}

run().catch(console.error);
