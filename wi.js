const fs = require('fs');
const path = require('path');
const axios = require('axios');
const cheerio = require('cheerio');
const readline = require('readline');
const { HfInference } = require('@huggingface/inference');
const { performance } = require('perf_hooks');

const hf = new HfInference('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'); // Replace with your API key
const knowledgeBaseFile = path.join(__dirname, 'knowledge_base.json');
let knowledgeBaseEmbeddings = {};

// Function to scrape data from a URL
async function scrapeWebsite(url) {
    try {
        console.log(`Scraping data from: ${url}`);
        const response = await axios.get(url);
        const $ = cheerio.load(response.data);

        $('script, style, img, link').remove();
        const title = $('title').text().trim();
        const bodyText = $('body').text().replace(/\s+/g, ' ').trim();

        console.log('Scraping completed successfully.');
        return { url, title, content: bodyText };
    } catch (error) {
        console.error(`Error scraping URL ${url}:`, error);
        return null;
    }
}

// Function to save data to the knowledge base
function saveToKnowledgeBase(entry) {
    let knowledgeBase = [];
    if (fs.existsSync(knowledgeBaseFile)) {
        const content = fs.readFileSync(knowledgeBaseFile, 'utf8');
        knowledgeBase = JSON.parse(content);
    }
    knowledgeBase.push({ ...entry, timestamp: new Date().toISOString() });
    fs.writeFileSync(knowledgeBaseFile, JSON.stringify(knowledgeBase, null, 2), 'utf8');
    console.log(`Data from ${entry.url} saved to the knowledge base.`);
}

// Function to generate embeddings using Hugging Face
async function generateEmbeddings(texts) {
    try {
        console.log(`Generating embeddings for texts at: ${new Date().toISOString()}`);
        const response = await hf.featureExtraction({
            inputs: texts,
            model: 'sentence-transformers/all-MiniLM-L12-v2', // More advanced model
        });
        console.log(`Received embeddings response at: ${new Date().toISOString()}`);
        return response; // Return the embedding vectors
    } catch (error) {
        console.error('Error generating embeddings:', error);
        return null;
    }
}

// Function to compute cosine similarity
function cosineSimilarity(vecA, vecB) {

    if (!Array.isArray(vecA) || !Array.isArray(vecB)) 
        throw new TypeError('Embeddings must be arrays');
    const dotProduct = vecA.reduce((sum, a, i) => sum + a * vecB[i], 0);
    const magnitudeA = Math.sqrt(vecA.reduce((sum, a) => sum + a ** 2, 0));
    const magnitudeB = Math.sqrt(vecB.reduce((sum, b) => sum + b ** 2, 0));
    return dotProduct / (magnitudeA * magnitudeB);
}

// Function to query the knowledge base using semantic similarity
async function queryKnowledgeBaseWithLLM(query) {
    if (!fs.existsSync(knowledgeBaseFile)) {
        console.error('Knowledge base file not found. Please scrape a URL first.');
        return 'Knowledge base is empty. Please add some data.';
    }

    const knowledgeBase = JSON.parse(fs.readFileSync(knowledgeBaseFile, 'utf8'));

    // Generate embedding for the query
    const queryEmbedding = await generateEmbeddings([query]);
    if (!queryEmbedding || queryEmbedding.length === 0) {
        return 'Error generating embedding for the query.';
    }

    // Generate embeddings for the knowledge base entries if not already cached
    const textsToEmbed = [];
    const textIndices = [];
    knowledgeBase.forEach((entry, index) => {
        if (entry.content && !knowledgeBaseEmbeddings[entry.url]) {
            textsToEmbed.push(entry.content);
            textIndices.push(index);
        }
    });

    if (textsToEmbed.length > 0) {
        const embeddings = await generateEmbeddings(textsToEmbed);
        if (embeddings) {
            textIndices.forEach((index, i) => {
                knowledgeBaseEmbeddings[knowledgeBase[index].url] = embeddings[i];
            });
        }
    }

    // Compute similarity for each entry
    const similarities = [];
    for (const entry of knowledgeBase) {
        if (!entry.content || !knowledgeBaseEmbeddings[entry.url]) continue; // Skip if content is missing or embedding is not available

        try {
            const score = cosineSimilarity(queryEmbedding[0], knowledgeBaseEmbeddings[entry.url]);
            similarities.push({ entry, score });
        } catch (error) {
            console.error(`Error calculating similarity for URL: ${entry.url}`, error);
        }
    }

    // Sort by similarity and get the top result
    const topResult = similarities
        .sort((a, b) => b.score - a.score)
        .slice(0, 1)
        .map(r => r.entry)[0];

    if (!topResult) {
        return 'No relevant information found in the knowledge base.';
    }

    console.log("Selected knowledge base entry:", topResult.url);

    const context = topResult.content;
    const prompt = `
You are a knowledgeable assistant. Use the context below to answer the user's query in a conversational tone. }

User's Query:
${query}

Your Response:
`;

    try {
        const response = await axios.post('http://localhost:5000/generate', { prompt });
        return response.data.response || "I'm sorry, I couldn't generate a response.";
    } catch (error) {
        console.error('Error querying the LLM:', error);
        return 'Error generating a response. Please try again later.';
    }
}

// Console interface
function startConsoleInterface() {
    const rl = readline.createInterface({ input: process.stdin, output: process.stdout });

    const promptMenu = () => {
        console.log('\nChoose an option:');
        console.log('1. Scrape a URL');
        console.log('2. Query the Knowledge Base');
        console.log('3. Exit');
        rl.prompt();
    };

    console.log('Welcome to the Web Scraper and Knowledge Base!');
    promptMenu();

    rl.on('line', async (input) => {
        const option = input.trim();

        if (option === '1') {
            rl.question('Enter the URL to scrape: ', async (url) => {
                const data = await scrapeWebsite(url);
                if (data) saveToKnowledgeBase(data);
                promptMenu();
            });
        } else if (option === '2') {
            rl.question('Enter your query: ', async (query) => {
                const response = await queryKnowledgeBaseWithLLM(query);
                console.log(`Response: ${response}`);
                promptMenu();
            });
        } else if (option === '3') {
            console.log('Goodbye!');
            rl.close();
        } else {
            console.log('Invalid option. Please choose 1, 2, or 3.');
            promptMenu();
        }
    });

    rl.on('close', () => {
        console.log('Exiting the application.');
        process.exit(0);
    });
}

// Start the console interface
startConsoleInterface();
