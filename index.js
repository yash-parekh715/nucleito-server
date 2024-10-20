import express from "express";
import { ChatGoogleGenerativeAI } from "@langchain/google-genai";
import * as dotenv from "dotenv";
import { Document } from "@langchain/core/documents";
import { PromptTemplate } from "@langchain/core/prompts";
import {
  ChatPromptTemplate,
  MessagesPlaceholder,
} from "@langchain/core/prompts";
import { createStuffDocumentsChain } from "langchain/chains/combine_documents";
import { CheerioWebBaseLoader } from "langchain/document_loaders/web/cheerio";
import { load } from "cheerio";
import { GoogleGenerativeAIEmbeddings } from "@langchain/google-genai";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import { MemoryVectorStore } from "langchain/vectorstores/memory";
import { createRetrievalChain } from "langchain/chains/retrieval";
import { Pinecone } from "@pinecone-database/pinecone";
import { PineconeStore } from "@langchain/pinecone";
import { RunnableWithMessageHistory } from "@langchain/core/runnables";
import { UpstashRedisChatMessageHistory } from "@langchain/community/stores/message/upstash_redis";
import { StringOutputParser } from "@langchain/core/output_parsers";
import { TextLoader } from "langchain/document_loaders/fs/text";
const cors = require("cors");

dotenv.config();

const app = express();
app.use(express.json());
app.use(cors());

async function loadTxt() {
  const files = [
    "nuclear_energy_and_the_environment.txt",
    "N4C_info.txt",
    "net_zero_needs_nuclear.txt",
  ];

  // Use Promise.all to load all the files asynchronously
  const loaders = files.map((file) => new TextLoader(file).load());

  // Wait for all the loaders to complete and flatten the result
  const docs = (await Promise.all(loaders)).flat();

  return docs;
}

async function main(input) {
  let answer;
  //pinecone
  const pinecone = new Pinecone({
    apiKey: process.env.PINECONE_API_KEY,
  });

  const pineconeIndex = pinecone.Index(
    process.env.PINECONE_INDEX,
    process.env.PINECONE_URL
  );

  //Define LLM
  const llm = new ChatGoogleGenerativeAI({
    apiKey: process.env.API_KEY,
    temperature: 0.7,
  });

  //Prompt Template
  const prompt = ChatPromptTemplate.fromMessages([
    [
      "system",
      `You are a specialized assistant focused ONLY on nuclear energy topics and your name is Nucleito.

        IMPORTANT RULES:
        1. ONLY answer questions related to nuclear energy and associated topics
        2. For any question not related to nuclear energy, respond with: "I can only answer questions about nuclear energy and related topics. Please rephrase your question to focus on nuclear energy."
        3. Base your answers strictly on the provided context
        4. If the question is about nuclear energy but the answer isn't in the context, say: "While this is related to nuclear energy, I don't have specific information about this in my current context."

        context: {context}`,
    ],
    new MessagesPlaceholder("history"),
    ["human", "{input}"],
  ]);
  console.log(prompt);

  // Chaining prompt and LLM
  const chain = await createStuffDocumentsChain({
    llm,
    prompt,
  });

  const chainwithHistory = new RunnableWithMessageHistory({
    runnable: chain,
    getMessageHistory: (sessionId) =>
      new UpstashRedisChatMessageHistory({
        sessionId,
        config: {
          url: process.env.UPSTASH_REDIS_REST_URL,
          token: process.env.UPSTASH_REDIS_REST_TOKEN,
        },
      }),
    inputMessagesKey: "input",
    historyMessagesKey: "history",
  });
  //Website loader
  // const loader = new CheerioWebBaseLoader(
  //     'https://js.langchain.com/v0.2/docs/tutorials/local_rag/'
  // );
  const docs = await loadTxt();
  // console.log(docs)

  //Splitting Text
  const textSplitter = new RecursiveCharacterTextSplitter({
    chunkSize: 200,
    chunkOverlap: 10,
  });

  const splitDocs = await textSplitter.splitDocuments(docs);
  // console.log(splitDocs[0])

  //embeddings
  const embeddings = new GoogleGenerativeAIEmbeddings({
    apiKey: process.env.API_KEY,
    modelName: "embedding-001",
  });

  //////////////Vector store///////////////////
  // const vectorStore2 = await MemoryVectorStore.fromDocuments(
  //     splitDocs,
  //     embeddings
  // );
  // console.log(vectorStore2)
  // //retrieve from the store based on similarity
  // const retriever = vectorStore.asRetriever({
  //     k:2
  // });
  //////////////Vector store///////////////////

  const namespace = `testing`;

  console.log("Starting document addition...");

  // let store
  async function addDocs() {
    try {
      store = await PineconeStore.fromDocuments(splitDocs, embeddings, {
        pineconeIndex: pineconeIndex,
        maxConcurrency: 5,
        namespace: namespace,
      });
      console.log("compke", store);
    } catch (error) {
      console.log(error);
    }
  }
  // await addDocs()
  //Pinecone
  async function stats() {
    const stats = await pineconeIndex.describeIndexStats();
    console.log(stats);
  }
  stats();

  const pineconeStore = new PineconeStore(embeddings, {
    namespace: namespace,
    pineconeIndex: pineconeIndex,
  });
  // const query = await embeddings.embedQuery("pinecone")
  // console.log(query)
  try {
    async function search() {
      const result = await pineconeStore.similaritySearch(
        "what is standard data",
        3
      );
      if (!result) {
        console.log("done");
      }
      // console.log("result",result)
    }
    search();
  } catch (error) {
    console.log(error);
  }
  console.log("here");

  async function retreiver() {
    const retriever = pineconeStore.asRetriever({
      k: 3,
      searchType: "similarity",
    });

    console.log("here retreiver", await retreiver);

    const retrievalChain = await createRetrievalChain({
      combineDocsChain: chainwithHistory,
      retriever: retriever,
    });
    await fetch(retrievalChain);
  }

  await retreiver();

  // const response = await retrievalChain.invoke(
  // {
  //     input:"Hello my name is Kush,What about sessionId? do i need to create sessionId for every user",
  // },
  // {
  //     configurable:{
  //         sessionId:"foobarbaz"
  //     }
  // })
  async function fetch(retrievalChain) {
    const fUresponse = await retrievalChain.invoke(
      {
        input,
      },
      {
        configurable: {
          sessionId: "abcdefghijklmnopqrstuvwxyz",
        },
      }
    );

    // console.log(response.answer)
    answer = await fUresponse.answer;
    console.log("resposne", await fUresponse.answer);
  }
  return answer;
}

app.post("/chat", async (req, res) => {
  const { input } = req.body;
  const ans = await main(input);

  res.json({ ans: ans });
});
app.get("/test", async (req, res) => {
  res.json({ dsa: "asds" });
});

const port = process.env.PORT || 3000;

app.listen(port, () => {
  console.log(`server listening on port ${port}`);
});
