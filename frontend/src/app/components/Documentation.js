"use client";

import Image from "next/image";
import { Button } from "./ui/button";
import { Card } from "./ui/card";
import { Home, BookOpen } from "lucide-react";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "./ui/tabs";
import { ScrollArea } from "./ui/scroll-area";

const motivationContent = `# Motivation

## Why We Built This Project

The modern creative landscape is changing rapidly. Generative AI has opened a new frontier of artistic expression, but it has also blurred the lines between **creation** and **inspiration**, between **innovation** and **appropriation**.
Our project was created to bridge this divide, to bring **transparency**, **credit**, and **respect** to the relationship between human artists and AI-generated art.

At its core, this system allows users to upload or explore AI-generated artworks and discover the **original artists, styles, or influences** that inspired the model's output. By using **CLIP**, we connect visual and semantic understanding to identify stylistic and conceptual similarities between AI-created works and the human art they echo.

---

## The Philosophy

We believe that **art and technology are not in opposition**. They represent two sides of human creativity: one emotional, one analytical.
Artists express what it means to be human. Engineers build tools that expand what humans can do.
Both are essential, and both should thrive.

Technology should **amplify** creativity, not replace it. AI can make art more accessible, help preserve artistic history, and reveal hidden relationships between creative works. But it must also **honor the origins**: the painters, sculptors, photographers, and illustrators whose work continues to shape visual culture.

Our platform is a small step toward that balance: an effort to give **credit where it's due**, while still celebrating the **innovative potential of machine learning**.

---

## The Broader Vision

- To **celebrate art as a human legacy** that informs every new creative system.
- To **encourage ethical AI use**, where creators are acknowledged and valued.
- To **build empathy** between engineers and artists, showing that both are architects of imagination.
- To **educate and inspire**, providing tools that deepen appreciation for art rather than obscuring its origins.

---

## Closing Thought

Human creativity and artificial intelligence can coexist harmoniously when guided by respect and curiosity.
Art reminds us **why** we create.
Technology helps us discover **how far** we can go.

This project stands for both.`;

const architectureContent = `# CLIP-Powered Art Similarity Search

## Overview

This project demonstrates a large-scale multimodal retrieval system built on **OpenAI's CLIP (Contrastive Language–Image Pretraining)** model. It enables semantic and visual search across approximately **90,000 artworks**, each hosted and served through **Cloudflare Images CDN**. The system allows users to upload an image and discover visually or thematically similar works in seconds.

A **React frontend** provides an intuitive interface for image uploads and ranked search results. The backend, built with **FastAPI** and **PostgreSQL (pgvector)**, performs real-time similarity computation using CLIP embeddings.

---

## Core AI Concept: CLIP

CLIP unites visual and textual understanding within a single embedding space. It consists of two encoders: a **Vision Transformer (ViT)** for images and a **Transformer-based text encoder** for natural language. Both produce 512-dimensional normalized vectors that can be directly compared through cosine similarity.

Trained on 400 million (image, text) pairs, CLIP learns a generalizable mapping between images and language, making it ideal for **zero-shot** semantic retrieval.

---

## Dataset

The original dataset used for this project is sourced from the **WikiArt collection**, available publicly through the **Kaggle dataset** [*WikiArt: Visual Art Encyclopedia*](https://www.kaggle.com/datasets/steubk/wikiart).
This dataset contains high-quality metadata and artwork images across a wide range of artistic movements, styles, and historical periods.

---

## Architecture

### Embedding Pipeline

1. **Input**: A dataset of artworks with metadata (artist, title, genre, description).
2. **Image Encoding**: Each artwork is preprocessed using CLIP's vision pipeline and encoded into a normalized vector.
3. **Text Encoding**: Multiple descriptive prompts are generated per artwork to capture semantic richness. Their embeddings are averaged for a stable representation.
4. **Output**: Both image and text embeddings are stored as JSON, later ingested into PostgreSQL with pgvector indexing.

Key detail: averaging across diverse textual prompts reduces bias and improves cross-modal alignment between image and text embeddings.

---

### Search API

A **FastAPI** service handles image uploads and vector retrieval:

- **CLIP Inference**: The uploaded image is embedded in real time.
- **Hybrid Scoring**: Combines image and text similarity via a weighted formula:

\`\`\`python
score = α * visual_similarity + (1 - α) * semantic_similarity
\`\`\`


The default weight α = 0.7 emphasizes visual match while retaining textual coherence.

- **Database Query**: PostgreSQL with **pgvector** performs nearest-neighbor search across 90,000 embedding vectors.
- **Result Delivery**: The API returns a ranked list of artworks, including metadata and direct CDN URLs from Cloudflare Images.

---

## Frontend and CDN Integration

The **React frontend** interfaces with the API, allowing users to drag and drop an image and view instant results. It fetches ranked results with associated **Cloudflare CDN URLs**, ensuring sub-second image loading regardless of geographic region.

Cloudflare's distributed image storage serves approximately **90,000 artworks**, providing both scalability and cost efficiency. This integration enables high-throughput visual exploration without overloading the API server.

---

## System Summary

| Component | Purpose | Key Technology |
|------------|----------|----------------|
| Embedding Generator | Offline encoding of dataset | CLIP (ViT-L/14), PyTorch |
| Database | Vector search | PostgreSQL + pgvector |
| API Service | Real-time similarity computation | FastAPI |
| Frontend | User interaction | React |
| CDN | Image hosting | Cloudflare Images |

---

## Technical Highlights

- **Multimodal Embedding Space**: Shared representation for visual and textual concepts enables cross-domain retrieval.
- **Hybrid Similarity Search**: Weighted blending of image and text features achieves semantic precision beyond raw visual matching.
- **Vector Indexing at Scale**: Efficient nearest-neighbor lookup using pgvector for 90,000 high-dimensional embeddings.
- **Cloud-native Deployment**: Stateless API and CDN integration deliver scalable, latency-minimized responses.

---

## Impact and Innovation

This project illustrates how a pre-trained foundation model can be operationalized into a performant, production-ready retrieval system. By integrating CLIP embeddings, vector databases, and edge-optimized delivery, it bridges research and application. The result is a seamless tool for exploring art through both meaning and appearance.`;

const cnnArchitectureContent = `# CNN-Based AI Art Detector

## Overview

The AI Art Detector is a convolutional neural network (CNN)–based system that determines whether a given artwork was created by an AI model or by a human.

It does this by comparing the embedding of a query (suspected AI) image against a database of human-created artwork embeddings derived from our art dataset.

The database of human art embeddings is stored as high-dimensional vectors, indexed and queried using FAISS (Facebook AI Similarity Search) for efficient nearest-neighbor retrieval.

---

## 1. Components

**Human Art Database:**
Precomputed image embeddings derived from the WikiArt dataset. These embeddings represent authentic, human-created artworks and serve as the reference distribution.

**AI Art Inputs:**
Images generated by AI models (e.g., Stable Diffusion, DALL·E, Midjourney) are encoded into vector form during inference for comparison.

---

## 2. CNN Encoder

The CNN encoder transforms a raw image into a fixed-length feature vector (embedding).

We custom-trained and deployed our own CNN fine-tuned specifically to art style recognition.

---

## 3. Vector Database

The WikiArt embeddings are stored locally in a FAISS index for fast approximate nearest-neighbor (ANN) retrieval.

Each stored vector corresponds to a single artwork and is associated with metadata (artist, title, URL, etc.).

---

## 4. Similarity Computation

When an AI-generated image is processed:

1. The CNN encoder generates its normalized embedding.

2. The embedding is compared to the WikiArt database using cosine similarity.

3. FAISS efficiently retrieves the top _k_ most similar human artworks.

---

## 5. Interpretation

**High similarity score:** The AI image closely resembles existing human artworks (possibly derivative).

**Low similarity score:** Indicates stylistic or structural divergence from human art patterns.`;

const apiContent = `# API Documentation

## Base URL

imagesimilarity.up.railway.app

---

## Endpoints

All endpoints return **JSON** responses.

### 1. \`POST /search\`

Uploads an image and returns the most visually and semantically similar artworks.

#### Description
The endpoint extracts CLIP embeddings from the uploaded image, performs a **hybrid similarity search**, and returns the top matches ranked by combined visual-textual similarity.

#### Request

**Content-Type:** \`multipart/form-data\`

**Parameters**

| Name | Type | Required | Default | Description |
|------|------|-----------|----------|-------------|
| \`file\` | \`UploadFile\` | Yes | — | Image file to search against the dataset |
| \`alpha\` | \`float\` | No | \`0.7\` | Weight balancing image (α) vs. text (1-α) similarity |
| \`top_k\` | \`int\` | No | \`5\` | Number of top results to return |

#### Example Request (cURL)

\`\`\`bash
curl -X POST "https://imagesimilarity.up.railway.app/search" \\
  -F "file=@example.jpg" \\
  -F "alpha=0.7" \\
  -F "top_k=5"
\`\`\`

#### Example Response

\`\`\`json
{
  "results": [
    {
      "rank": 1,
      "artist": "Claude Monet",
      "title": "Water Lilies",
      "year": "1916",
      "style": "Impressionism",
      "filepath": "impressionism/monet_water_lilies.jpg",
      "cdn_url": "https://cdn.cloudflare.com/art/monet_water_lilies.jpg",
      "score": 0.9123
    },
    {
      "rank": 2,
      "artist": "Camille Pissarro",
      "title": "The Boulevard Montmartre",
      "year": "1897",
      "style": "Impressionism",
      "filepath": "impressionism/pissarro_boulevard.jpg",
      "cdn_url": "https://cdn.cloudflare.com/art/pissarro_boulevard.jpg",
      "score": 0.9082
    }
  ]
}
\`\`\``;

export function Documentation({ onBackToHome }) {
  const processInlineMarkdown = (text) => {
    // Process bold text
    let processed = text.replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>");
    // Process links
    processed = processed.replace(
      /\[([^\]]+)\]\(([^)]+)\)/g,
      '<a href="$2" class="text-primary underline" target="_blank" rel="noopener noreferrer">$1</a>'
    );
    // Process inline code
    processed = processed.replace(
      /`([^`]+)`/g,
      '<code class="bg-muted px-1 py-0.5 rounded text-sm">$1</code>'
    );
    return processed;
  };

  const renderMarkdown = (content, isArchitecture = false) => {
    // Simple markdown-to-HTML conversion for basic formatting
    const lines = content.split("\n");
    const elements = [];
    let currentList = [];
    let inCodeBlock = false;
    let codeBlockContent = [];
    let tableRows = [];
    let inTable = false;
    let inOverviewSection = false;
    let overviewDiagramAdded = false;

    lines.forEach((line, idx) => {
      // Code blocks
      if (line.startsWith("```")) {
        if (!inCodeBlock) {
          inCodeBlock = true;
          codeBlockContent = [];
        } else {
          inCodeBlock = false;
          elements.push(
            <pre
              key={idx}
              className="bg-muted p-4 rounded-lg overflow-x-auto my-4"
            >
              <code className="text-sm">{codeBlockContent.join("\n")}</code>
            </pre>
          );
          codeBlockContent = [];
        }
        return;
      }

      if (inCodeBlock) {
        codeBlockContent.push(line);
        return;
      }

      // Flush any pending list
      if (!line.startsWith("-") && currentList.length > 0) {
        elements.push(
          <ul
            key={`list-${idx}`}
            className="list-disc list-inside space-y-2 my-4"
          >
            {currentList.map((item, i) => (
              <li
                key={i}
                className="text-foreground"
                dangerouslySetInnerHTML={{
                  __html: processInlineMarkdown(item),
                }}
              />
            ))}
          </ul>
        );
        currentList = [];
      }

      // Table handling
      if (line.startsWith("|") && line.endsWith("|")) {
        if (!inTable) {
          inTable = true;
          tableRows = [];
        }
        // Skip separator rows (|---|---|)
        if (!line.match(/^\|[\s-|]+\|$/)) {
          const cells = line
            .split("|")
            .slice(1, -1)
            .map((cell) => cell.trim());
          tableRows.push(cells);
        }
        return;
      } else if (inTable) {
        // End of table, render it
        inTable = false;
        if (tableRows.length > 0) {
          elements.push(
            <div key={`table-${idx}`} className="my-6 overflow-x-auto">
              <table className="w-full border-collapse">
                <thead>
                  <tr className="border-b-2 border-border">
                    {tableRows[0].map((header, i) => (
                      <th key={i} className="text-left p-3 bg-muted/50">
                        {header}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {tableRows.slice(1).map((row, rowIdx) => (
                    <tr key={rowIdx} className="border-b border-border">
                      {row.map((cell, cellIdx) => (
                        <td key={cellIdx} className="p-3">
                          {cell}
                        </td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          );
        }
        tableRows = [];
      }

      // Headers
      if (line.startsWith("# ")) {
        elements.push(
          <h1 key={idx} className="mt-8 mb-4">
            {line.replace("# ", "")}
          </h1>
        );
      } else if (line.startsWith("## ")) {
        const headerText = line.replace("## ", "");

        // Track when we enter/exit Overview section
        if (isArchitecture && headerText === "Overview") {
          inOverviewSection = true;
        } else if (inOverviewSection) {
          inOverviewSection = false;
        }

        elements.push(
          <h2 key={idx} className="mt-6 mb-3">
            {headerText}
          </h2>
        );
      } else if (line.startsWith("### ")) {
        elements.push(
          <h3 key={idx} className="mt-4 mb-2">
            {line.replace("### ", "")}
          </h3>
        );
      } else if (line.startsWith("#### ")) {
        elements.push(
          <h4 key={idx} className="mt-3 mb-2">
            {line.replace("#### ", "")}
          </h4>
        );
      }
      // Lists
      else if (line.startsWith("- ")) {
        currentList.push(line.replace("- ", ""));
      }
      // Horizontal rule
      else if (line === "---") {
        // Add architecture diagram before the first horizontal rule after Overview section
        if (isArchitecture && inOverviewSection && !overviewDiagramAdded) {
          elements.push(
            <div key={`${idx}-diagram`} className="my-6 flex justify-center">
              <Image
                src="/architecture_design.png"
                alt="Architecture Diagram"
                width={1200}
                height={800}
                className="w-full max-w-3xl rounded-lg border border-border"
              />
            </div>
          );
          overviewDiagramAdded = true;
        }

        elements.push(<hr key={idx} className="my-8 border-border" />);
      }
      // Regular paragraph
      else if (line.trim()) {
        elements.push(
          <p
            key={idx}
            className="my-3 text-foreground leading-relaxed"
            dangerouslySetInnerHTML={{ __html: processInlineMarkdown(line) }}
          />
        );
      }
    });

    // Flush any remaining list
    if (currentList.length > 0) {
      elements.push(
        <ul key="final-list" className="list-disc list-inside space-y-2 my-4">
          {currentList.map((item, i) => (
            <li
              key={i}
              className="text-foreground"
              dangerouslySetInnerHTML={{ __html: processInlineMarkdown(item) }}
            />
          ))}
        </ul>
      );
    }

    // Flush any remaining table
    if (tableRows.length > 0) {
      elements.push(
        <div key="final-table" className="my-6 overflow-x-auto">
          <table className="w-full border-collapse">
            <thead>
              <tr className="border-b-2 border-border">
                {tableRows[0].map((header, i) => (
                  <th key={i} className="text-left p-3 bg-muted/50">
                    {header}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {tableRows.slice(1).map((row, rowIdx) => (
                <tr key={rowIdx} className="border-b border-border">
                  {row.map((cell, cellIdx) => (
                    <td key={cellIdx} className="p-3">
                      {cell}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      );
    }

    return elements;
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-purple-50 to-blue-50">
      <div className="container mx-auto px-4 py-8 max-w-5xl">
        <div className="flex items-center justify-between mb-8">
          <div className="flex items-center gap-3">
            <BookOpen className="w-8 h-8 text-primary" />
            <h1>Documentation</h1>
          </div>
          <Button onClick={onBackToHome} variant="outline" className="gap-2">
            <Home className="w-4 h-4" />
            Back to App
          </Button>
        </div>

        <Card className="p-6">
          <Tabs defaultValue="motivation" className="w-full">
            <TabsList className="grid w-full grid-cols-4 mb-6">
              <TabsTrigger value="motivation">Motivation</TabsTrigger>
              <TabsTrigger value="architecture">CLIP Architecture</TabsTrigger>
              <TabsTrigger value="cnn">CNN Architecture</TabsTrigger>
              <TabsTrigger value="api">API</TabsTrigger>
            </TabsList>

            <ScrollArea className="h-[calc(100vh-250px)]">
              <TabsContent value="motivation" className="mt-0">
                <div className="prose prose-slate max-w-none">
                  {renderMarkdown(motivationContent, false)}
                </div>
              </TabsContent>

              <TabsContent value="architecture" className="mt-0">
                <div className="prose prose-slate max-w-none">
                  {renderMarkdown(architectureContent, true)}
                </div>
              </TabsContent>

              <TabsContent value="cnn" className="mt-0">
                <div className="prose prose-slate max-w-none">
                  {renderMarkdown(cnnArchitectureContent, false)}
                </div>
              </TabsContent>

              <TabsContent value="api" className="mt-0">
                <div className="prose prose-slate max-w-none">
                  {renderMarkdown(apiContent, false)}
                </div>
              </TabsContent>
            </ScrollArea>
          </Tabs>
        </Card>
      </div>
    </div>
  );
}
