# Project goal
Build a retrieval-augmented generation (RAG) system that answers questions about the NIST AI Risk Management Framework (AI RMF 1.0) and provides the supporting source passage from the document.

# Source document and why it was chosen
Source: NIST Artificial Intelligence Risk Management Framework (AI RMF 1.0)
The NIST AI RMF was chosen because it is a public, authoritative, and manageable regulatory/risk-management document related directly to artificial intelligence. Its size makes it appropriate for a first version of a RAG system.

# Five starter questions
1. What is the goal of AI Risk Management Framework?
  a. To provide a resource for organizations designing, developing, deploying, or using AI systems to manage AI risks and promote trustworthy and responsible AI. (p. 1, Executive Summary)

2. What are the characteristics of trustworthy AI systems? 
  a. Valid and reliable; safe; secure and resilient; fair with harmful bias managed; explainable and interpretable; privacy-enhanced; and accountable and transparent. (p. 12, Section 3: AI Risks and Trustworthiness)

3. What are the 4 functions of the AI RMF Core?
  a. Govern, Map, Measure, and Manage AI risks. (p. 21, Section 4: AI RMF Core)

4. Is the AI RMF mandatory? 
  a. No. The AI RMF is intended for voluntary use and provide flexibility to organizations to all sizes. (p. 2, Executive Summary / Section 1)

5. What does the Govern function do? 
  a. It establishes and implements organizational policies and practices for managing AI risks across the organization and AI lifecycle. (p. 23, Section 4, Govern 4.1)

# Version-one boundaries
Input: NIST AI Risk Management Framework (AI RMF 1.0)
Output: An answer with the supporting source passage
Interface: Command line
Not included: User accounts, cloud deployment, elaborate UI, or a large document corpus

# Known limitations
Version one uses only a single document set.
The system may not answer questions that are outside the scope of the source document.
Retrieval quality may be limited by how the document is chunked and indexed.
The system is not intended to replace professional legal or regulatory advice.
