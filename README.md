# Financial-chatbot-RAG-Powered
A RAG-based chatbot for financial analysis of Microsoft, Apple, and Tesla.

Financial Analysis Chatbot
An AI-powered chatbot for financial insights on Microsoft, Apple, and Tesla using RAG (Retrieval-Augmented Generation).

🔹 Overview
This chatbot extracts financial data from 10-K filings and provides concise, easy-to-understand insights. It uses:
✅ OpenAI’s LLM for generating human-like responses
✅ LangChain embeddings & FAISS for efficient retrieval
✅ Contextualized history retriever to maintain conversation flow

 Features
✔️ Summarized Financial Insights – Provides key financial metrics such as revenue, net income, and assets
✔️ Interactive Dialogue – Suggests follow-up questions for deeper exploration
✔️ Conversational Memory – Maintains session history for context-aware responses
✔️ Data Retrieval from SEC Filings – Extracts insights from financial reports
✔️ Future-Ready – Can be expanded with real-time stock market integration


 Example Queries
💬 User: "What was Tesla’s revenue in 2023?"
🤖 Bot: "Tesla's revenue in 2023 was $96.7 billion, reflecting a strong year-over-year growth of 18.8%."

💬 User: "How does Microsoft’s revenue compare over the last 3 years?"
🤖 Bot: "Microsoft showed a steady increase in revenue: $198B (2022), $211B (2023), and $245B (2024), with the highest growth in 2024 at 15.7%."

💬 User: "Can you summarize Apple’s financial performance?"
🤖 Bot: "Apple’s revenue declined slightly in 2023 but rebounded in 2024. Net income saw a marginal drop, and total assets increased, indicating stable financial health."

Limitations
  . No Real-Time Data – Only answers based on uploaded financial reports
  . No Graphs – Describes visualization suggestions but does not generate charts
  . Limited Companies – Currently supports Microsoft, Apple, and Tesla only
